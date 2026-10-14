"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

# Helpers used by the describe_vector_index, add_records and vector_search MCP
# tools. They work with any OpenSearch index with knn_vector fields: the vector
# fields, their dimension and space type are read from the index mapping, so the
# LLM never has to build or pass embedding vectors itself.

from typing import Any, Iterator, Optional

from opensearchpy import OpenSearch

from .embeddings import StratioLitellmGatewayEmbeddings

# Key of the pointer an LLM puts in a record instead of a vector, e.g.
# {"text": "...", "embedding": {"$embed": "text"}}. The value is the field (or list
# of fields, joined with newlines) whose text is embedded to fill the vector field.
EMBED_POINTER_KEY = "$embed"


def _walk_properties(
    properties: dict[str, Any], prefix: str = ""
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yields (dotted path, mapping) for every field of an index mapping."""
    for name, spec in properties.items():
        path = f"{prefix}{name}"
        if "properties" in spec:
            yield from _walk_properties(spec["properties"], f"{path}.")
        else:
            yield path, spec


def get_index_fields(client: OpenSearch, index: str) -> dict[str, dict[str, Any]]:
    """Returns the fields of an index (or of every index matching a pattern or alias).

    knn_vector fields include their dimension, space_type and engine.

    :raises ValueError: if a field has different types or dimensions across indices.
    """
    mappings = client.indices.get_mapping(index=index)
    fields: dict[str, dict[str, Any]] = {}
    for index_mapping in mappings.values():
        properties = index_mapping.get("mappings", {}).get("properties", {})
        for path, spec in _walk_properties(properties):
            field: dict[str, Any] = {"type": spec.get("type", "object")}
            if field["type"] == "knn_vector":
                method = spec.get("method", {})
                field["dimension"] = spec.get("dimension")
                field["space_type"] = method.get("space_type") or spec.get("space_type")
                field["engine"] = method.get("engine")
            previous = fields.setdefault(path, field)
            if previous != field:
                raise ValueError(
                    f"Field '{path}' has different mappings in the indices matching "
                    f"'{index}': {previous} and {field}"
                )
    return fields


def knn_vector_fields(fields: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Filters the knn_vector fields of the output of get_index_fields."""
    return {
        path: field for path, field in fields.items() if field["type"] == "knn_vector"
    }


def _get_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _is_embed_pointer(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {EMBED_POINTER_KEY}


def collect_embed_requests(
    records: list[dict[str, Any]], vector_fields: dict[str, dict[str, Any]]
) -> list[tuple[int, str, str]]:
    """Finds the embed pointers of the records.

    :param records: The records to index, possibly with {"$embed": ...} pointers in
        their top-level fields.
    :param vector_fields: The knn_vector fields of the target index.
    :return: One (record position, vector field, text to embed) per pointer.
    :raises ValueError: if a pointer is placed in a field that is not a knn_vector
        or references a missing or empty text field.
    """
    requests = []
    for position, record in enumerate(records):
        for field, value in record.items():
            if not _is_embed_pointer(value):
                continue
            if field not in vector_fields:
                raise ValueError(
                    f"Record {position}: '{field}' is not a knn_vector field of the "
                    f"index. Vector fields: {sorted(vector_fields)}"
                )
            sources = value[EMBED_POINTER_KEY]
            if isinstance(sources, str):
                sources = [sources]
            texts = []
            for source in sources:
                text = _get_path(record, source)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Record {position}: field '{source}' referenced by "
                        f"'{field}' is missing or is not a non-empty text"
                    )
                texts.append(text)
            requests.append((position, field, "\n".join(texts)))
    return requests


def embed_for_field(
    embeddings: StratioLitellmGatewayEmbeddings,
    texts: list[str],
    dimension: Optional[int],
    headers: dict[str, str],
) -> list[list[float]]:
    """Embeds the texts with the dimension of a knn_vector field.

    The dimensions parameter is only sent when the field dimension differs from the
    model's native one, so models that do not support it keep working.

    :raises ValueError: if the vectors do not match the field dimension.
    """
    dimensions = None
    if dimension is not None and dimension != embeddings.native_dimensions(headers):
        dimensions = dimension
    vectors = embeddings.embed_documents(texts, dimensions=dimensions, headers=headers)
    if dimension is not None and vectors and len(vectors[0]) != dimension:
        raise ValueError(
            f"The embedding model '{embeddings.model}' returned vectors of "
            f"{len(vectors[0])} dimensions but the field expects {dimension}. Use "
            "an index created with the same embedding model."
        )
    return vectors


def build_bulk_body(index: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Builds a bulk request that indexes the records, using their '_id' key as id."""
    body: list[dict[str, Any]] = []
    for record in records:
        document = dict(record)
        action: dict[str, Any] = {"_index": index}
        record_id = document.pop("_id", None)
        if record_id is not None:
            action["_id"] = str(record_id)
        body.extend([{"index": action}, document])
    return body


def summarize_bulk_response(
    response: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Returns the ids of the indexed records and the failures (with position)."""
    ids, failures = [], []
    for position, item in enumerate(response.get("items", [])):
        result = item.get("index", {})
        if "error" in result:
            failures.append(
                {
                    "position": position,
                    "id": result.get("_id"),
                    "error": result["error"],
                }
            )
        else:
            ids.append(result.get("_id"))
    return ids, failures


def build_knn_query(
    vector_field: str,
    vector: list[float],
    k: int,
    min_score: Optional[float] = None,
    filter_query: Optional[dict[str, Any]] = None,
    source_fields: Optional[list[str]] = None,
    excluded_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Builds a k-NN search request body.

    :param excluded_fields: Fields left out of _source, i.e. the vectors, which are
        useless and huge for an LLM.
    """
    knn: dict[str, Any] = {"vector": vector, "k": k}
    if filter_query:
        # Efficient filtering, supported by the lucene and faiss engines
        knn["filter"] = filter_query
    body: dict[str, Any] = {
        "size": k,
        "query": {"knn": {vector_field: knn}},
        "_source": {
            "includes": source_fields or ["*"],
            "excludes": excluded_fields or [],
        },
    }
    if min_score is not None:
        body["min_score"] = min_score
    return body


def hits_to_chunks(
    hits: list[dict[str, Any]], text_field: Optional[str] = None
) -> list[dict[str, Any]]:
    """Converts search hits into chunks: id, index, score, text and metadata."""
    chunks = []
    for hit in hits:
        source = dict(hit.get("_source", {}))
        chunks.append(
            {
                "id": hit.get("_id"),
                "index": hit.get("_index"),
                "score": hit.get("_score"),
                "text": source.pop(text_field, None) if text_field else None,
                "metadata": source,
            }
        )
    return chunks
