"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional

from fastmcp import FastMCP
from genai_core.constants.constants import CHAIN_KEY_GENAI_AUTH, CHAIN_KEY_USER_REQUEST
from genai_core.logger.logger import log
from genai_core.model.mcp_models import McpCheckpointToolResponse, McpToolResponse
from genai_core.runnables.genai_auth import (
    MCP_READ_ROLES,
    MCP_WRITE_ROLES,
    GenAiAuthRunnable,
)
from genai_core.runnables.genai_metadata import (
    GenAiMetadata,
    mcp_metadata_to_runnable_config,
)
from genai_core.runnables.mcp_tool_wrapper import mcp_tool_wrapper

from .constants.constants import (
    ADD_RECORDS_MAX_RECORDS,
    CHAIN_KEY_USER_NAME,
    VECTOR_SEARCH_MAX_K,
)
from .vector_index import (
    build_bulk_body,
    build_knn_query,
    collect_embed_requests,
    embed_for_field,
    get_index_fields,
    hits_to_chunks,
    knn_vector_fields,
    summarize_bulk_response,
)

if TYPE_CHECKING:
    from .chain import SimpleMcpChain


class GetCurrentTimeResponse(McpToolResponse):
    """Response for the get_current_time MCP tool."""

    time: Optional[str] = None
    timezone: Optional[str] = None


class FormatTextResponse(McpToolResponse):
    """Response for the format_text MCP tool."""

    result: Optional[str] = None
    style: Optional[str] = None


class AnalyzeTextResponse(McpToolResponse):
    """Response for the analyze_text MCP tool."""

    text: Optional[str] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None


class MadHatterResponse(McpCheckpointToolResponse):
    """Response for the mad_hatter MCP tool.

    Extends McpCheckpointToolResponse (instead of McpToolResponse) because the
    tool calls an LLM and can be slow: if the server has the checkpoint-timeout
    fallback enabled, a task_id is returned and the client polls for the result.
    Note that the base class already defines a ``message`` field (progress
    message), so the actor's explanation is exposed as ``riddle_explanation``.
    """

    user_request_explanation: Optional[str] = None
    riddle: Optional[str] = None
    riddle_explanation: Optional[str] = None


class RunOpenSearchQueryResponse(McpToolResponse):
    """Response for the run_opensearch_query MCP tool."""

    index: Optional[str] = None
    total_hits: Optional[int] = None
    hits: Optional[list[dict[str, Any]]] = None
    aggregations: Optional[dict[str, Any]] = None


class GenerateEmbeddingsResponse(McpToolResponse):
    """Response for the generate_embeddings MCP tool."""

    model: Optional[str] = None
    dimensions: Optional[int] = None
    embeddings: Optional[list[list[float]]] = None


class CreateIndexResponse(McpToolResponse):
    """Response for the create_index MCP tool."""

    index: Optional[str] = None
    acknowledged: Optional[bool] = None


class DescribeVectorIndexResponse(McpToolResponse):
    """Response for the describe_vector_index MCP tool."""

    index: Optional[str] = None
    vector_fields: Optional[dict[str, dict[str, Any]]] = None
    fields: Optional[dict[str, str]] = None
    embedding_model: Optional[str] = None


class AddRecordsResponse(McpCheckpointToolResponse):
    """Response for the add_records MCP tool.

    Extends McpCheckpointToolResponse because embedding many records can be slow.
    """

    index: Optional[str] = None
    indexed: Optional[int] = None
    ids: Optional[list[str]] = None
    failures: Optional[list[dict[str, Any]]] = None


class VectorSearchResponse(McpToolResponse):
    """Response for the vector_search MCP tool."""

    index: Optional[str] = None
    vector_field: Optional[str] = None
    chunks: Optional[list[dict[str, Any]]] = None


def configure_mcp_tools(mcp: FastMCP, chain: "SimpleMcpChain") -> None:
    """Register all MCP tools on the given FastMCP instance.

    Args:
        mcp: The FastMCP server instance to register tools on.
        chain: The SimpleMcpChain instance used by analyze_text to delegate
            to the chain's invoke logic, by mad_hatter to reach its actor, by
            the OpenSearch tools to reach its OpenSearch service and by the
            embeddings and vector tools to reach its embeddings model.
    """

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        }
    )
    @mcp_tool_wrapper("get_current_time", allowed_roles=MCP_READ_ROLES)
    def get_current_time(
        metadata: GenAiMetadata,
    ) -> GetCurrentTimeResponse:
        """Returns the current UTC date and time in ISO 8601 format."""
        now = datetime.now(timezone.utc)
        return GetCurrentTimeResponse(
            time=now.isoformat(),
            timezone="UTC",
        )

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        }
    )
    @mcp_tool_wrapper("format_text", allowed_roles=MCP_READ_ROLES)
    def format_text(
        metadata: GenAiMetadata,
        text: str,
        style: Literal["upper", "lower", "title", "reverse"] = "upper",
    ) -> FormatTextResponse:
        """Formats the given text according to the specified style.

        Args:
            text: The input text to format.
            style: Transformation to apply. One of:
                - "upper": convert to uppercase.
                - "lower": convert to lowercase.
                - "title": capitalize each word.
                - "reverse": reverse the character order.
        """
        formatters = {
            "upper": str.upper,
            "lower": str.lower,
            "title": str.title,
            "reverse": lambda t: t[::-1],
        }
        return FormatTextResponse(result=formatters[style](text), style=style)

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        }
    )
    @mcp_tool_wrapper("analyze_text", allowed_roles=MCP_READ_ROLES)
    def analyze_text(
        metadata: GenAiMetadata,
        text: str,
    ) -> AnalyzeTextResponse:
        """Returns word count and character count for the given text.

        Delegates to the chain's invoke logic.

        Args:
            text: The input text to analyze.
        """
        result = chain.chain().invoke({"text": text})
        return AnalyzeTextResponse(
            text=result["text"],
            word_count=result["word_count"],
            char_count=result["char_count"],
        )

    # The run_opensearch_query tool needs OpenSearch, so it is only registered when
    # the chain was configured with an opensearch_url.
    opensearch_service = chain.opensearch_service
    if opensearch_service is not None:

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            }
        )
        @mcp_tool_wrapper("run_opensearch_query", allowed_roles=MCP_READ_ROLES)
        def run_opensearch_query(
            metadata: GenAiMetadata,
            index: str,
            query: dict[str, Any],
        ) -> RunOpenSearchQueryResponse:
            """Runs an OpenSearch search query on the given index and returns the hits.

            Args:
                index: The OpenSearch index (or index pattern) to search.
                query: The search request body in OpenSearch Query DSL, e.g.
                    {"size": 5, "query": {"match": {"value": "Scott"}}}.
            """
            # Same index normalization (lowercase) as example-chain-opensearch
            transformed_index = index.lower()
            try:
                result = opensearch_service.client.search(
                    index=transformed_index, body=query
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error(
                    f"OpenSearch query on index '{transformed_index}' failed: {e}"
                )
                return RunOpenSearchQueryResponse(
                    index=transformed_index, error=f"OpenSearch query failed: {e}"
                )

            hits = result.get("hits", {})
            total = hits.get("total")
            return RunOpenSearchQueryResponse(
                index=transformed_index,
                total_hits=total.get("value") if isinstance(total, dict) else total,
                hits=hits.get("hits", []),
                aggregations=result.get("aggregations"),
            )

        @mcp.tool(
            annotations={
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": False,
            }
        )
        @mcp_tool_wrapper("create_index", allowed_roles=MCP_WRITE_ROLES)
        def create_index(
            metadata: GenAiMetadata,
            index: str,
            body: dict[str, Any],
        ) -> CreateIndexResponse:
            """Creates an OpenSearch index with the given settings and mappings.

            For a vector index, enable k-NN and map the vector field as knn_vector with
            the dimension of the embedding model (see describe_vector_index), e.g.
            {"settings": {"index": {"knn": true}},
             "mappings": {"properties": {
                "text": {"type": "text"},
                "embedding": {"type": "knn_vector", "dimension": 1024,
                    "method": {"name": "hnsw", "engine": "lucene",
                               "space_type": "cosinesimil"}}}}}

            Args:
                index: The name of the index to create.
                body: The index settings and mappings, in OpenSearch format.
            """
            transformed_index = index.lower()
            try:
                result = opensearch_service.client.indices.create(
                    index=transformed_index, body=body
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error(f"Creation of index '{transformed_index}' failed: {e}")
                return CreateIndexResponse(
                    index=transformed_index, error=f"Index creation failed: {e}"
                )
            return CreateIndexResponse(
                index=transformed_index, acknowledged=result.get("acknowledged")
            )

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            }
        )
        @mcp_tool_wrapper("describe_vector_index", allowed_roles=MCP_READ_ROLES)
        def describe_vector_index(
            metadata: GenAiMetadata,
            index: str,
        ) -> DescribeVectorIndexResponse:
            """Describes the fields of an index: its knn_vector fields (with dimension,
            space_type and engine) and the type of the other fields. Use it before
            add_records or vector_search to know which vector and text fields to use.

            Args:
                index: The OpenSearch index (or index pattern) to describe.
            """
            transformed_index = index.lower()
            try:
                fields = get_index_fields(opensearch_service.client, transformed_index)
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error(f"Describing index '{transformed_index}' failed: {e}")
                return DescribeVectorIndexResponse(
                    index=transformed_index, error=f"Index description failed: {e}"
                )
            vector_fields = knn_vector_fields(fields)
            return DescribeVectorIndexResponse(
                index=transformed_index,
                vector_fields=vector_fields,
                fields={
                    path: field["type"]
                    for path, field in fields.items()
                    if path not in vector_fields
                },
                embedding_model=chain.embeddings.model if chain.embeddings else None,
            )

    # The generate_embeddings tool needs an embedding model, so it is only registered
    # when the chain was configured with an embeddings_endpoint.
    embeddings = chain.embeddings
    if embeddings is not None:

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            }
        )
        @mcp_tool_wrapper("generate_embeddings", allowed_roles=MCP_READ_ROLES)
        def generate_embeddings(
            metadata: GenAiMetadata,
            texts: list[str],
            dimensions: Optional[int] = None,
        ) -> GenerateEmbeddingsResponse:
            """Generates an embedding vector for each of the given texts, in the same order.

            Args:
                texts: The texts to embed.
                dimensions: Optional number of dimensions of the output vectors, for models
                    that support it (e.g. text-embedding-3-*). Model default if omitted.
            """
            # As in mad_hatter: LiteLLM is called on behalf of the nominal user
            auth = GenAiAuthRunnable().invoke(
                {}, mcp_metadata_to_runnable_config(metadata)
            )
            try:
                vectors = embeddings.embed_documents(
                    texts,
                    dimensions=dimensions,
                    headers=embeddings.on_behalf_of_headers(auth),
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error(
                    f"Embeddings generation with '{embeddings.model}' failed: {e}"
                )
                return GenerateEmbeddingsResponse(
                    model=embeddings.model, error=f"Embeddings generation failed: {e}"
                )
            return GenerateEmbeddingsResponse(
                model=embeddings.model,
                dimensions=len(vectors[0]) if vectors else None,
                embeddings=vectors,
            )

    # The add_records and vector_search tools embed texts and use OpenSearch, so they
    # are only registered when both opensearch_url and embeddings_endpoint are set.
    # They compute the vectors on the server, so the LLM never has to copy embedding
    # vectors from generate_embeddings into an OpenSearch query.
    if opensearch_service is not None and embeddings is not None:

        @mcp.tool(
            annotations={
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
            }
        )
        @mcp_tool_wrapper("add_records", allowed_roles=MCP_WRITE_ROLES)
        def add_records(
            metadata: GenAiMetadata,
            index: str,
            records: list[dict[str, Any]],
            refresh: bool = True,
        ) -> AddRecordsResponse:
            """Indexes records into an OpenSearch index with knn_vector fields,
            computing their vectors with the chain's embedding model.

            Do not pass vectors: in the vector field of each record put the pointer
            {"$embed": "<field>"} and the tool replaces it with the embedding of the
            text of <field>. Use a list of fields to embed several fields joined with
            newlines, e.g. {"$embed": ["title", "text"]}. The optional "_id" key is used
            as the document id (an existing document with that id is replaced).
            Example record:
            {"_id": "alice-ch1-0", "text": "Alice follows the White Rabbit",
             "chapter": 1, "embedding": {"$embed": "text"}}

            Args:
                index: The OpenSearch index to add the records to. It must already exist
                    with its knn_vector fields mapped (see create_index).
                records: The records (documents) to index, at most 500 per call.
                refresh: Wait until the records are searchable before returning.
            """
            transformed_index = index.lower()
            if not records or len(records) > ADD_RECORDS_MAX_RECORDS:
                return AddRecordsResponse(
                    index=transformed_index,
                    error=f"Pass between 1 and {ADD_RECORDS_MAX_RECORDS} records",
                )
            # As in generate_embeddings: LiteLLM is called on behalf of the nominal user
            auth = GenAiAuthRunnable().invoke(
                {}, mcp_metadata_to_runnable_config(metadata)
            )
            headers = embeddings.on_behalf_of_headers(auth)
            try:
                vector_fields = knn_vector_fields(
                    get_index_fields(opensearch_service.client, transformed_index)
                )
                requests = collect_embed_requests(records, vector_fields)
                documents = [dict(record) for record in records]
                # One embeddings call per vector field, with that field's dimension
                for field in dict.fromkeys(field for _, field, _ in requests):
                    field_requests = [
                        (position, text)
                        for position, request_field, text in requests
                        if request_field == field
                    ]
                    vectors = embed_for_field(
                        embeddings,
                        [text for _, text in field_requests],
                        vector_fields[field]["dimension"],
                        headers,
                    )
                    for (position, _), vector in zip(field_requests, vectors):
                        documents[position][field] = vector
                response = opensearch_service.client.bulk(
                    body=build_bulk_body(transformed_index, documents),
                    refresh="wait_for" if refresh else "false",
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error(f"Adding records to index '{transformed_index}' failed: {e}")
                return AddRecordsResponse(
                    index=transformed_index, error=f"Adding records failed: {e}"
                )
            ids, failures = summarize_bulk_response(response)
            return AddRecordsResponse(
                index=transformed_index,
                indexed=len(ids),
                ids=ids,
                failures=failures or None,
            )

        @mcp.tool(
            annotations={
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
            }
        )
        @mcp_tool_wrapper("vector_search", allowed_roles=MCP_READ_ROLES)
        # pylint: disable-next=too-many-positional-arguments
        def vector_search(
            metadata: GenAiMetadata,
            question: str,
            index: str,
            vector_field: Optional[str] = None,
            k: int = 5,
            min_score: Optional[float] = None,
            text_field: Optional[str] = None,
            # pylint: disable-next=redefined-builtin
            filter: Optional[dict[str, Any]] = None,
            source_fields: Optional[list[str]] = None,
        ) -> VectorSearchResponse:
            """Retrieves the chunks of a vector index most similar to the question (RAG
            retrieval). The question is embedded with the chain's embedding model, so
            the index must have been filled with the same model (e.g. with add_records).

            Args:
                question: The question or text to search for.
                index: The OpenSearch index (or index pattern) to search.
                vector_field: The knn_vector field to search. Optional when the index
                    has a single knn_vector field (see describe_vector_index).
                k: The number of chunks to return (1-100).
                min_score: Optional minimum OpenSearch score of the chunks. Its scale
                    depends on the field's space_type: for cosinesimil it is
                    (1 + cosine similarity) / 2, from 0 to 1; for l2 it is
                    1 / (1 + squared distance).
                text_field: Optional top-level field with the chunk text, returned as
                    "text" (the rest of the document is returned as "metadata").
                filter: Optional OpenSearch Query DSL filter applied during the k-NN
                    search, e.g. {"term": {"chapter": 1}}.
                source_fields: Optional fields of the documents to return (all but the
                    vectors by default).
            """
            transformed_index = index.lower()
            if not 1 <= k <= VECTOR_SEARCH_MAX_K:
                return VectorSearchResponse(
                    index=transformed_index,
                    error=f"k must be between 1 and {VECTOR_SEARCH_MAX_K}",
                )
            # As in generate_embeddings: LiteLLM is called on behalf of the nominal user
            auth = GenAiAuthRunnable().invoke(
                {}, mcp_metadata_to_runnable_config(metadata)
            )
            try:
                vector_fields = knn_vector_fields(
                    get_index_fields(opensearch_service.client, transformed_index)
                )
                if vector_field is None and len(vector_fields) == 1:
                    vector_field = next(iter(vector_fields))
                if vector_field not in vector_fields:
                    raise ValueError(
                        f"vector_field must be one of the knn_vector fields of the "
                        f"index: {sorted(vector_fields)}"
                    )
                vector = embed_for_field(
                    embeddings,
                    [question],
                    vector_fields[vector_field]["dimension"],
                    embeddings.on_behalf_of_headers(auth),
                )[0]
                body = build_knn_query(
                    vector_field,
                    vector,
                    k,
                    min_score=min_score,
                    filter_query=filter,
                    source_fields=source_fields,
                    excluded_fields=list(vector_fields),
                )
                result = opensearch_service.client.search(
                    index=transformed_index, body=body
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                log.error(f"Vector search on index '{transformed_index}' failed: {e}")
                return VectorSearchResponse(
                    index=transformed_index,
                    vector_field=vector_field,
                    error=f"Vector search failed: {e}",
                )
            return VectorSearchResponse(
                index=transformed_index,
                vector_field=vector_field,
                chunks=hits_to_chunks(
                    result.get("hits", {}).get("hits", []), text_field
                ),
            )

    # The mad_hatter tool needs an LLM, so it is only registered when the chain
    # was configured with a gateway_endpoint (and therefore built the actor).
    mad_hatter_actor = chain.mad_hatter_actor
    if mad_hatter_actor is None:
        return

    @mcp.tool(
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
        }
    )
    @mcp_tool_wrapper("mad_hatter", allowed_roles=MCP_READ_ROLES)
    def mad_hatter(
        metadata: GenAiMetadata,
        user_request: str,
    ) -> MadHatterResponse:
        """Answers the user's request as the Mad Hatter from Alice in Wonderland,
        with a rhyming riddle. Use it when the user asks for a mad hatter style answer.

        Args:
            user_request: The user's request or question to be answered as a riddle.
        """
        # 1. MCP tools receive GenAiMetadata (from the MCP request _meta) instead of
        #    a RunnableConfig. Convert it so it can be passed to Langchain runnables.
        config = mcp_metadata_to_runnable_config(metadata)

        # 2. Same as _extract_genai_auth in example-chain-basic-actor: build the
        #    GenAiAuth of the nominal user from the metadata.
        auth = GenAiAuthRunnable().invoke({}, config)

        # 3. Actors read genai_auth from the config metadata to call Stratio GenAI
        #    LiteLLM on behalf of the user (instead of with the chain identity).
        config["metadata"][CHAIN_KEY_GENAI_AUTH] = auth

        # 4. Same as _extract_user_name: the actor replies backwards to Alice.
        actor_input = {
            CHAIN_KEY_USER_REQUEST: user_request,
            CHAIN_KEY_USER_NAME: auth.user_id_impersonated or auth.user_id,
        }

        # 5. Same as _invoke_actor: run the actor chain.
        try:
            output = mad_hatter_actor.get_chain().invoke(actor_input, config)
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error(f"mad_hatter actor invocation failed: {e}")
            return MadHatterResponse(error=f"Mad Hatter actor failed: {e}")

        return MadHatterResponse(
            user_request_explanation=output.user_request_explanation,
            riddle=output.mad_hutter_riddle,
            riddle_explanation=output.message,
        )
