"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

"""Example MCP client that calls the tools exposed by SimpleMcpChain.

Prerequisites:
  1. The chain must be running locally:
       poetry run python custom_mcp_chain/main.py

  2. Run this script from the example directory:
       poetry run python scripts/mcp_client_example.py

The mad_hatter, run_opensearch_query and generate_embeddings tools are
optional: they are only called when the server lists them, i.e. when the chain
runs with "gateway_endpoint", "opensearch_url" and "embeddings_endpoint"
respectively (see main.py). The RAG tools (create_index, add_records and
vector_search) need both "opensearch_url" and "embeddings_endpoint".
"""

import asyncio

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

MCP_URL = "http://127.0.0.1:8080/mcp"

# Auth metadata passed as MCP request _meta.
# Locally the server falls back to env vars if this is omitted, but the
# platform always injects proper auth in production.
AUTH_META = {
    "genai_state": {
        "client_auth_type": "mtls",
        "client_user_id": "local-dev-user",
        "client_tenant": "local-tenant",
        "client_role": "consumer",
    }
}

# Index and query used by run_opensearch_query. Adapt them to your data, e.g.
# OPENSEARCH_INDEX = "semantic_banking_customer_product360" and
# OPENSEARCH_QUERY = {"size": 3, "query": {"match": {"value": "Scott"}}}
OPENSEARCH_INDEX = "*"
OPENSEARCH_QUERY = {"size": 3, "query": {"match_all": {}}}

# Texts and optional output size used by generate_embeddings
EMBEDDINGS_TEXTS = ["hello mcp world", "the Queen of Hearts"]
EMBEDDINGS_DIMENSIONS = 256

# Vector index used by the RAG tools. create_index and add_records require a write
# role, so they are called with this metadata.
RAG_AUTH_META = {
    "genai_state": {**AUTH_META["genai_state"], "client_role": "contributor"}
}
RAG_INDEX = "custom_mcp_rag_example"
RAG_INDEX_BODY = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "chapter": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                },
            },
        }
    },
}
# The "$embed" pointer asks add_records to fill "embedding" with the embedding of "text"
RAG_RECORDS = [
    {
        "_id": f"alice-{i}",
        "text": text,
        "chapter": chapter,
        "embedding": {"$embed": "text"},
    }
    for i, (chapter, text) in enumerate(
        [
            (1, "Alice follows the White Rabbit down the rabbit hole."),
            (7, "The Mad Hatter hosts an endless tea party with the March Hare."),
            (
                8,
                "The Queen of Hearts shouts 'Off with their heads!' at the croquet game.",
            ),
        ]
    )
]
RAG_QUESTION = "Who gives a tea party?"


async def main() -> None:
    print(f"Connecting to MCP server at {MCP_URL}\n")

    async with Client(StreamableHttpTransport(MCP_URL)) as client:

        # ------------------------------------------------------------------
        # Discover available tools
        # ------------------------------------------------------------------
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        print()

        # ------------------------------------------------------------------
        # Call get_current_time (no user arguments)
        # ------------------------------------------------------------------
        print("Calling get_current_time ...")
        result = await client.call_tool("get_current_time", {}, meta=AUTH_META)
        if result.is_error:
            print(f"  ERROR: {result.structured_content}")
        else:
            print(f"  time:     {result.structured_content['time']}")
            print(f"  timezone: {result.structured_content['timezone']}")
        print()

        # ------------------------------------------------------------------
        # Call format_text with each supported style
        # ------------------------------------------------------------------
        sample_text = "hello mcp world"
        for style in ("upper", "lower", "title", "reverse"):
            print(f"Calling format_text(text={sample_text!r}, style={style!r}) ...")
            result = await client.call_tool(
                "format_text",
                {"text": sample_text, "style": style},
                meta=AUTH_META,
            )
            if result.is_error:
                print(f"  ERROR: {result.structured_content}")
            else:
                print(f"  result: {result.structured_content['result']}")
        print()

        # ------------------------------------------------------------------
        # Call analyze_text — delegates to chain.chain().invoke()
        # ------------------------------------------------------------------
        print("Calling analyze_text (delegates to chain invoke) ...")
        result = await client.call_tool(
            "analyze_text",
            {"text": sample_text},
            meta=AUTH_META,
        )
        if result.is_error:
            print(f"  ERROR: {result.structured_content}")
        else:
            print(f"  word_count: {result.structured_content['word_count']}")
            print(f"  char_count: {result.structured_content['char_count']}")
        print()

        # ------------------------------------------------------------------
        # Call mad_hatter — uses a Stratio actor (LLM).
        # Only registered when the chain runs with "gateway_endpoint".
        # ------------------------------------------------------------------
        if "mad_hatter" in tool_names:
            user_request = "Where is the Queen of Hearts?"
            print(f"Calling mad_hatter(user_request={user_request!r}) ...")
            result = await client.call_tool(
                "mad_hatter",
                {"user_request": user_request},
                meta=AUTH_META,
            )
            if result.is_error or result.structured_content.get("error"):
                print(f"  ERROR: {result.structured_content}")
            else:
                print(f"  riddle:      {result.structured_content['riddle']}")
                print(
                    f"  explanation: {result.structured_content['riddle_explanation']}"
                )
        else:
            print("Skipping mad_hatter (chain running without gateway_endpoint)")
        print()

        # ------------------------------------------------------------------
        # Call run_opensearch_query — runs a search on OpenSearch.
        # Only registered when the chain runs with "opensearch_url".
        # ------------------------------------------------------------------
        if "run_opensearch_query" in tool_names:
            print(
                f"Calling run_opensearch_query(index={OPENSEARCH_INDEX!r}, "
                f"query={OPENSEARCH_QUERY}) ..."
            )
            result = await client.call_tool(
                "run_opensearch_query",
                {"index": OPENSEARCH_INDEX, "query": OPENSEARCH_QUERY},
                meta=AUTH_META,
            )
            if result.is_error or result.structured_content.get("error"):
                print(f"  ERROR: {result.structured_content}")
            else:
                print(f"  total_hits: {result.structured_content['total_hits']}")
                for hit in result.structured_content["hits"]:
                    print(f"  - [{hit['_index']}] {hit['_id']}: {hit['_source']}")
        else:
            print(
                "Skipping run_opensearch_query (chain running without opensearch_url)"
            )
        print()

        # ------------------------------------------------------------------
        # Call generate_embeddings — uses a LiteLLM embedding model.
        # Only registered when the chain runs with "embeddings_endpoint".
        # ------------------------------------------------------------------
        if "generate_embeddings" in tool_names:
            print(
                f"Calling generate_embeddings(texts={EMBEDDINGS_TEXTS}, "
                f"dimensions={EMBEDDINGS_DIMENSIONS}) ..."
            )
            result = await client.call_tool(
                "generate_embeddings",
                {"texts": EMBEDDINGS_TEXTS, "dimensions": EMBEDDINGS_DIMENSIONS},
                meta=AUTH_META,
            )
            if result.is_error or result.structured_content.get("error"):
                print(f"  ERROR: {result.structured_content}")
            else:
                print(f"  model:      {result.structured_content['model']}")
                print(f"  dimensions: {result.structured_content['dimensions']}")
                for text, vector in zip(
                    EMBEDDINGS_TEXTS, result.structured_content["embeddings"]
                ):
                    print(f"  - {text!r}: {vector[:3]} ...")
        else:
            print(
                "Skipping generate_embeddings (chain running without embeddings_endpoint)"
            )
        print()

        # ------------------------------------------------------------------
        # RAG: create a vector index, add records (the vectors are computed by
        # the server from "$embed" pointers) and retrieve the closest chunks.
        # Only registered when the chain runs with "opensearch_url" and
        # "embeddings_endpoint".
        # ------------------------------------------------------------------
        if "vector_search" in tool_names:
            print(f"Calling create_index(index={RAG_INDEX!r}) ...")
            result = await client.call_tool(
                "create_index",
                {"index": RAG_INDEX, "body": RAG_INDEX_BODY},
                meta=RAG_AUTH_META,
            )
            # Fails when the index already exists, e.g. on a second run
            print(f"  {result.structured_content}")

            print(
                f"Calling add_records(index={RAG_INDEX!r}, {len(RAG_RECORDS)} records) ..."
            )
            result = await client.call_tool(
                "add_records",
                {"index": RAG_INDEX, "records": RAG_RECORDS},
                meta=RAG_AUTH_META,
            )
            if result.is_error or result.structured_content.get("error"):
                print(f"  ERROR: {result.structured_content}")
            else:
                print(f"  indexed: {result.structured_content['indexed']}")

            print(f"Calling vector_search(question={RAG_QUESTION!r}) ...")
            result = await client.call_tool(
                "vector_search",
                {
                    "question": RAG_QUESTION,
                    "index": RAG_INDEX,
                    "k": 2,
                    "text_field": "text",
                },
                meta=AUTH_META,
            )
            if result.is_error or result.structured_content.get("error"):
                print(f"  ERROR: {result.structured_content}")
            else:
                for chunk in result.structured_content["chunks"]:
                    print(
                        f"  - {chunk['score']:.3f} {chunk['text']} {chunk['metadata']}"
                    )
        else:
            print(
                "Skipping RAG tools (chain running without opensearch_url "
                "and embeddings_endpoint)"
            )
        print()

        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
