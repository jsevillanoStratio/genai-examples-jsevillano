"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import FastMCPTransport
from genai_core.constants.constants import CHAIN_KEY_GENAI_AUTH, CHAIN_KEY_USER_REQUEST
from genai_core.runnables.genai_auth import GenAiAuth
from genai_core.test.mock_helper import mock_init_litellm_gateway

from custom_mcp_chain.actors.mad_hatter_actor import MadHatterActorOutput
from custom_mcp_chain.chain import SimpleMcpChain
from custom_mcp_chain.constants.constants import CHAIN_KEY_USER_NAME


@pytest.fixture
def mcp_server() -> FastMCP:
    """Build an in-memory FastMCP server with tools registered from SimpleMcpChain."""
    mcp = FastMCP("simple-mcp-chain-test")
    chain = SimpleMcpChain()
    chain.mcp_server(mcp)
    return mcp


@pytest.fixture(autouse=True)
def setup_auth_env(monkeypatch):
    """Provide auth env vars so mcp_tool_wrapper can extract role metadata.

    In production the Stratio GenAI API injects this automatically via the MCP
    request _meta field. Locally and in unit tests env vars are the fallback.
    """
    monkeypatch.setenv("GENAI_METADATA_USER_ID", "test-user")
    monkeypatch.setenv("GENAI_METADATA_TENANT", "test-tenant")
    monkeypatch.setenv("GENAI_METADATA_ROLE", "consumer")


class TestMcpEndpointTools:
    """Integration tests that call MCP tools through FastMCPTransport (in-memory).

    FastMCPTransport connects the Client directly to the FastMCP server in the
    same process without any HTTP server. The protocol path is identical to
    calling the live /mcp endpoint, making these tests suitable for verifying
    tool registration, argument handling, and response shapes.
    """

    @pytest.mark.asyncio
    async def test_list_tools_exposes_expected_tools(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            assert "get_current_time" in tool_names
            assert "format_text" in tool_names
            # No gateway_endpoint configured -> no actor -> no mad_hatter tool
            assert "mad_hatter" not in tool_names
            # No opensearch_url configured -> no run_opensearch_query tool
            assert "run_opensearch_query" not in tool_names

    @pytest.mark.asyncio
    async def test_get_current_time_returns_utc_time(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool("get_current_time", {})
            assert not result.is_error
            assert result.structured_content is not None
            assert "time" in result.structured_content
            assert result.structured_content.get("timezone") == "UTC"

    @pytest.mark.asyncio
    async def test_format_text_upper(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool(
                "format_text", {"text": "hello world", "style": "upper"}
            )
            assert not result.is_error
            assert result.structured_content["result"] == "HELLO WORLD"
            assert result.structured_content["style"] == "upper"

    @pytest.mark.asyncio
    async def test_format_text_lower(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool(
                "format_text", {"text": "HELLO WORLD", "style": "lower"}
            )
            assert result.structured_content["result"] == "hello world"

    @pytest.mark.asyncio
    async def test_format_text_title(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool(
                "format_text", {"text": "hello world", "style": "title"}
            )
            assert result.structured_content["result"] == "Hello World"

    @pytest.mark.asyncio
    async def test_format_text_reverse(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool(
                "format_text", {"text": "hello", "style": "reverse"}
            )
            assert result.structured_content["result"] == "olleh"

    @pytest.mark.asyncio
    async def test_format_text_default_style_is_upper(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool("format_text", {"text": "hello"})
            assert result.structured_content["result"] == "HELLO"

    @pytest.mark.asyncio
    async def test_analyze_text_calls_chain_invoke(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool("analyze_text", {"text": "hello world"})
            assert not result.is_error
            assert result.structured_content["word_count"] == 2
            assert result.structured_content["char_count"] == 11
            assert result.structured_content["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_analyze_text_empty_string(self, mcp_server):
        async with Client(FastMCPTransport(mcp_server)) as client:
            result = await client.call_tool("analyze_text", {"text": ""})
            assert result.structured_content["word_count"] == 0
            assert result.structured_content["char_count"] == 0


@pytest.fixture
def mad_hatter_actor_chain() -> MagicMock:
    """Mocked actor chain so the mad_hatter tool can be tested without an LLM."""
    actor_chain = MagicMock()
    actor_chain.invoke.return_value = MadHatterActorOutput(
        user_request_explanation="The user asks where the Queen is.",
        mad_hutter_riddle="Where hearts are red and roses too...",
        message="The riddle points to the Queen's garden.",
    )
    return actor_chain


@pytest.fixture
def mad_hatter_mcp_server(mocker, mad_hatter_actor_chain) -> FastMCP:
    """In-memory FastMCP server whose chain has the Mad Hatter actor (mocked LLM)."""
    mock_init_litellm_gateway(mocker)
    mcp = FastMCP("simple-mcp-chain-mad-hatter-test")
    chain = SimpleMcpChain(gateway_endpoint="test-endpoint")
    chain.mad_hatter_actor.get_chain = MagicMock(return_value=mad_hatter_actor_chain)
    chain.mcp_server(mcp)
    return mcp


class TestMadHatterTool:
    @pytest.mark.asyncio
    async def test_mad_hatter_tool_is_listed(self, mad_hatter_mcp_server):
        async with Client(FastMCPTransport(mad_hatter_mcp_server)) as client:
            tool_names = {t.name for t in await client.list_tools()}
            assert "mad_hatter" in tool_names

    @pytest.mark.asyncio
    async def test_mad_hatter_returns_actor_output(
        self, mad_hatter_mcp_server, mad_hatter_actor_chain
    ):
        async with Client(FastMCPTransport(mad_hatter_mcp_server)) as client:
            result = await client.call_tool(
                "mad_hatter", {"user_request": "Where is the Queen of Hearts?"}
            )
            assert not result.is_error
            assert (
                result.structured_content["riddle"]
                == "Where hearts are red and roses too..."
            )
            assert result.structured_content["riddle_explanation"] == (
                "The riddle points to the Queen's garden."
            )

        actor_input, config = mad_hatter_actor_chain.invoke.call_args.args
        assert actor_input[CHAIN_KEY_USER_REQUEST] == "Where is the Queen of Hearts?"
        # The user name comes from the MCP metadata (env vars in local/test runs)
        assert actor_input[CHAIN_KEY_USER_NAME] == "test-user"
        # genai_auth is propagated so the actor calls LiteLLM on behalf of the user
        auth = config["metadata"][CHAIN_KEY_GENAI_AUTH]
        assert isinstance(auth, GenAiAuth)
        assert auth.user_id == "test-user"

    @pytest.mark.asyncio
    async def test_mad_hatter_returns_error_when_actor_fails(
        self, mad_hatter_mcp_server, mad_hatter_actor_chain
    ):
        mad_hatter_actor_chain.invoke.side_effect = RuntimeError("LLM unavailable")
        async with Client(FastMCPTransport(mad_hatter_mcp_server)) as client:
            result = await client.call_tool("mad_hatter", {"user_request": "Hi!"})
            assert "LLM unavailable" in result.structured_content["error"]


@pytest.fixture
def opensearch_client(mocker) -> MagicMock:
    """Mocked OpenSearch client so run_opensearch_query can be tested without OpenSearch."""
    mock_vault = mocker.patch("custom_mcp_chain.chain.VaultClient")
    mock_vault.return_value.get_service_certificate_pem_paths.return_value = (
        "cert",
        "key",
    )
    mock_service = mocker.patch("custom_mcp_chain.chain.OpenSearchService")
    mock_service.return_value.test_connection.return_value = (True, "")
    client = mock_service.return_value.client
    client.search.return_value = {
        "hits": {
            "total": {"value": 1, "relation": "eq"},
            "hits": [{"_id": "1", "_source": {"value": "Scott"}}],
        },
        "aggregations": {"by_table": {"buckets": []}},
    }
    return client


@pytest.fixture
def opensearch_mcp_server(opensearch_client) -> FastMCP:
    """In-memory FastMCP server whose chain has an OpenSearch service (mocked client)."""
    mcp = FastMCP("simple-mcp-chain-opensearch-test")
    chain = SimpleMcpChain(opensearch_url="https://opensearch:9200")
    chain.mcp_server(mcp)
    return mcp


class TestRunOpenSearchQueryTool:
    @pytest.mark.asyncio
    async def test_run_opensearch_query_tool_is_listed(self, opensearch_mcp_server):
        async with Client(FastMCPTransport(opensearch_mcp_server)) as client:
            tool_names = {t.name for t in await client.list_tools()}
            assert "run_opensearch_query" in tool_names

    @pytest.mark.asyncio
    async def test_run_opensearch_query_returns_hits(
        self, opensearch_mcp_server, opensearch_client
    ):
        query = {"size": 1, "query": {"match": {"value": "Scott"}}}
        async with Client(FastMCPTransport(opensearch_mcp_server)) as client:
            result = await client.call_tool(
                "run_opensearch_query", {"index": "My-Index", "query": query}
            )
            assert not result.is_error
            assert result.structured_content["index"] == "my-index"
            assert result.structured_content["total_hits"] == 1
            assert result.structured_content["hits"][0]["_source"]["value"] == "Scott"
            assert result.structured_content["aggregations"] == {
                "by_table": {"buckets": []}
            }

        # The index is lowercased and the query is sent as the search body
        opensearch_client.search.assert_called_once_with(index="my-index", body=query)

    @pytest.mark.asyncio
    async def test_run_opensearch_query_returns_error_when_search_fails(
        self, opensearch_mcp_server, opensearch_client
    ):
        opensearch_client.search.side_effect = RuntimeError("index_not_found")
        async with Client(FastMCPTransport(opensearch_mcp_server)) as client:
            result = await client.call_tool(
                "run_opensearch_query",
                {"index": "missing", "query": {"query": {"match_all": {}}}},
            )
            assert "index_not_found" in result.structured_content["error"]
