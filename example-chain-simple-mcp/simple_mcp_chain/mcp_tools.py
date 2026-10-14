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
from genai_core.runnables.genai_auth import MCP_READ_ROLES, GenAiAuthRunnable
from genai_core.runnables.genai_metadata import (
    GenAiMetadata,
    mcp_metadata_to_runnable_config,
)
from genai_core.runnables.mcp_tool_wrapper import mcp_tool_wrapper

from .constants.constants import CHAIN_KEY_USER_NAME

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


def configure_mcp_tools(mcp: FastMCP, chain: "SimpleMcpChain") -> None:
    """Register all MCP tools on the given FastMCP instance.

    Args:
        mcp: The FastMCP server instance to register tools on.
        chain: The SimpleMcpChain instance used by analyze_text to delegate
            to the chain's invoke logic, by mad_hatter to reach its actor and by
            run_opensearch_query to reach its OpenSearch service.
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
