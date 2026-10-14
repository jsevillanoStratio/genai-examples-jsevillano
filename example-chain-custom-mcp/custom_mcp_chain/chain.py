"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

from typing import Optional

from fastmcp import FastMCP
from genai_core.chain.base import BaseGenAiChain, GenAiChainParams
from genai_core.clients.vault.vault_client import VaultClient
from genai_core.logger.logger import log
from genai_core.services.opensearch.opensearch_service import OpenSearchService
from langchain_core.runnables import Runnable, chain

from .actors.mad_hatter_actor import MadHatterActor
from .embeddings import StratioLitellmGatewayEmbeddings
from .mcp_tools import configure_mcp_tools


# This chain demonstrates how to embed an MCP server in a GenAI chain.
# The chain's main invoke endpoint accepts a text input and returns basic
# statistics about it. In addition, the chain exposes two MCP tools
# (get_current_time and format_text) that AI assistants can call directly.
# Neither the invoke endpoint nor those MCP tools require an LLM, which shows
# that MCP tools can be pure utility functions independent of the chain logic.
# Optionally, when a gateway_endpoint is configured, the chain also builds a
# Stratio actor (the Mad Hatter, taken from example-chain-basic-actor) and
# exposes it as the mad_hatter MCP tool, showing how to use actors from MCP.
# Likewise, when an opensearch_url is configured, the chain connects to
# OpenSearch (as in example-chain-opensearch) and exposes the
# run_opensearch_query, create_index and describe_vector_index MCP tools, and when
# an embeddings_endpoint is configured, it exposes the generate_embeddings MCP tool
# using that LiteLLM embedding model. With both, it also exposes add_records and
# vector_search, which compute the vectors on the server for RAG over any index
# with knn_vector fields.
class SimpleMcpChain(BaseGenAiChain):
    # Actor used by the mad_hatter MCP tool. None when no LLM is configured.
    mad_hatter_actor: Optional[MadHatterActor]
    # Service used by the run_opensearch_query MCP tool. None when no OpenSearch is configured.
    opensearch_service: Optional[OpenSearchService]
    # Embeddings model used by the generate_embeddings MCP tool. None when not configured.
    embeddings: Optional[StratioLitellmGatewayEmbeddings]

    def __init__(
        self,
        gateway_endpoint: Optional[str] = None,
        llm_timeout: int = 30,
        opensearch_url: Optional[str] = None,
        embeddings_endpoint: Optional[str] = None,
    ):
        """
        Initializes the SimpleMcpChain.

        :param gateway_endpoint: The ID of the model group in Stratio GenAI LiteLLM pointing
            to the desired model. If not provided, the mad_hatter MCP tool is not registered.
        :param llm_timeout: Timeout for the LLM model, default is 30 seconds.
        :param opensearch_url: The URL of the OpenSearch instance. If not provided, the
            run_opensearch_query MCP tool is not registered.
        :param embeddings_endpoint: The ID of the model group in Stratio GenAI LiteLLM pointing
            to the embedding model. If not provided, the generate_embeddings MCP tool is not
            registered.
        """
        log.info("Preparing Simple MCP Example chain")
        self.mad_hatter_actor = None
        if gateway_endpoint:
            self.mad_hatter_actor = MadHatterActor(
                gateway_endpoint=gateway_endpoint,
                timeout=llm_timeout,
            )
        else:
            log.info("No gateway_endpoint configured: mad_hatter MCP tool disabled")
        self.opensearch_service = None
        if opensearch_url:
            self.opensearch_service = self._init_opensearch(opensearch_url)
        else:
            log.info(
                "No opensearch_url configured: run_opensearch_query MCP tool disabled"
            )
        self.embeddings = None
        if embeddings_endpoint:
            self.embeddings = StratioLitellmGatewayEmbeddings(
                endpoint=embeddings_endpoint,
                timeout=llm_timeout,
            )
        else:
            log.info(
                "No embeddings_endpoint configured: generate_embeddings MCP tool disabled"
            )
        log.info("Simple MCP Example chain ready!")

    @staticmethod
    def _init_opensearch(opensearch_url: str) -> OpenSearchService:
        """Initializes the OpenSearch client, as in example-chain-opensearch.

        The certificates are obtained from Vault. For local development, the
        VaultClient uses the VAULT_LOCAL_CLIENT_CERT, VAULT_LOCAL_CLIENT_KEY and
        VAULT_LOCAL_CA_CERTS environment variables instead.

        :param opensearch_url: The URL of the OpenSearch instance.
        :return: The connected OpenSearch service.
        """
        try:
            vault = VaultClient()
            cert, key = vault.get_service_certificate_pem_paths()
            ca = vault.get_ca_bundle_pem_path()
            log.info(f"Trying to connect with OpenSearch {opensearch_url}...")
            opensearch_service = OpenSearchService(
                opensearch_url=opensearch_url,
                ca_certs=ca,
                client_cert=cert,
                client_key=key,
            )
            connected, error_msg = opensearch_service.test_connection()
            if not connected:
                raise RuntimeError(
                    f"Unable to test connection with {opensearch_url}: {error_msg}"
                )
            log.info("Connected with OpenSearch")
        except Exception as error:
            error_msg = f"Unable to init Simple MCP chain. Unable to connect with OpenSearch. Error: {error}"
            log.error(error_msg)
            raise RuntimeError(error_msg) from error
        return opensearch_service

    def chain(self) -> Runnable:
        """Returns a Runnable that computes basic statistics for the input text.

        Accepts a dict with a ``text`` key and returns a dict with:
        - ``text``: the original input.
        - ``word_count``: number of whitespace-separated words.
        - ``char_count``: total number of characters.
        """

        @chain
        def _process(input_data: dict) -> dict:
            text = input_data.get("text", "")
            return {
                "text": text,
                "word_count": len(text.split()) if text else 0,
                "char_count": len(text),
            }

        return _process

    def chain_params(self) -> GenAiChainParams:
        """Defines the chain parameters.

        mcp_server_name is required when the chain implements mcp_server(): it is
        the name of the MCP server registered in Stratio GenAI LiteLLM when the
        chain is deployed. mcp_server_description is used by LiteLLM to present
        the server to agents.

        :return: The chain parameters.
        """
        return GenAiChainParams(
            mcp_server_name="Custom MCP Tests",
            mcp_server_description=(
                "Utility tools to get the current UTC time, format text and count "
                "its words and characters, plus a Mad Hatter tool that answers "
                "requests with a rhyming riddle, tools to run OpenSearch queries and "
                "create and describe indices, a tool to generate text embeddings, and "
                "RAG tools to add records to a vector index (computing their vectors) "
                "and to retrieve the chunks most similar to a question."
            ),
        )

    def mcp_server(self, mcp: FastMCP) -> Optional[FastMCP]:
        """Configures and returns the embedded MCP server.

        Receives the FastMCP instance managed by the framework, registers the
        utility tools defined in mcp_tools.py, and returns it. Returning None
        would disable MCP for this chain.

        Args:
            mcp: The FastMCP server instance provided by the framework.

        Returns:
            The configured FastMCP instance, or None to disable MCP.
        """
        if mcp is None:
            return None
        configure_mcp_tools(mcp, self)
        return mcp
