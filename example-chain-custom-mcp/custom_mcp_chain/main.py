"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

from genai_core.server.server import GenAiServer
import os
from dotenv import load_dotenv
load_dotenv(".local_env/genai-env.env", override=True)


def patch_litellm_keos_routes():
    """
    LOCAL DEVELOPMENT WORKAROUND - do not commit.

    The GenAI Developer Proxy 0.6.x does not expose genai-litellm, so we call LiteLLM directly through
    its ingress with an API key. The Stratio '/keos/*' routes reject API keys (they require the mTLS identity),
    so they are redirected to the standard LiteLLM routes, which return the same data shape.
    Remove it once the Developer Proxy is upgraded to 0.7.x with genai-litellm enabled.
    """
    from genai_core.clients.gateway.litellm_gateway_client import LitellmGatewayClient

    keos_to_standard_routes = {
        "/keos/models": "/v1/model/info",
        "/keos/models/groups": "/model_group/info",
    }
    original_request_keos = LitellmGatewayClient._request_keos

    def _request_keos(self, url, params):
        return original_request_keos(self, keos_to_standard_routes.get(url, url), params)

    LitellmGatewayClient._request_keos = _request_keos


if os.environ.get("GENAI_LITELLM_API_KEY") and os.environ.get("PATCH_LITELLM_KEOS_ROUTES", "false").lower() == "true":
    patch_litellm_keos_routes()
    #pass

def main(chain_config):
    """
    Starts a stand-alone GenAI-api-like server with the chain loaded.

    The invoke endpoint and the utility MCP tools do not call an LLM. Only the
    mad_hatter MCP tool does: it is registered only when "gateway_endpoint" is
    set in the chain config, and then it needs access to a Stratio GenAI LiteLLM
    server (e.g. through the GenAI development proxy). Without "gateway_endpoint"
    the chain runs locally without any additional infrastructure.

    Invoke endpoint example body (POST /invoke):
    ```json
    {
      "input": {
        "text": "Hello, MCP world!"
      },
      "config": {
        "metadata": {
          "__genai_state": {
            "client_auth_type": "mtls",
            "client_user_id": "<your-user>",
            "client_tenant": "<your-tenant>"
          }
        }
      }
    }
    ```

    MCP tools available at /mcp (SSE) or /mcp/messages (JSON-RPC):
    - get_current_time: returns the current UTC date and time.
    - format_text: formats a string (upper / lower / title / reverse).
    - analyze_text: word and character count, delegating to the chain.
    - mad_hatter: answers a request with a riddle using the Mad Hatter actor
      (only when "gateway_endpoint" is configured).
    - run_opensearch_query: runs an OpenSearch search query on an index
      (only when "opensearch_url" is configured).
    - create_index / describe_vector_index: create an index and describe its
      knn_vector fields (only when "opensearch_url" is configured).
    - generate_embeddings: generates text embeddings with a LiteLLM embedding model
      (only when "embeddings_endpoint" is configured).
    - add_records / vector_search: index records computing their vectors from
      {"$embed": "<field>"} pointers, and retrieve the chunks most similar to a
      question (only when both "opensearch_url" and "embeddings_endpoint" are set).
    """
    app = GenAiServer(
        module_name="custom_mcp_chain.chain",
        class_name="SimpleMcpChain",
        config=chain_config,
    )
    app.start_server()


if __name__ == "__main__":
    # Without "gateway_endpoint" this chain requires no external services and the
    # mad_hatter MCP tool is disabled. Remove it to run without LiteLLM.
    # gateway_endpoint: The ID of the model group in Stratio GenAI LiteLLM pointing to the desired model.
    chain_config = {
        "gateway_endpoint": "gpt-5.4-mini",
        "llm_timeout": 30,
        # OpenSearch url reachable from your local machine (the developer proxy), e.g.
        # https://genai-developer-proxy-loadbalancer.stratio-genai.k8s.bravo.labs.stratio.com:8080/service/opensearch/
        # When deployed in GenAI API use the internal url instead:
        # https://opensearch-genai.stratio-genai:9200
        # Remove it (or leave OPENSEARCH_URL unset) to disable the run_opensearch_query MCP tool.
        "opensearch_url": os.getenv("OPENSEARCH_URL"),
        # embeddings_endpoint: The ID of the model group in Stratio GenAI LiteLLM pointing to
        # an embedding model. Remove it to disable the generate_embeddings MCP tool.
        "embeddings_endpoint": "text-embedding-3-large",
    }
    main(chain_config)
