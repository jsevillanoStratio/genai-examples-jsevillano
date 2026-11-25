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
    """
    app = GenAiServer(
        module_name="simple_mcp_chain.chain",
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
    }
    main(chain_config)
