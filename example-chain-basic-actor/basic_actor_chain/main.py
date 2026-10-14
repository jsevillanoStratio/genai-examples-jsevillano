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
    Starts a stand-alone GenAI-api-like server with the chain loaded so that in can be easily executed locally.
    Note that the chain will need access to a Stratio GenAI LiteLLM server, which could be provided from your
    local machine via the GenAI development proxy. An example of json body to send in invoke POST is
    ```json
       {
          "input": {
             "user_request": "Hi! Nice to meet you! Where's the Queen of Hearts?"
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
      The "config" -> "metadata" -> "__genai_state" is only needed to test while developing locally.
      In a real environment GenAI API adds automatically that fields from the auth info before
      passing the data to the chain
    """
    app = GenAiServer(
        module_name="basic_actor_chain.chain",
        class_name="BasicActorChain",
        config=chain_config,
    )
    app.start_server()


if __name__ == "__main__":
    # Before running this script, refer to the README.md file to know how to set up
    # your environment correctly in order to communicate with the Stratio GenAI LiteLLM

    # # Stratio GenAI LiteLLM data (chain config)
    # gateway_endpoint: The ID of the model group in Stratio GenAI LiteLLM pointing to the desired model.
    chain_config = {
        # Change the endpoint according to the model you will use
        "gateway_endpoint": "gpt-5.4-mini",
        "llm_timeout": 30,
    }
    main(chain_config)
