"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

from typing import Any, Optional

from genai_core.clients.gateway.litellm_gateway_client import LitellmGatewayClient
from genai_core.constants.constants import EMBEDDINGS_STRATIO_CHUNK_SIZE
from genai_core.runnables.genai_auth import GenAiAuth
from langchain_core.embeddings import Embeddings


# genai-core 0.7 does not ship an embeddings model for Stratio GenAI LiteLLM, only
# the chat model StratioLitellmGatewayChat. This class calls the OpenAI-compatible
# /embeddings route of LiteLLM with the shared LitellmGatewayClient of genai-core,
# which is configured (URL, mTLS certificates and API key) from the GENAI_LITELLM_*
# environment variables.
class StratioLitellmGatewayEmbeddings(Embeddings):
    """LangChain Embeddings backed by an embedding model group in Stratio GenAI LiteLLM."""

    def __init__(self, endpoint: str, timeout: int = 30):
        """
        :param endpoint: The ID of the model group in Stratio GenAI LiteLLM pointing to
            the embedding model (e.g. "text-embedding-3-large").
        :param timeout: Timeout for the requests, in seconds. The gateway client is shared
            per LiteLLM URL, so it only applies if this is the first client created.
        """
        self.model = endpoint
        self._native_dimensions: Optional[int] = None
        # => Shared LiteLLM Gateway client (singleton per gateway URL)
        try:
            self.gateway_client = LitellmGatewayClient.get_or_create(
                model_group=endpoint, timeout=timeout
            )
        except Exception as e:
            raise ValueError(f"Failed to create the gateway client. Cause: {e}") from e

    def embed_documents(
        self,
        texts: list[str],
        dimensions: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> list[list[float]]:
        """Returns one embedding per text, in the same order.

        :param texts: The texts to embed.
        :param dimensions: Optional number of dimensions of the output vectors.
        :param headers: Additional request headers (e.g. x-on-behalf-of).
        :return: The embeddings.
        """
        request_headers = dict(headers or {})
        # Same as LitellmGatewayClient._request_keos: Bearer token only with an API key
        if self.gateway_client.api_key:
            request_headers["Authorization"] = f"Bearer {self.gateway_client.api_key}"

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDINGS_STRATIO_CHUNK_SIZE):
            body: dict[str, Any] = {
                "model": self.model,
                "input": texts[i : i + EMBEDDINGS_STRATIO_CHUNK_SIZE],
            }
            if dimensions is not None:
                body["dimensions"] = dimensions
            # pylint: disable=protected-access
            response = self.gateway_client._call_endpoint(
                url="/embeddings",
                method="POST",
                json_body=body,
                headers=request_headers or None,
            )
            data = sorted(response["data"], key=lambda item: item["index"])
            embeddings.extend(item["embedding"] for item in data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def native_dimensions(self, headers: Optional[dict[str, str]] = None) -> int:
        """Returns the default number of dimensions of the model's vectors.

        It is obtained by embedding a probe text the first time and then cached.

        :param headers: Additional request headers (e.g. x-on-behalf-of).
        :return: The number of dimensions.
        """
        if self._native_dimensions is None:
            probe = self.embed_documents(["dimensions probe"], headers=headers)
            self._native_dimensions = len(probe[0])
        return self._native_dimensions

    def on_behalf_of_headers(self, genai_auth: Optional[GenAiAuth]) -> dict[str, str]:
        """Returns the x-on-behalf-of header so LiteLLM attributes the call to the user.

        Same logic as StratioLitellmGatewayChat.map_invocation_kwargs: the header is
        skipped when there is no user or the user is the chain itself (certificate CN).

        :param genai_auth: The GenAiAuth of the nominal user.
        :return: The extra headers to send with the embedding request.
        """
        user_id = genai_auth.extract_uid() if genai_auth else None
        if not user_id or user_id == self.gateway_client.client_cert_cn:
            return {}
        return {"x-on-behalf-of": user_id}
