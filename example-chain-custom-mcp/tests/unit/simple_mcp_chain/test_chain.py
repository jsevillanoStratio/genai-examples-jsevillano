"""
© 2025 Stratio Big Data Inc., Sucursal en España. All rights reserved.

This software – including all its source code – contains proprietary
information of Stratio Big Data Inc., Sucursal en España and
may not be revealed, sold, transferred, modified, distributed or
otherwise made available, licensed or sublicensed to third parties;
nor reverse engineered, disassembled or decompiled, without express
written authorization from Stratio Big Data Inc., Sucursal en España.
"""

from unittest.mock import MagicMock, patch

import pytest
from genai_core.test.mock_helper import mock_init_litellm_gateway

from custom_mcp_chain.chain import SimpleMcpChain
from custom_mcp_chain.embeddings import StratioLitellmGatewayEmbeddings
from custom_mcp_chain.mcp_tools import (
    AddRecordsResponse,
    AnalyzeTextResponse,
    FormatTextResponse,
    GenerateEmbeddingsResponse,
    GetCurrentTimeResponse,
    MadHatterResponse,
    RunOpenSearchQueryResponse,
    VectorSearchResponse,
    configure_mcp_tools,
)
from custom_mcp_chain.vector_index import (
    build_bulk_body,
    build_knn_query,
    collect_embed_requests,
    embed_for_field,
    get_index_fields,
    hits_to_chunks,
    knn_vector_fields,
    summarize_bulk_response,
)


class TestSimpleMcpChain:
    def test_chain_invoke_returns_text_stats(self):
        c = SimpleMcpChain()
        result = c.chain().invoke({"text": "hello world"})
        assert result["text"] == "hello world"
        assert result["word_count"] == 2
        assert result["char_count"] == 11

    def test_chain_invoke_empty_text(self):
        c = SimpleMcpChain()
        result = c.chain().invoke({"text": ""})
        assert result["text"] == ""
        assert result["word_count"] == 0
        assert result["char_count"] == 0

    def test_chain_invoke_missing_text_key(self):
        c = SimpleMcpChain()
        result = c.chain().invoke({})
        assert result["text"] == ""
        assert result["word_count"] == 0
        assert result["char_count"] == 0

    def test_chain_without_gateway_endpoint_has_no_actor(self):
        c = SimpleMcpChain()
        assert c.mad_hatter_actor is None

    def test_chain_with_gateway_endpoint_builds_actor(self, mocker):
        mock_init_litellm_gateway(mocker)
        c = SimpleMcpChain(gateway_endpoint="test-endpoint")
        assert c.mad_hatter_actor is not None

    def test_chain_without_opensearch_url_has_no_opensearch_service(self):
        c = SimpleMcpChain()
        assert c.opensearch_service is None

    @patch("custom_mcp_chain.chain.OpenSearchService")
    @patch("custom_mcp_chain.chain.VaultClient")
    def test_chain_with_opensearch_url_connects(self, mock_vault, mock_service):
        mock_vault.return_value.get_service_certificate_pem_paths.return_value = (
            "cert",
            "key",
        )
        mock_vault.return_value.get_ca_bundle_pem_path.return_value = "ca"
        mock_service.return_value.test_connection.return_value = (True, "")
        c = SimpleMcpChain(opensearch_url="https://opensearch:9200")
        assert c.opensearch_service is mock_service.return_value
        mock_service.assert_called_once_with(
            opensearch_url="https://opensearch:9200",
            ca_certs="ca",
            client_cert="cert",
            client_key="key",
        )

    @patch("custom_mcp_chain.chain.OpenSearchService")
    @patch("custom_mcp_chain.chain.VaultClient")
    def test_chain_with_unreachable_opensearch_raises(self, mock_vault, mock_service):
        mock_vault.return_value.get_service_certificate_pem_paths.return_value = (
            "cert",
            "key",
        )
        mock_service.return_value.test_connection.return_value = (False, "refused")
        with pytest.raises(RuntimeError, match="refused"):
            SimpleMcpChain(opensearch_url="https://opensearch:9200")

    def test_chain_without_embeddings_endpoint_has_no_embeddings(self):
        c = SimpleMcpChain()
        assert c.embeddings is None

    @patch("custom_mcp_chain.chain.StratioLitellmGatewayEmbeddings")
    def test_chain_with_embeddings_endpoint_builds_embeddings(self, mock_embeddings):
        c = SimpleMcpChain(embeddings_endpoint="text-embedding-3-large", llm_timeout=10)
        assert c.embeddings is mock_embeddings.return_value
        mock_embeddings.assert_called_once_with(
            endpoint="text-embedding-3-large", timeout=10
        )

    def test_mcp_server_returns_none_when_mcp_is_none(self):
        c = SimpleMcpChain()
        assert c.mcp_server(None) is None

    def test_mcp_server_returns_mcp_and_registers_tools(self):
        c = SimpleMcpChain()
        mock_mcp = MagicMock()
        result = c.mcp_server(mock_mcp)
        assert result is mock_mcp
        # configure_mcp_tools calls mcp.tool() three times (one per tool)
        assert mock_mcp.tool.call_count == 3


class TestMcpToolLogic:
    """Tests for the underlying tool logic, bypassing the async wrapper."""

    def test_get_current_time_response_fields(self):
        response = GetCurrentTimeResponse(
            time="2025-01-01T00:00:00+00:00", timezone="UTC"
        )
        assert response.time == "2025-01-01T00:00:00+00:00"
        assert response.timezone == "UTC"
        assert response.error is None

    def test_format_text_response_upper(self):
        response = FormatTextResponse(result="HELLO", style="upper")
        assert response.result == "HELLO"
        assert response.style == "upper"
        assert response.error is None

    def test_mad_hatter_response_fields(self):
        response = MadHatterResponse(
            user_request_explanation="explanation",
            riddle="riddle",
            riddle_explanation="msg",
        )
        assert response.riddle == "riddle"
        assert response.task_id is None
        assert response.error is None

    def test_run_opensearch_query_response_fields(self):
        response = RunOpenSearchQueryResponse(
            index="my-index", total_hits=1, hits=[{"_id": "1"}]
        )
        assert response.index == "my-index"
        assert response.total_hits == 1
        assert response.hits == [{"_id": "1"}]
        assert response.aggregations is None
        assert response.error is None

    def test_generate_embeddings_response_fields(self):
        response = GenerateEmbeddingsResponse(
            model="text-embedding-3-large", dimensions=2, embeddings=[[0.1, 0.2]]
        )
        assert response.model == "text-embedding-3-large"
        assert response.dimensions == 2
        assert response.embeddings == [[0.1, 0.2]]
        assert response.error is None

    def test_analyze_text_response_fields(self):
        response = AnalyzeTextResponse(text="hello world", word_count=2, char_count=11)
        assert response.text == "hello world"
        assert response.word_count == 2
        assert response.char_count == 11
        assert response.error is None

    def test_configure_mcp_tools_registers_three_tools(self):
        mock_mcp = MagicMock()
        mock_chain = MagicMock()
        mock_chain.mad_hatter_actor = None
        mock_chain.opensearch_service = None
        mock_chain.embeddings = None
        configure_mcp_tools(mock_mcp, mock_chain)
        assert mock_mcp.tool.call_count == 3

    def test_configure_mcp_tools_registers_mad_hatter_when_actor_exists(self):
        mock_mcp = MagicMock()
        mock_chain = MagicMock()
        mock_chain.opensearch_service = None
        mock_chain.embeddings = None
        configure_mcp_tools(mock_mcp, mock_chain)
        assert mock_mcp.tool.call_count == 4

    def test_configure_mcp_tools_registers_opensearch_when_service_exists(self):
        mock_mcp = MagicMock()
        mock_chain = MagicMock()
        mock_chain.mad_hatter_actor = None
        mock_chain.embeddings = None
        configure_mcp_tools(mock_mcp, mock_chain)
        # run_opensearch_query, create_index and describe_vector_index
        assert mock_mcp.tool.call_count == 6

    def test_configure_mcp_tools_registers_embeddings_when_model_exists(self):
        mock_mcp = MagicMock()
        mock_chain = MagicMock()
        mock_chain.mad_hatter_actor = None
        mock_chain.opensearch_service = None
        configure_mcp_tools(mock_mcp, mock_chain)
        assert mock_mcp.tool.call_count == 4

    def test_configure_mcp_tools_registers_all_tools(self):
        mock_mcp = MagicMock()
        configure_mcp_tools(mock_mcp, MagicMock())
        assert mock_mcp.tool.call_count == 10

    def test_configure_mcp_tools_passes_annotations(self):
        mock_mcp = MagicMock()
        configure_mcp_tools(mock_mcp, MagicMock())
        annotations = [
            call_args.kwargs.get("annotations", {})
            for call_args in mock_mcp.tool.call_args_list
        ]
        # Only create_index and add_records write; only add_records can replace data
        assert [a["readOnlyHint"] for a in annotations].count(False) == 2
        assert [a["destructiveHint"] for a in annotations].count(True) == 1


class TestStratioLitellmGatewayEmbeddings:
    @pytest.fixture
    def embeddings(self, mocker):
        gateway_client = MagicMock()
        gateway_client.api_key = None
        gateway_client.client_cert_cn = "custom-mcp-chain"
        mocker.patch(
            "custom_mcp_chain.embeddings.LitellmGatewayClient.get_or_create",
            return_value=gateway_client,
        )
        return StratioLitellmGatewayEmbeddings(
            endpoint="text-embedding-3-large", timeout=10
        )

    def test_embed_documents_calls_litellm_embeddings_route(self, embeddings):
        # Items returned out of order to check that they are sorted by index
        embeddings.gateway_client._call_endpoint.return_value = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        vectors = embeddings.embed_documents(
            ["a", "b"], dimensions=2, headers={"x-on-behalf-of": "alice"}
        )
        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        embeddings.gateway_client._call_endpoint.assert_called_once_with(
            url="/embeddings",
            method="POST",
            json_body={
                "model": "text-embedding-3-large",
                "input": ["a", "b"],
                "dimensions": 2,
            },
            headers={"x-on-behalf-of": "alice"},
        )

    def test_embed_documents_sends_api_key_and_splits_in_chunks(self, embeddings):
        embeddings.gateway_client.api_key = "sk-test"
        embeddings.gateway_client._call_endpoint.side_effect = lambda **kwargs: {
            "data": [
                {"index": i, "embedding": [float(i)]}
                for i in range(len(kwargs["json_body"]["input"]))
            ]
        }
        vectors = embeddings.embed_documents([f"t{i}" for i in range(25)])
        assert len(vectors) == 25
        calls = embeddings.gateway_client._call_endpoint.call_args_list
        assert [len(c.kwargs["json_body"]["input"]) for c in calls] == [20, 5]
        assert "dimensions" not in calls[0].kwargs["json_body"]
        assert calls[0].kwargs["headers"] == {"Authorization": "Bearer sk-test"}

    def test_native_dimensions_probes_once(self, embeddings):
        embeddings.gateway_client._call_endpoint.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
        }
        assert embeddings.native_dimensions() == 3
        assert embeddings.native_dimensions() == 3
        embeddings.gateway_client._call_endpoint.assert_called_once()

    def test_embed_query_returns_single_vector(self, embeddings):
        embeddings.gateway_client._call_endpoint.return_value = {
            "data": [{"index": 0, "embedding": [0.5]}]
        }
        assert embeddings.embed_query("a") == [0.5]

    def test_on_behalf_of_headers_for_user(self, embeddings):
        auth = MagicMock()
        auth.extract_uid.return_value = "alice"
        assert embeddings.on_behalf_of_headers(auth) == {"x-on-behalf-of": "alice"}

    def test_on_behalf_of_headers_skipped_for_chain_identity(self, embeddings):
        auth = MagicMock()
        auth.extract_uid.return_value = "custom-mcp-chain"
        assert embeddings.on_behalf_of_headers(auth) == {}
        assert embeddings.on_behalf_of_headers(None) == {}


class TestVectorIndex:
    MAPPING = {
        "docs": {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "meta": {"properties": {"chapter": {"type": "integer"}}},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 3,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": "cosinesimil",
                        },
                    },
                }
            }
        }
    }
    VECTOR_FIELDS = {
        "embedding": {
            "type": "knn_vector",
            "dimension": 3,
            "space_type": "cosinesimil",
            "engine": "lucene",
        }
    }

    def test_get_index_fields_reads_vector_and_nested_fields(self):
        client = MagicMock()
        client.indices.get_mapping.return_value = self.MAPPING
        fields = get_index_fields(client, "docs")
        assert fields["text"] == {"type": "text"}
        assert fields["meta.chapter"] == {"type": "integer"}
        assert knn_vector_fields(fields) == self.VECTOR_FIELDS

    def test_get_index_fields_rejects_inconsistent_indices(self):
        other = {"mappings": {"properties": {"text": {"type": "keyword"}}}}
        client = MagicMock()
        client.indices.get_mapping.return_value = {**self.MAPPING, "docs-2": other}
        with pytest.raises(ValueError, match="different mappings"):
            get_index_fields(client, "docs*")

    def test_collect_embed_requests_resolves_pointers(self):
        records = [
            {"text": "a", "embedding": {"$embed": "text"}},
            {"text": "b", "embedding": [0.1, 0.2, 0.3]},
            {"title": "t", "text": "c", "embedding": {"$embed": ["title", "text"]}},
        ]
        assert collect_embed_requests(records, self.VECTOR_FIELDS) == [
            (0, "embedding", "a"),
            (2, "embedding", "t\nc"),
        ]

    def test_collect_embed_requests_rejects_non_vector_field(self):
        records = [{"text": "a", "other": {"$embed": "text"}}]
        with pytest.raises(ValueError, match="'other' is not a knn_vector"):
            collect_embed_requests(records, self.VECTOR_FIELDS)

    def test_collect_embed_requests_rejects_missing_text(self):
        records = [{"embedding": {"$embed": "text"}}]
        with pytest.raises(ValueError, match="'text' referenced by 'embedding'"):
            collect_embed_requests(records, self.VECTOR_FIELDS)

    def test_embed_for_field_sends_dimensions_only_when_needed(self):
        embeddings = MagicMock()
        embeddings.native_dimensions.return_value = 3
        embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        embed_for_field(embeddings, ["a"], 3, {})
        assert embeddings.embed_documents.call_args.kwargs["dimensions"] is None
        embeddings.native_dimensions.return_value = 3072
        embed_for_field(embeddings, ["a"], 3, {})
        assert embeddings.embed_documents.call_args.kwargs["dimensions"] == 3

    def test_embed_for_field_rejects_wrong_dimension(self):
        embeddings = MagicMock()
        embeddings.native_dimensions.return_value = 2
        embeddings.embed_documents.return_value = [[0.1, 0.2]]
        with pytest.raises(ValueError, match="expects 3"):
            embed_for_field(embeddings, ["a"], 3, {})

    def test_build_bulk_body_uses_id_key(self):
        body = build_bulk_body("docs", [{"_id": 1, "text": "a"}, {"text": "b"}])
        assert body == [
            {"index": {"_index": "docs", "_id": "1"}},
            {"text": "a"},
            {"index": {"_index": "docs"}},
            {"text": "b"},
        ]

    def test_summarize_bulk_response(self):
        response = {
            "items": [
                {"index": {"_id": "1", "result": "created"}},
                {"index": {"_id": "2", "error": {"type": "mapper_parsing_exception"}}},
            ]
        }
        ids, failures = summarize_bulk_response(response)
        assert ids == ["1"]
        assert failures == [
            {"position": 1, "id": "2", "error": {"type": "mapper_parsing_exception"}}
        ]

    def test_build_knn_query(self):
        body = build_knn_query(
            "embedding",
            [0.1],
            3,
            min_score=0.8,
            filter_query={"term": {"meta.chapter": 1}},
            excluded_fields=["embedding"],
        )
        assert body == {
            "size": 3,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": [0.1],
                        "k": 3,
                        "filter": {"term": {"meta.chapter": 1}},
                    }
                }
            },
            "_source": {"includes": ["*"], "excludes": ["embedding"]},
            "min_score": 0.8,
        }

    def test_hits_to_chunks_splits_text_and_metadata(self):
        hits = [
            {
                "_id": "1",
                "_index": "docs",
                "_score": 0.9,
                "_source": {"text": "a", "page": 2},
            }
        ]
        assert hits_to_chunks(hits, "text") == [
            {
                "id": "1",
                "index": "docs",
                "score": 0.9,
                "text": "a",
                "metadata": {"page": 2},
            }
        ]

    def test_vector_tool_responses(self):
        assert AddRecordsResponse(index="docs", indexed=1, ids=["1"]).task_id is None
        response = VectorSearchResponse(
            index="docs", vector_field="embedding", chunks=[]
        )
        assert response.error is None


if __name__ == "__main__":
    pytest.main()
