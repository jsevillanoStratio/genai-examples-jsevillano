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
from custom_mcp_chain.mcp_tools import (
    AnalyzeTextResponse,
    FormatTextResponse,
    GetCurrentTimeResponse,
    MadHatterResponse,
    RunOpenSearchQueryResponse,
    configure_mcp_tools,
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
        configure_mcp_tools(mock_mcp, mock_chain)
        assert mock_mcp.tool.call_count == 3

    def test_configure_mcp_tools_registers_mad_hatter_when_actor_exists(self):
        mock_mcp = MagicMock()
        mock_chain = MagicMock()
        mock_chain.opensearch_service = None
        configure_mcp_tools(mock_mcp, mock_chain)
        assert mock_mcp.tool.call_count == 4

    def test_configure_mcp_tools_registers_opensearch_when_service_exists(self):
        mock_mcp = MagicMock()
        mock_chain = MagicMock()
        mock_chain.mad_hatter_actor = None
        configure_mcp_tools(mock_mcp, mock_chain)
        assert mock_mcp.tool.call_count == 4

    def test_configure_mcp_tools_registers_all_tools(self):
        mock_mcp = MagicMock()
        configure_mcp_tools(mock_mcp, MagicMock())
        assert mock_mcp.tool.call_count == 5

    def test_configure_mcp_tools_passes_readonly_annotations(self):
        mock_mcp = MagicMock()
        configure_mcp_tools(mock_mcp, MagicMock())
        for call_args in mock_mcp.tool.call_args_list:
            annotations = call_args.kwargs.get("annotations", {})
            assert annotations.get("readOnlyHint") is True
            assert annotations.get("destructiveHint") is False


if __name__ == "__main__":
    pytest.main()
