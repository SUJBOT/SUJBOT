"""
Tests for web_search tool (src/agent/tools/web_search.py).

Covers:
- Input validation: valid/invalid queries
- Disabled state: returns error when web_search_enabled=False
- Missing API key: returns error
- Tool result format: check ToolResult structure with mocked Gemini response
- Citation format: verify sources list format
- Redirect URL resolution: resolve Gemini redirect URLs to actual page URLs
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.tools.web_search import WebSearchTool, WebSearchInput
from src.agent.tools._base import ToolResult
from src.agent.config import ToolConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(web_search_enabled=True, web_search_model="gemini-2.0-flash"):
    """Create a WebSearchTool with a mock config."""
    config = ToolConfig(
        web_search_enabled=web_search_enabled,
        web_search_model=web_search_model,
    )
    return WebSearchTool(
        vector_store=MagicMock(),
        config=config,
    )


def _make_grounding_response(text="Answer text", sources=None):
    """Create a mock Gemini response with grounding metadata."""
    if sources is None:
        sources = [
            {"uri": "https://example.com/1", "title": "Source One"},
            {"uri": "https://example.com/2", "title": "Source Two"},
        ]

    # Build grounding chunks
    chunks = []
    for s in sources:
        web = MagicMock()
        web.uri = s["uri"]
        web.title = s["title"]
        chunk = MagicMock()
        chunk.web = web
        chunks.append(chunk)

    grounding_metadata = MagicMock()
    grounding_metadata.web_search_queries = ["test query"]
    grounding_metadata.grounding_chunks = chunks

    candidate = MagicMock()
    candidate.grounding_metadata = grounding_metadata

    response = MagicMock()
    response.text = text
    response.candidates = [candidate]

    return response


def _passthrough_resolve(raw_sources):
    """Identity function — returns raw sources unchanged (skip redirect resolution)."""
    return raw_sources


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestWebSearchInput:
    def test_valid_query(self):
        inp = WebSearchInput(query="What is IAEA?")
        assert inp.query == "What is IAEA?"

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            WebSearchInput(query="")

    def test_long_query_rejected(self):
        with pytest.raises(Exception):
            WebSearchInput(query="x" * 501)

    def test_max_length_query_accepted(self):
        inp = WebSearchInput(query="x" * 500)
        assert len(inp.query) == 500


# ---------------------------------------------------------------------------
# Disabled state
# ---------------------------------------------------------------------------


class TestWebSearchDisabled:
    def test_disabled_returns_error(self):
        tool = _make_tool(web_search_enabled=False)
        result = tool.execute(query="test query")
        assert not result.success
        assert "disabled" in result.error.lower()


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


class TestWebSearchMissingKey:
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_returns_error(self):
        tool = _make_tool()
        result = tool.execute_impl(query="test query")
        assert not result.success
        assert "GOOGLE_API_KEY" in result.error


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------


class TestWebSearchExecution:
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key-1234567890123456789012345678901234"})
    @patch.object(WebSearchTool, "_resolve_redirect_urls", side_effect=_passthrough_resolve)
    @patch("src.agent.tools.web_search.genai")
    def test_successful_search(self, mock_genai, _mock_resolve):
        """Test successful web search with mocked Gemini response."""
        mock_response = _make_grounding_response(
            text="IAEA is the International Atomic Energy Agency.",
            sources=[
                {"uri": "https://iaea.org", "title": "IAEA"},
                {"uri": "https://en.wikipedia.org/wiki/IAEA", "title": "IAEA - Wikipedia"},
            ],
        )

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        tool = _make_tool()
        result = tool.execute_impl(query="What is IAEA?")

        assert result.success
        assert "IAEA" in result.data["answer"]
        assert len(result.data["sources"]) == 2
        assert result.data["sources"][0]["url"] == "https://iaea.org"
        assert result.data["sources"][0]["title"] == "IAEA"
        assert result.data["sources"][1]["index"] == 2
        assert result.metadata["query"] == "What is IAEA?"
        assert result.metadata["source_count"] == 2

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key-1234567890123456789012345678901234"})
    @patch.object(WebSearchTool, "_resolve_redirect_urls", side_effect=_passthrough_resolve)
    @patch("src.agent.tools.web_search.genai")
    def test_search_with_no_sources(self, mock_genai, _mock_resolve):
        """Test web search when Gemini returns no grounding chunks."""
        mock_response = _make_grounding_response(text="Some answer", sources=[])

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        tool = _make_tool()
        result = tool.execute_impl(query="test")

        assert result.success
        assert result.data["sources"] == []
        assert result.metadata["source_count"] == 0

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key-1234567890123456789012345678901234"})
    @patch("src.agent.tools.web_search.genai")
    def test_search_api_error(self, mock_genai):
        """Test graceful handling of API errors."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")

        tool = _make_tool()
        result = tool.execute_impl(query="test")

        assert not result.success
        assert "API quota exceeded" in result.error


# ---------------------------------------------------------------------------
# Citation format
# ---------------------------------------------------------------------------


class TestWebSearchCitationFormat:
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key-1234567890123456789012345678901234"})
    @patch.object(WebSearchTool, "_resolve_redirect_urls", side_effect=_passthrough_resolve)
    @patch("src.agent.tools.web_search.genai")
    def test_citation_instruction_in_answer(self, mock_genai, _mock_resolve):
        """Test that citation instruction is appended to the answer."""
        mock_response = _make_grounding_response(
            text="Test answer.",
            sources=[{"uri": "https://example.com", "title": "Example"}],
        )

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        tool = _make_tool()
        result = tool.execute_impl(query="test")

        assert "\\webcite{" in result.data["answer"]
        assert "https://example.com" in result.data["answer"]
        assert "Example" in result.data["answer"]


# ---------------------------------------------------------------------------
# ToolConfig integration
# ---------------------------------------------------------------------------


class TestWebSearchConfig:
    def test_tool_config_defaults(self):
        config = ToolConfig()
        assert config.web_search_enabled is True
        assert config.web_search_model == "gemini-2.0-flash"

    def test_tool_config_custom(self):
        config = ToolConfig(web_search_enabled=False, web_search_model="gemini-2.5-flash")
        assert config.web_search_enabled is False
        assert config.web_search_model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Redirect URL resolution
# ---------------------------------------------------------------------------


class TestRedirectResolution:
    def test_empty_sources(self):
        """Empty input returns empty output."""
        result = WebSearchTool._resolve_redirect_urls([])
        assert result == []

    @patch("src.agent.tools.web_search.requests")
    def test_resolves_redirect_to_real_url(self, mock_requests):
        """Redirect URL gets resolved to the actual page URL."""
        mock_resp = MagicMock()
        mock_resp.url = "https://sujb.gov.cz/jaderna-bezpecnost/legislativa"
        mock_requests.head.return_value = mock_resp

        raw = [{"url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/ABC", "title": "sujb.gov.cz"}]
        result = WebSearchTool._resolve_redirect_urls(raw)

        assert len(result) == 1
        assert result[0]["url"] == "https://sujb.gov.cz/jaderna-bezpecnost/legislativa"
        assert "sujb.gov.cz" in result[0]["title"]
        assert "legislativa" in result[0]["title"]

    @patch("src.agent.tools.web_search.requests")
    def test_preserves_specific_title(self, mock_requests):
        """When Gemini provides a title that is NOT just the domain, preserve it."""
        mock_resp = MagicMock()
        mock_resp.url = "https://example.com/some-page"
        mock_requests.head.return_value = mock_resp

        raw = [{"url": "https://redirect.example.com/xyz", "title": "IAEA Safety Standards"}]
        result = WebSearchTool._resolve_redirect_urls(raw)

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/some-page"
        assert result[0]["title"] == "IAEA Safety Standards"  # kept because != domain

    @patch("src.agent.tools.web_search.requests")
    def test_fallback_on_timeout(self, mock_requests):
        """If redirect resolution fails, fall back to original URL."""
        mock_requests.head.side_effect = Exception("Connection timeout")

        raw = [{"url": "https://vertexaisearch.cloud.google.com/redirect/X", "title": "example.com"}]
        result = WebSearchTool._resolve_redirect_urls(raw)

        assert len(result) == 1
        assert result[0]["url"] == "https://vertexaisearch.cloud.google.com/redirect/X"
        assert result[0]["title"] == "example.com"

    @patch("src.agent.tools.web_search.requests")
    def test_title_from_url_path(self, mock_requests):
        """When title is just a domain, build a better title from the URL path."""
        mock_resp = MagicMock()
        mock_resp.url = "https://ekolist.cz/cz/zpravodajstvi/zpravy/sujb-povolil-jaderne-elektrarne"
        mock_requests.head.return_value = mock_resp

        raw = [{"url": "https://redirect.example.com/abc", "title": "ekolist.cz"}]
        result = WebSearchTool._resolve_redirect_urls(raw)

        assert result[0]["url"] == "https://ekolist.cz/cz/zpravodajstvi/zpravy/sujb-povolil-jaderne-elektrarne"
        assert "ekolist.cz" in result[0]["title"]
        assert "sujb povolil jaderne elektrarne" in result[0]["title"]

    @patch("src.agent.tools.web_search.requests")
    def test_title_strips_file_extension(self, mock_requests):
        """PDF/HTML extensions are stripped from auto-generated titles."""
        mock_resp = MagicMock()
        mock_resp.url = "https://sujb.gov.cz/docs/safety-report.pdf"
        mock_requests.head.return_value = mock_resp

        raw = [{"url": "https://redirect.example.com/def", "title": "sujb.gov.cz"}]
        result = WebSearchTool._resolve_redirect_urls(raw)

        assert "safety report" in result[0]["title"]
        assert ".pdf" not in result[0]["title"]

    @patch("src.agent.tools.web_search.requests")
    def test_multiple_sources_resolved_in_parallel(self, mock_requests):
        """Multiple sources are resolved (order preserved)."""
        def _head(url, **kwargs):
            resp = MagicMock()
            if "AAA" in url:
                resp.url = "https://first.com/page-one"
            else:
                resp.url = "https://second.com/page-two"
            return resp

        mock_requests.head.side_effect = _head

        raw = [
            {"url": "https://redirect.example.com/AAA", "title": "first.com"},
            {"url": "https://redirect.example.com/BBB", "title": "second.com"},
        ]
        result = WebSearchTool._resolve_redirect_urls(raw)

        assert len(result) == 2
        assert result[0]["url"] == "https://first.com/page-one"
        assert result[1]["url"] == "https://second.com/page-two"
