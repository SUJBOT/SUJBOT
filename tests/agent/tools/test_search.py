"""
Tests for search agent tool (VL-only).

Covers input validation, VL retriever interaction, page image loading,
failure handling, and result structure.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agent.tools.search import SearchInput, SearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockSearchResult:
    """Mimics VLSearchResult from vl_retriever."""

    page_id: str
    document_id: str
    page_number: int
    score: float


def _make_tool(vl_retriever=None, page_store=None, vector_store=None):
    """Factory for SearchTool with mocked dependencies."""
    return SearchTool(
        vector_store=vector_store or MagicMock(),
        vl_retriever=vl_retriever or MagicMock(),
        page_store=page_store or MagicMock(),
    )


def _mock_results(count=2):
    """Create mock VL search results."""
    return [
        MockSearchResult(
            page_id=f"doc1_p{str(i+1).zfill(3)}",
            document_id="doc1",
            page_number=i + 1,
            score=0.9 - i * 0.1,
        )
        for i in range(count)
    ]


# ===========================================================================
# TestSearchInput
# ===========================================================================


class TestSearchInput:
    """Pydantic input validation."""

    def test_minimal_input(self):
        inp = SearchInput(query="What is safety?")
        assert inp.query == "What is safety?"
        assert inp.k == 5
        assert inp.filter_document is None
        assert inp.filter_category is None

    def test_full_input(self):
        inp = SearchInput(
            query="test",
            k=10,
            filter_document="doc1",
            filter_category="legislation",
        )
        assert inp.k == 10
        assert inp.filter_document == "doc1"
        assert inp.filter_category == "legislation"

    def test_k_bounds(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", k=0)
        with pytest.raises(ValidationError):
            SearchInput(query="test", k=101)

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            SearchInput(query="test", filter_category="invalid")


# ===========================================================================
# TestSearchExecution
# ===========================================================================


class TestSearchExecution:
    """Core search tool execution."""

    def test_successful_search(self):
        """Basic search returns formatted results with page images."""
        vl = MagicMock()
        vl.search.return_value = _mock_results(2)
        ps = MagicMock()
        ps.get_image_base64.return_value = "base64data"

        tool = _make_tool(vl_retriever=vl, page_store=ps)
        result = tool.execute_impl(query="safety margins")

        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["page_id"] == "doc1_p001"
        assert result.data[0]["score"] == 0.9
        assert len([c for c in result.citations if "doc1" in c]) == 2

    def test_retriever_called_with_filters(self):
        """Filters are passed through to VL retriever."""
        vl = MagicMock()
        vl.search.return_value = []
        tool = _make_tool(vl_retriever=vl)

        tool.execute_impl(
            query="test", k=3, filter_document="BZ_VR1", filter_category="legislation"
        )
        vl.search.assert_called_once_with(
            query="test", k=3, document_filter="BZ_VR1", category_filter="legislation"
        )

    def test_empty_results(self):
        """Empty search returns success with empty data."""
        vl = MagicMock()
        vl.search.return_value = []
        tool = _make_tool(vl_retriever=vl)

        result = tool.execute_impl(query="nonexistent topic")
        assert result.success is True
        assert result.data == []

    def test_page_images_in_metadata(self):
        """Page images are included in metadata for multimodal injection."""
        vl = MagicMock()
        vl.search.return_value = _mock_results(1)
        ps = MagicMock()
        ps.get_image_base64.return_value = "img_b64"

        tool = _make_tool(vl_retriever=vl, page_store=ps)
        result = tool.execute_impl(query="test")

        images = result.metadata["page_images"]
        assert len(images) == 1
        assert images[0]["base64_data"] == "img_b64"
        assert images[0]["media_type"] == "image/png"

    def test_partial_image_failure(self):
        """When some images fail, remaining succeed."""
        vl = MagicMock()
        vl.search.return_value = _mock_results(3)
        ps = MagicMock()
        ps.get_image_base64.side_effect = ["ok1", Exception("disk"), "ok3"]

        tool = _make_tool(vl_retriever=vl, page_store=ps)
        result = tool.execute_impl(query="test")

        assert result.success is True
        assert len(result.metadata["page_images"]) == 2
        assert result.metadata["failed_pages"] == ["doc1_p002"]

    def test_all_images_fail(self):
        """When ALL images fail, returns error."""
        vl = MagicMock()
        vl.search.return_value = _mock_results(2)
        ps = MagicMock()
        ps.get_image_base64.side_effect = Exception("disk error")

        tool = _make_tool(vl_retriever=vl, page_store=ps)
        result = tool.execute_impl(query="test")

        assert result.success is False
        assert "failed to load" in result.error.lower()

    def test_retriever_exception(self):
        """VL retriever exception returns error with type info."""
        vl = MagicMock()
        vl.search.side_effect = ConnectionError("DB down")
        tool = _make_tool(vl_retriever=vl)

        result = tool.execute_impl(query="test")
        assert result.success is False
        assert "ConnectionError" in result.error

    def test_citation_format(self):
        """Citations include document_id, page_number, and score."""
        vl = MagicMock()
        vl.search.return_value = _mock_results(1)
        ps = MagicMock()
        ps.get_image_base64.return_value = "b64"

        tool = _make_tool(vl_retriever=vl, page_store=ps)
        result = tool.execute_impl(query="test")

        assert "doc1" in result.citations[0]
        assert "p.1" in result.citations[0]
        assert "0.900" in result.citations[0]
