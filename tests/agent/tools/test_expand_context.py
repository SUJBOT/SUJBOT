"""
Tests for expand_context agent tool (VL-only).

Covers input validation, page ID parsing, adjacent page retrieval,
image loading, and failure handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.tools.expand_context import ExpandContextInput, ExpandContextTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(vector_store=None, page_store=None):
    """Factory for ExpandContextTool with mocked dependencies."""
    return ExpandContextTool(
        vector_store=vector_store or MagicMock(),
        page_store=page_store or MagicMock(),
    )


def _adjacent_pages(document_id, center, k):
    """Create mock adjacent page dicts."""
    pages = []
    for offset in range(-k, k + 1):
        if offset == 0:
            continue
        pn = center + offset
        if pn < 1:
            continue
        pages.append(
            {
                "page_id": f"{document_id}_p{str(pn).zfill(3)}",
                "document_id": document_id,
                "page_number": pn,
            }
        )
    return pages


# ===========================================================================
# TestExpandContextInput
# ===========================================================================


class TestExpandContextInput:
    """Pydantic input validation."""

    def test_minimal_input(self):
        inp = ExpandContextInput(page_ids=["doc1_p005"])
        assert inp.page_ids == ["doc1_p005"]
        assert inp.k == 3

    def test_custom_k(self):
        inp = ExpandContextInput(page_ids=["doc1_p001"], k=5)
        assert inp.k == 5

    def test_k_bounds(self):
        with pytest.raises(ValidationError):
            ExpandContextInput(page_ids=["x"], k=0)
        with pytest.raises(ValidationError):
            ExpandContextInput(page_ids=["x"], k=11)

    def test_multiple_page_ids(self):
        inp = ExpandContextInput(page_ids=["doc1_p001", "doc2_p010"])
        assert len(inp.page_ids) == 2


# ===========================================================================
# TestExpandContextExecution
# ===========================================================================


class TestExpandContextExecution:
    """Core expand_context execution."""

    @patch("src.vl.page_store.PageStore")
    def test_successful_expansion(self, mock_page_store_cls):
        """Basic expansion returns adjacent pages with images."""
        mock_page_store_cls.page_id_to_components.return_value = ("doc1", 5)

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = _adjacent_pages("doc1", 5, 2)
        ps = MagicMock()
        ps.get_image_base64.return_value = "b64data"

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["doc1_p005"], k=2)

        assert result.success is True
        assert len(result.data["expansions"]) == 1
        assert result.data["expansions"][0]["target_page_id"] == "doc1_p005"
        assert result.data["expansions"][0]["expansion_count"] == 4  # 2 before + 2 after
        assert result.metadata["images_loaded"] == 4

    @patch("src.vl.page_store.PageStore")
    def test_invalid_page_id_skipped(self, mock_page_store_cls):
        """Invalid page ID format is skipped with warning."""
        mock_page_store_cls.page_id_to_components.side_effect = ValueError("bad format")

        tool = _make_tool()
        result = tool.execute_impl(page_ids=["invalid_format"])

        assert result.success is False
        assert "No valid page IDs" in result.error

    @patch("src.vl.page_store.PageStore")
    def test_mixed_valid_invalid_page_ids(self, mock_page_store_cls):
        """Mix of valid and invalid page IDs — valid ones succeed."""
        def parse_side_effect(page_id):
            if page_id == "doc1_p005":
                return ("doc1", 5)
            raise ValueError("bad format")

        mock_page_store_cls.page_id_to_components.side_effect = parse_side_effect

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = _adjacent_pages("doc1", 5, 1)
        ps = MagicMock()
        ps.get_image_base64.return_value = "b64"

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["bad_id", "doc1_p005"])

        assert result.success is True
        assert len(result.data["expansions"]) == 1

    @patch("src.vl.page_store.PageStore")
    def test_all_images_fail(self, mock_page_store_cls):
        """When all image loads fail, returns error."""
        mock_page_store_cls.page_id_to_components.return_value = ("doc1", 5)

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = _adjacent_pages("doc1", 5, 1)
        ps = MagicMock()
        ps.get_image_base64.side_effect = Exception("disk error")

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["doc1_p005"])

        assert result.success is False
        assert "image loads failed" in result.error

    @patch("src.vl.page_store.PageStore")
    def test_partial_image_failure(self, mock_page_store_cls):
        """When some images fail, succeeds with loaded ones."""
        mock_page_store_cls.page_id_to_components.return_value = ("doc1", 5)

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = _adjacent_pages("doc1", 5, 1)
        ps = MagicMock()
        ps.get_image_base64.side_effect = ["ok", Exception("fail")]

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["doc1_p005"])

        assert result.success is True
        assert result.metadata["images_loaded"] == 1
        assert result.metadata["images_expected"] == 2

    @patch("src.vl.page_store.PageStore")
    def test_position_labels(self, mock_page_store_cls):
        """Pages before center are 'before', after are 'after'."""
        mock_page_store_cls.page_id_to_components.return_value = ("doc1", 5)

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = [
            {"page_id": "doc1_p004", "document_id": "doc1", "page_number": 4},
            {"page_id": "doc1_p006", "document_id": "doc1", "page_number": 6},
        ]
        ps = MagicMock()
        ps.get_image_base64.return_value = "b64"

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["doc1_p005"])

        images = result.metadata["page_images"]
        positions = {img["page_id"]: img["position"] for img in images}
        assert positions["doc1_p004"] == "before"
        assert positions["doc1_p006"] == "after"

    @patch("src.vl.page_store.PageStore")
    def test_citations_from_document_ids(self, mock_page_store_cls):
        """Citations contain unique document IDs."""
        mock_page_store_cls.page_id_to_components.return_value = ("doc1", 5)

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = _adjacent_pages("doc1", 5, 1)
        ps = MagicMock()
        ps.get_image_base64.return_value = "b64"

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["doc1_p005"])

        assert "doc1" in result.citations

    @patch("src.vl.page_store.PageStore")
    def test_no_adjacent_pages(self, mock_page_store_cls):
        """No adjacent pages found (e.g., single-page document)."""
        mock_page_store_cls.page_id_to_components.return_value = ("doc1", 1)

        vs = MagicMock()
        vs.get_adjacent_vl_pages.return_value = []
        ps = MagicMock()

        tool = _make_tool(vector_store=vs, page_store=ps)
        result = tool.execute_impl(page_ids=["doc1_p001"])

        assert result.success is True
        assert result.data["expansions"][0]["expansion_count"] == 0
