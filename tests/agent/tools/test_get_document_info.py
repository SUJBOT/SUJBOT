"""
Tests for get_document_info agent tool (VL-only).

Covers input validation, document listing, summary/metadata retrieval,
category lookup, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.tools.get_document_info import GetDocumentInfoInput, GetDocumentInfoTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(vector_store=None):
    """Factory for GetDocumentInfoTool with mocked dependencies."""
    return GetDocumentInfoTool(
        vector_store=vector_store or MagicMock(),
    )


def _mock_pages(document_id, count, with_summaries=False):
    """Create mock page rows as returned by _get_document_pages."""
    pages = []
    for i in range(1, count + 1):
        metadata = {}
        if with_summaries and i <= count // 2:
            metadata["page_summary"] = f"Summary for page {i}"
        pages.append(
            {
                "page_id": f"{document_id}_p{str(i).zfill(3)}",
                "document_id": document_id,
                "page_number": i,
                "metadata": metadata,
            }
        )
    return pages


# ===========================================================================
# TestGetDocumentInfoInput
# ===========================================================================


class TestGetDocumentInfoInput:
    """Pydantic input validation."""

    def test_list_input(self):
        inp = GetDocumentInfoInput(info_type="list")
        assert inp.info_type == "list"
        assert inp.document_id is None

    def test_summary_input(self):
        inp = GetDocumentInfoInput(info_type="summary", document_id="doc1")
        assert inp.info_type == "summary"
        assert inp.document_id == "doc1"

    def test_metadata_input(self):
        inp = GetDocumentInfoInput(info_type="metadata", document_id="doc1")
        assert inp.info_type == "metadata"


# ===========================================================================
# TestGetDocumentInfoList
# ===========================================================================


class TestGetDocumentInfoList:
    """info_type='list' — list all documents."""

    def test_list_returns_documents_with_categories(self):
        vs = MagicMock()
        vs.get_document_list.return_value = ["doc_b", "doc_a"]
        vs.get_document_categories.return_value = {
            "doc_a": "documentation",
            "doc_b": "legislation",
        }

        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(info_type="list")

        assert result.success is True
        docs = result.data["documents"]
        assert len(docs) == 2
        # Sorted alphabetically
        assert docs[0]["id"] == "doc_a"
        assert docs[0]["category"] == "documentation"
        assert docs[1]["id"] == "doc_b"
        assert docs[1]["category"] == "legislation"

    def test_list_with_unknown_category(self):
        """Documents without category get 'unknown'."""
        vs = MagicMock()
        vs.get_document_list.return_value = ["doc1"]
        vs.get_document_categories.return_value = {}

        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(info_type="list")

        assert result.data["documents"][0]["category"] == "unknown"

    def test_list_with_document_id_errors(self):
        """info_type='list' with document_id set returns error."""
        tool = _make_tool()
        result = tool.execute_impl(info_type="list", document_id="doc1")

        assert result.success is False
        assert "document_id=None" in result.error

    def test_list_empty_corpus(self):
        vs = MagicMock()
        vs.get_document_list.return_value = []
        vs.get_document_categories.return_value = {}

        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(info_type="list")

        assert result.success is True
        assert result.data["count"] == 0


# ===========================================================================
# TestGetDocumentInfoSummary
# ===========================================================================


class TestGetDocumentInfoSummary:
    """info_type='summary' — page summaries for a document."""

    def test_summary_with_page_summaries(self):
        vs = MagicMock()
        vs.get_document_categories.return_value = {"doc1": "documentation"}

        tool = _make_tool(vector_store=vs)
        pages = _mock_pages("doc1", 10, with_summaries=True)
        with patch.object(tool, "_get_document_pages", return_value=pages):
            result = tool.execute_impl(info_type="summary", document_id="doc1")

        assert result.success is True
        assert result.data["page_count"] == 10
        assert result.data["category"] == "documentation"
        assert len(result.data["page_summaries"]) == 5  # half have summaries

    def test_summary_requires_document_id(self):
        tool = _make_tool()
        result = tool.execute_impl(info_type="summary")

        assert result.success is False
        assert "document_id is required" in result.error

    def test_summary_document_not_found(self):
        tool = _make_tool()
        with patch.object(tool, "_get_document_pages", return_value=[]):
            result = tool.execute_impl(info_type="summary", document_id="nonexistent")

        assert result.success is True
        assert result.data is None
        assert result.metadata["found"] is False

    def test_summary_caps_at_20_pages(self):
        """Page summaries are capped at 20."""
        vs = MagicMock()
        vs.get_document_categories.return_value = {"doc1": "documentation"}

        tool = _make_tool(vector_store=vs)
        pages = []
        for i in range(1, 31):
            pages.append({
                "page_id": f"doc1_p{str(i).zfill(3)}",
                "document_id": "doc1",
                "page_number": i,
                "metadata": {"page_summary": f"Summary {i}"},
            })
        with patch.object(tool, "_get_document_pages", return_value=pages):
            result = tool.execute_impl(info_type="summary", document_id="doc1")

        assert len(result.data["page_summaries"]) == 20
        assert result.data["has_more"] is True


# ===========================================================================
# TestGetDocumentInfoMetadata
# ===========================================================================


class TestGetDocumentInfoMetadata:
    """info_type='metadata' — page count, category, page numbers."""

    def test_metadata_returns_page_info(self):
        vs = MagicMock()
        vs.get_document_categories.return_value = {"doc1": "legislation"}

        tool = _make_tool(vector_store=vs)
        pages = _mock_pages("doc1", 5, with_summaries=True)
        with patch.object(tool, "_get_document_pages", return_value=pages):
            result = tool.execute_impl(info_type="metadata", document_id="doc1")

        assert result.success is True
        assert result.data["page_count"] == 5
        assert result.data["category"] == "legislation"
        assert result.data["page_numbers"] == [1, 2, 3, 4, 5]
        assert result.data["pages_with_summary"] == 2  # half of 5, rounded down

    def test_invalid_info_type(self):
        tool = _make_tool()
        result = tool.execute_impl(info_type="invalid", document_id="doc1")

        assert result.success is False
        assert "Invalid info_type" in result.error


# ===========================================================================
# TestGetDocumentInfoErrors
# ===========================================================================


class TestGetDocumentInfoErrors:
    """Error handling."""

    def test_database_error_caught(self):
        """Database errors in _get_document_pages are caught by outer handler."""
        tool = _make_tool()
        with patch.object(
            tool, "_get_document_pages", side_effect=RuntimeError("DB connection lost")
        ):
            result = tool.execute_impl(info_type="summary", document_id="doc1")

        assert result.success is False
        assert "DB connection lost" in result.error
