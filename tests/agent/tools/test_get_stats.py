"""
Tests for get_stats agent tool.

Covers input validation, stat scopes (corpus, index, document),
vector store interaction, and error handling.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agent.tools.get_stats import GetStatsInput, GetStatsTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(vector_store=None):
    """Factory for GetStatsTool with mocked dependencies."""
    return GetStatsTool(
        vector_store=vector_store or MagicMock(),
    )


def _mock_vs(doc_count=5, page_count=50, doc_list=None):
    """Create a vector store mock with configurable stats."""
    vs = MagicMock()
    vs.get_stats.return_value = {
        "documents": doc_count,
        "pages": page_count,
    }
    vs.get_document_list.return_value = doc_list or [f"doc{i}" for i in range(doc_count)]
    return vs


# ===========================================================================
# TestGetStatsInput
# ===========================================================================


class TestGetStatsInput:
    """Pydantic input validation."""

    def test_valid_scopes(self):
        for scope in ("corpus", "index", "document"):
            inp = GetStatsInput(stat_scope=scope)
            assert inp.stat_scope == scope

    def test_stat_scope_required(self):
        with pytest.raises(ValidationError):
            GetStatsInput()


# ===========================================================================
# TestGetStatsExecution
# ===========================================================================


class TestGetStatsExecution:
    """Core get_stats execution."""

    def test_corpus_scope(self):
        """Corpus scope returns document counts and list."""
        vs = _mock_vs(doc_count=3)
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="corpus")

        assert result.success is True
        assert result.data["unique_documents"] == 3
        assert len(result.data["document_list"]) == 3
        assert result.metadata["stat_scope"] == "corpus"

    def test_index_scope(self):
        """Index scope includes VL architecture info."""
        vs = _mock_vs(doc_count=5, doc_list=["a", "b", "c", "d", "e"])
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="index")

        assert result.success is True
        assert result.data["architecture"] == "vl"
        assert result.data["embedding_model"] == "jina-embeddings-v4"
        assert result.data["embedding_dimensions"] == 2048
        assert result.data["documents"]["count"] == 5

    def test_document_scope(self):
        """Document scope returns per-document info."""
        vs = _mock_vs(doc_count=2, doc_list=["doc_a", "doc_b"])
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="document")

        assert result.success is True
        assert result.data["document_list"] == ["doc_a", "doc_b"]

    def test_invalid_scope(self):
        """Invalid stat_scope returns error."""
        tool = _make_tool()
        result = tool.execute_impl(stat_scope="invalid")

        assert result.success is False
        assert "Invalid stat_scope" in result.error

    def test_empty_corpus(self):
        """Empty corpus returns zero counts."""
        vs = _mock_vs(doc_count=0, doc_list=[])
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="corpus")

        assert result.success is True
        assert result.data["unique_documents"] == 0
        assert result.data["document_list"] == []

    def test_document_list_sorted(self):
        """Document list is sorted alphabetically."""
        vs = _mock_vs(doc_list=["zebra", "alpha", "middle"])
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="corpus")

        assert result.data["document_list"] == ["alpha", "middle", "zebra"]

    def test_vector_store_error(self):
        """Vector store error is caught and returned."""
        vs = MagicMock()
        vs.get_stats.side_effect = RuntimeError("connection lost")
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="corpus")

        assert result.success is False
        assert "connection lost" in result.error

    def test_index_scope_has_all_keys(self):
        """Index scope includes required architecture keys."""
        vs = _mock_vs()
        tool = _make_tool(vector_store=vs)
        result = tool.execute_impl(stat_scope="index")

        required_keys = {"architecture", "embedding_model", "embedding_dimensions", "documents"}
        assert required_keys.issubset(result.data.keys())
