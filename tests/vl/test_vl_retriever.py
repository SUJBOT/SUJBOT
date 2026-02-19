"""
Tests for VLRetriever.search() and VLPageResult validation.
"""

import pytest
from unittest.mock import MagicMock

import numpy as np

from src.retrieval.adaptive_k import AdaptiveKConfig
from src.vl.vl_retriever import VLPageResult, VLRetriever


# ---------------------------------------------------------------------------
# VLPageResult validation tests
# ---------------------------------------------------------------------------


class TestVLPageResult:
    """Test frozen dataclass validation in __post_init__."""

    def test_valid_construction(self):
        result = VLPageResult(
            page_id="DOC_p001",
            document_id="DOC",
            page_number=1,
            score=0.85,
            image_path="/tmp/page.png",
        )
        assert result.page_id == "DOC_p001"
        assert result.score == 0.85

    def test_boundary_score_zero(self):
        result = VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=1, score=0.0)
        assert result.score == 0.0

    def test_boundary_score_one(self):
        result = VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=1, score=1.0)
        assert result.score == 1.0

    def test_negative_score_clamped(self):
        result = VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=1, score=-0.1)
        assert result.score == 0.0

    def test_score_above_one_clamped(self):
        result = VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=1, score=1.01)
        assert result.score == 1.0

    def test_page_number_zero_raises(self):
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            VLPageResult(page_id="DOC_p000", document_id="DOC", page_number=0, score=0.5)

    def test_negative_page_number_raises(self):
        with pytest.raises(ValueError, match="page_number must be >= 1"):
            VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=-1, score=0.5)

    def test_frozen_immutability(self):
        result = VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=1, score=0.5)
        with pytest.raises(AttributeError):
            result.score = 0.9

    def test_image_path_defaults_to_none(self):
        result = VLPageResult(page_id="DOC_p001", document_id="DOC", page_number=1, score=0.5)
        assert result.image_path is None


# ---------------------------------------------------------------------------
# VLRetriever.search() tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_jina():
    client = MagicMock()
    client.embed_query.return_value = np.zeros(2048, dtype=np.float32)
    return client


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search_vl_pages.return_value = [
        {
            "page_id": "DOC_p001",
            "document_id": "DOC",
            "page_number": 1,
            "score": 0.92,
            "image_path": "/data/vl_pages/DOC_p001.png",
        },
        {
            "page_id": "DOC_p002",
            "document_id": "DOC",
            "page_number": 2,
            "score": 0.87,
            "image_path": "/data/vl_pages/DOC_p002.png",
        },
    ]
    return store


@pytest.fixture
def mock_page_store():
    store = MagicMock()
    store.get_image_path.side_effect = lambda pid: f"/rendered/{pid}.png"
    return store


@pytest.fixture
def retriever(mock_jina, mock_vector_store, mock_page_store):
    return VLRetriever(
        jina_client=mock_jina,
        vector_store=mock_vector_store,
        page_store=mock_page_store,
        default_k=5,
    )


class TestVLRetrieverSearch:
    """Tests for VLRetriever.search() orchestration."""

    def test_basic_search_returns_results(self, retriever, mock_jina, mock_vector_store):
        results = retriever.search("test query")

        assert len(results) == 2
        assert all(isinstance(r, VLPageResult) for r in results)
        mock_jina.embed_query.assert_called_once_with("test query")
        mock_vector_store.search_vl_pages.assert_called_once()

    def test_uses_default_k(self, retriever, mock_vector_store):
        retriever.search("query")
        call_kwargs = mock_vector_store.search_vl_pages.call_args
        assert call_kwargs.kwargs["k"] == 5

    def test_custom_k_overrides_default(self, retriever, mock_vector_store):
        retriever.search("query", k=3)
        call_kwargs = mock_vector_store.search_vl_pages.call_args
        assert call_kwargs.kwargs["k"] == 3

    def test_document_filter_passed_through(self, retriever, mock_vector_store):
        retriever.search("query", document_filter="BZ_VR1")
        call_kwargs = mock_vector_store.search_vl_pages.call_args
        assert call_kwargs.kwargs["document_filter"] == "BZ_VR1"

    def test_image_path_resolved_from_page_store(self, retriever, mock_page_store):
        results = retriever.search("query")
        # page_store.get_image_path should override the DB path
        assert results[0].image_path == "/rendered/DOC_p001.png"
        assert results[1].image_path == "/rendered/DOC_p002.png"
        assert mock_page_store.get_image_path.call_count == 2

    def test_page_store_error_falls_back_to_db_path(self, retriever, mock_page_store):
        """If page_store.get_image_path raises, fall back to DB image_path."""
        mock_page_store.get_image_path.side_effect = OSError("disk error")
        results = retriever.search("query")
        # Should fall back to the DB-stored path
        assert results[0].image_path == "/data/vl_pages/DOC_p001.png"
        assert results[1].image_path == "/data/vl_pages/DOC_p002.png"

    def test_empty_results(self, retriever, mock_vector_store):
        mock_vector_store.search_vl_pages.return_value = []
        results = retriever.search("obscure query")
        assert results == []

    def test_result_ordering_preserved(self, retriever):
        """Results should maintain the order from vector store (by score)."""
        results = retriever.search("query")
        assert results[0].score >= results[1].score

    def test_category_filter_passed_through(self, retriever, mock_vector_store):
        retriever.search("query", category_filter="legislation")
        call_kwargs = mock_vector_store.search_vl_pages.call_args
        assert call_kwargs.kwargs["category_filter"] == "legislation"

    def test_category_filter_none_by_default(self, retriever, mock_vector_store):
        retriever.search("query")
        call_kwargs = mock_vector_store.search_vl_pages.call_args
        assert call_kwargs.kwargs["category_filter"] is None

    def test_query_embedding_passed_to_vector_store(self, retriever, mock_jina, mock_vector_store):
        fake_embedding = np.ones(2048, dtype=np.float32)
        mock_jina.embed_query.return_value = fake_embedding

        retriever.search("query")

        call_kwargs = mock_vector_store.search_vl_pages.call_args
        assert np.array_equal(call_kwargs.kwargs["query_embedding"], fake_embedding)


# ---------------------------------------------------------------------------
# Adaptive-k integration tests
# ---------------------------------------------------------------------------


def _bimodal_vector_store():
    """Mock vector store returning 6 results: 3 high + 3 low scores."""
    store = MagicMock()
    store.search_vl_pages.return_value = [
        {"page_id": "DOC_p001", "document_id": "DOC", "page_number": 1, "score": 0.90},
        {"page_id": "DOC_p002", "document_id": "DOC", "page_number": 2, "score": 0.85},
        {"page_id": "DOC_p003", "document_id": "DOC", "page_number": 3, "score": 0.80},
        {"page_id": "DOC_p004", "document_id": "DOC", "page_number": 4, "score": 0.25},
        {"page_id": "DOC_p005", "document_id": "DOC", "page_number": 5, "score": 0.20},
        {"page_id": "DOC_p006", "document_id": "DOC", "page_number": 6, "score": 0.15},
    ]
    return store


class TestAdaptiveKIntegration:
    """Test adaptive-k filtering integrated into VLRetriever."""

    def test_adaptive_reduces_bimodal_results(self, mock_jina, mock_page_store):
        """With bimodal scores, adaptive-k should keep only the high group."""
        vs = _bimodal_vector_store()
        adaptive_config = AdaptiveKConfig(
            enabled=True, method="otsu", fetch_k=20, min_k=1, max_k=10,
        )
        retriever = VLRetriever(
            jina_client=mock_jina,
            vector_store=vs,
            page_store=mock_page_store,
            default_k=5,
            adaptive_config=adaptive_config,
        )

        results = retriever.search("test query")

        # Should keep only the 3 high-scoring results
        assert len(results) == 3
        assert all(r.score >= 0.7 for r in results)

    def test_adaptive_fetches_larger_pool(self, mock_jina, mock_page_store):
        """When adaptive is enabled, DB query should use fetch_k, not k."""
        vs = _bimodal_vector_store()
        adaptive_config = AdaptiveKConfig(enabled=True, fetch_k=20, min_k=1, max_k=10)
        retriever = VLRetriever(
            jina_client=mock_jina,
            vector_store=vs,
            page_store=mock_page_store,
            default_k=5,
            adaptive_config=adaptive_config,
        )

        retriever.search("query", k=5)

        # Should request fetch_k (20) from DB, not k (5)
        call_kwargs = vs.search_vl_pages.call_args
        assert call_kwargs.kwargs["k"] == 20

    def test_disabled_adaptive_returns_fixed_k(self, mock_jina, mock_page_store):
        """Disabled adaptive config should not filter results."""
        vs = _bimodal_vector_store()
        adaptive_config = AdaptiveKConfig(enabled=False)
        retriever = VLRetriever(
            jina_client=mock_jina,
            vector_store=vs,
            page_store=mock_page_store,
            default_k=5,
            adaptive_config=adaptive_config,
        )

        results = retriever.search("query", k=5)

        # Should fetch exactly k=5 from DB, no filtering
        call_kwargs = vs.search_vl_pages.call_args
        assert call_kwargs.kwargs["k"] == 5
        # Returns whatever the DB returns (6 in mock)
        assert len(results) == 6

    def test_no_adaptive_config_is_disabled(self, mock_jina, mock_page_store):
        """VLRetriever without adaptive_config should behave as before."""
        vs = _bimodal_vector_store()
        retriever = VLRetriever(
            jina_client=mock_jina,
            vector_store=vs,
            page_store=mock_page_store,
            default_k=5,
        )

        results = retriever.search("query")

        # No adaptive config → disabled → fetch exactly default_k
        call_kwargs = vs.search_vl_pages.call_args
        assert call_kwargs.kwargs["k"] == 5
        assert len(results) == 6  # all DB results returned


# ---------------------------------------------------------------------------
# search_with_embedding() tests
# ---------------------------------------------------------------------------


class TestSearchWithEmbedding:
    """Test that search_with_embedding() returns results AND query embedding."""

    def test_returns_tuple(self, retriever, mock_jina):
        """search_with_embedding returns (results, embedding) tuple."""
        fake_embedding = np.ones(2048, dtype=np.float32) * 0.5
        mock_jina.embed_query.return_value = fake_embedding

        results, embedding = retriever.search_with_embedding("test query")

        assert len(results) == 2
        assert all(isinstance(r, VLPageResult) for r in results)
        assert np.array_equal(embedding, fake_embedding)

    def test_search_delegates_to_search_with_embedding(self, retriever, mock_jina):
        """search() should return same results as search_with_embedding()."""
        results_plain = retriever.search("test query")
        mock_jina.embed_query.return_value = np.zeros(2048, dtype=np.float32)
        results_with_emb, _ = retriever.search_with_embedding("test query")

        assert len(results_plain) == len(results_with_emb)
        for r1, r2 in zip(results_plain, results_with_emb):
            assert r1.page_id == r2.page_id
            assert r1.score == r2.score

    def test_embedding_shape(self, retriever, mock_jina):
        """Returned embedding should match Jina output shape."""
        emb = np.random.default_rng(42).random(2048).astype(np.float32)
        mock_jina.embed_query.return_value = emb

        _, returned_emb = retriever.search_with_embedding("query")

        assert returned_emb.shape == (2048,)
        assert returned_emb.dtype == np.float32
