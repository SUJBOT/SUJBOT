"""
Tests for QPP-based retrieval confidence scoring integration.

Verifies that:
- score_retrieval_general() works with synthetic data
- Confidence bands map correctly to score thresholds
- SearchTool includes confidence in text search results
- Image searches skip QPP scoring (no text query for QPP)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from rag_confidence import score_retrieval_general
from rag_confidence.scorer import _compute_band


class TestScoreRetrievalGeneral:
    """Test the rag_confidence.score_retrieval_general() function."""

    def test_basic_scoring(self):
        """Verify scoring works with synthetic similarity distribution."""
        # Simulate a good retrieval: one strong match, rest are noise
        similarities = np.concatenate([
            np.array([0.85, 0.72, 0.65, 0.60, 0.55]),
            0.2 + 0.1 * np.random.default_rng(42).random(500),
        ])

        result = score_retrieval_general("What are the safety requirements?", similarities)

        assert "confidence" in result
        assert "band" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["band"] in ("HIGH", "MEDIUM", "LOW", "VERY_LOW")

    def test_returns_features_when_requested(self):
        """Verify return_features=True includes feature dict."""
        similarities = np.random.default_rng(42).random(500).astype(np.float32)

        result = score_retrieval_general(
            "test query", similarities, return_features=True
        )

        assert "features" in result
        assert isinstance(result["features"], dict)
        assert "top1_minus_p99" in result["features"]

    def test_flat_distribution_low_confidence(self):
        """A flat similarity distribution should yield lower confidence."""
        # All scores are nearly identical — no clear winner
        flat_sims = 0.50 + 0.01 * np.random.default_rng(42).random(500)

        result = score_retrieval_general("query", flat_sims)

        # Flat distribution means poor retrieval — should not be HIGH
        assert result["band"] in ("MEDIUM", "LOW", "VERY_LOW")

    def test_single_strong_match_higher_confidence(self):
        """One strong outlier should yield higher confidence than flat dist."""
        strong = np.concatenate([
            np.array([0.95]),
            0.3 + 0.05 * np.random.default_rng(42).random(499),
        ])
        flat = 0.50 + 0.01 * np.random.default_rng(42).random(500)

        result_strong = score_retrieval_general("query", strong)
        result_flat = score_retrieval_general("query", flat)

        assert result_strong["confidence"] > result_flat["confidence"]

    def test_small_similarity_array(self):
        """QPP should handle very small arrays (1-3 elements) gracefully."""
        small_arrays = [
            np.array([0.85], dtype=np.float32),
            np.array([0.85, 0.40], dtype=np.float32),
            np.array([0.85, 0.60, 0.35], dtype=np.float32),
        ]
        for sims in small_arrays:
            result = score_retrieval_general("test", sims)
            assert 0.0 <= result["confidence"] <= 1.0
            assert result["band"] in ("HIGH", "MEDIUM", "LOW", "VERY_LOW")


class TestConfidenceBands:
    """Test confidence band threshold mapping."""

    def test_high_band(self):
        assert _compute_band(0.95) == "HIGH"
        assert _compute_band(0.90) == "HIGH"

    def test_medium_band(self):
        assert _compute_band(0.89) == "MEDIUM"
        assert _compute_band(0.75) == "MEDIUM"

    def test_low_band(self):
        assert _compute_band(0.74) == "LOW"
        assert _compute_band(0.50) == "LOW"

    def test_very_low_band(self):
        assert _compute_band(0.49) == "VERY_LOW"
        assert _compute_band(0.0) == "VERY_LOW"

    def test_boundary_values(self):
        """Verify exact boundary behavior."""
        assert _compute_band(0.90) == "HIGH"
        assert _compute_band(0.8999) == "MEDIUM"
        assert _compute_band(0.75) == "MEDIUM"
        assert _compute_band(0.7499) == "LOW"
        assert _compute_band(0.50) == "LOW"
        assert _compute_band(0.4999) == "VERY_LOW"


class TestSearchToolConfidence:
    """Test QPP confidence integration in SearchTool."""

    def _make_search_tool(self):
        """Create a SearchTool with mocked dependencies."""
        from src.agent.tools.search import SearchTool

        mock_vector_store = MagicMock()
        mock_vl_retriever = MagicMock()
        mock_page_store = MagicMock()

        tool = SearchTool(
            vector_store=mock_vector_store,
            vl_retriever=mock_vl_retriever,
            page_store=mock_page_store,
        )

        return tool, mock_vector_store, mock_vl_retriever, mock_page_store

    def test_text_search_includes_confidence(self):
        """Text search should include retrieval_confidence in metadata."""
        from src.vl.vl_retriever import VLPageResult

        tool, mock_vs, mock_vl, mock_ps = self._make_search_tool()

        # Mock search_with_embedding to return results + embedding
        query_embedding = np.random.default_rng(42).random(2048).astype(np.float32)
        mock_vl.search_with_embedding.return_value = (
            [
                VLPageResult(
                    page_id="DOC1_p001",
                    document_id="DOC1",
                    page_number=1,
                    score=0.85,
                ),
            ],
            query_embedding,
        )

        # Mock get_all_vl_similarities
        all_sims = np.concatenate([
            np.array([0.85]),
            0.3 + 0.1 * np.random.default_rng(42).random(499),
        ]).astype(np.float32)
        mock_vs.get_all_vl_similarities.return_value = all_sims

        # Mock page image loading
        mock_ps.get_image_base64.return_value = "base64data"

        result = tool.execute_impl(query="safety requirements", k=5)

        assert result.success
        assert "retrieval_confidence" in result.metadata
        conf = result.metadata["retrieval_confidence"]
        assert "score" in conf
        assert "band" in conf
        assert 0.0 <= conf["score"] <= 1.0
        assert conf["band"] in ("HIGH", "MEDIUM", "LOW", "VERY_LOW")

        # Check confidence annotation in citations
        confidence_citations = [c for c in result.citations if "Retrieval confidence" in c]
        assert len(confidence_citations) == 1

    def test_image_search_no_confidence(self):
        """Image search should NOT include retrieval confidence."""
        from src.vl.vl_retriever import VLPageResult

        tool, mock_vs, mock_vl, mock_ps = self._make_search_tool()

        # Set up attachment images in request context
        tool._request_context = {
            "attachment_images": [
                {"base64_data": "fake_image_data", "filename": "test.png"}
            ]
        }

        mock_vl.search_by_image.return_value = [
            VLPageResult(
                page_id="DOC1_p001",
                document_id="DOC1",
                page_number=1,
                score=0.80,
            ),
        ]
        mock_ps.get_image_base64.return_value = "base64data"

        result = tool.execute_impl(
            query="similar pages",
            image_attachment_index=0,
            k=5,
        )

        assert result.success
        assert "retrieval_confidence" not in result.metadata
        # get_all_vl_similarities should NOT be called for image search
        mock_vs.get_all_vl_similarities.assert_not_called()

    def test_confidence_failure_doesnt_break_search(self):
        """If QPP scoring fails, search should still succeed without confidence."""
        from src.vl.vl_retriever import VLPageResult

        tool, mock_vs, mock_vl, mock_ps = self._make_search_tool()

        query_embedding = np.random.default_rng(42).random(2048).astype(np.float32)
        mock_vl.search_with_embedding.return_value = (
            [
                VLPageResult(
                    page_id="DOC1_p001",
                    document_id="DOC1",
                    page_number=1,
                    score=0.85,
                ),
            ],
            query_embedding,
        )

        # Simulate QPP failure
        mock_vs.get_all_vl_similarities.side_effect = RuntimeError("DB connection lost")
        mock_ps.get_image_base64.return_value = "base64data"

        result = tool.execute_impl(query="safety requirements", k=5)

        # Search should still succeed
        assert result.success
        # But confidence should be absent
        assert "retrieval_confidence" not in result.metadata
