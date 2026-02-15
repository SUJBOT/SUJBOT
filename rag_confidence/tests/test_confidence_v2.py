"""
Integration tests for RAG Confidence v2.

Tests the main score_retrieval_v2 function and backward compatibility.
"""

import numpy as np
import pytest

from rag_confidence.config import V2Config
from rag_confidence.scorer import (
    _compute_band,
    _compute_p_final,
    score_retrieval,
    score_retrieval_v1,
    score_retrieval_v2,
)


class TestBackwardCompatibility:
    """Tests for backward compatibility with v1."""

    @pytest.fixture
    def sample_data(self):
        """Create sample query and similarities."""
        np.random.seed(42)
        query = "Jaké jsou limity pro radiační ochranu?"
        # Create realistic similarity distribution
        similarities = np.concatenate([
            np.array([0.85, 0.72, 0.68, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]),
            np.random.uniform(0.1, 0.3, 90),
        ])
        return query, similarities

    def test_v2_without_config_equals_v1(self, sample_data):
        """score_retrieval_v2 without config should equal v1."""
        query, similarities = sample_data

        v1_result = score_retrieval_v1(query, similarities)
        v2_result = score_retrieval_v2(query, similarities, config=None)

        assert v2_result["confidence"] == pytest.approx(v1_result["confidence"])
        assert v2_result["band"] == v1_result["band"]

        # v2 without config should not have extra keys
        assert "uqpp" not in v2_result
        assert "sca" not in v2_result
        assert "p_final" not in v2_result

    def test_convenience_function_equals_v1(self, sample_data):
        """score_retrieval() should equal v1."""
        query, similarities = sample_data

        v1_result = score_retrieval_v1(query, similarities)
        convenience_result = score_retrieval(query, similarities)

        assert convenience_result == v1_result

    def test_v2_disabled_features_equals_v1(self, sample_data):
        """v2 with all features disabled should equal v1."""
        query, similarities = sample_data

        config = V2Config(enable_uqpp=False, enable_sca=False)

        v1_result = score_retrieval_v1(query, similarities)
        v2_result = score_retrieval_v2(query, similarities, config=config)

        assert v2_result["confidence"] == pytest.approx(v1_result["confidence"])
        assert v2_result["band"] == v1_result["band"]


class TestV2Output:
    """Tests for v2 output structure."""

    @pytest.fixture
    def sample_data_with_embeddings(self):
        """Create sample data including embeddings."""
        np.random.seed(42)
        query = "Jaké jsou limity pro radiační ochranu?"
        similarities = np.concatenate([
            np.array([0.85, 0.72, 0.68, 0.65, 0.60]),
            np.random.uniform(0.1, 0.3, 95),
        ])
        # 10 top-k embeddings of dimension 64
        topk_embeddings = np.random.randn(10, 64)
        topk_ids = [f"chunk_{i}" for i in range(10)]
        return query, similarities, topk_embeddings, topk_ids

    def test_v2_output_structure_with_uqpp(self, sample_data_with_embeddings):
        """v2 should return full output with UQPP enabled.

        NOTE: UQPP is disabled by default after Dense-QPP evaluation showed
        no improvement. This test verifies the output structure when UQPP
        is enabled but not actually computing (no retriever provided).
        """
        query, similarities, topk_embeddings, topk_ids = sample_data_with_embeddings

        # Enable UQPP but without retriever, stability won't be computed
        config = V2Config(
            enable_uqpp=True,
            enable_uqpp_stability=True,  # Enabled but no retriever
            enable_sca=False,
            p_final_w_sup=1.0,  # Must sum to 1.0
            p_final_w_uqpp=0.0,
            p_final_w_sca=0.0,
        )

        result = score_retrieval_v2(
            query,
            similarities,
            topk_doc_embeddings=topk_embeddings,
            topk_doc_ids=topk_ids,
            config=config,
        )

        # Check structure
        assert "confidence" in result
        assert "band" in result
        assert "uqpp" in result
        assert "p_final" in result

        # Check UQPP structure (stability only, no coherence)
        uqpp = result["uqpp"]
        assert "u_stability" in uqpp
        assert "u_score" in uqpp

        # Stability should be None (no retriever callback provided)
        assert uqpp["u_stability"] is None

    def test_v2_p_final_combines_signals(self, sample_data_with_embeddings):
        """p_final should combine p_sup and UQPP when UQPP is computed.

        NOTE: Since UQPP stability requires a retriever callback (not provided here),
        u_score will be None and p_final equals p_sup.
        """
        query, similarities, topk_embeddings, topk_ids = sample_data_with_embeddings

        config = V2Config(
            enable_uqpp=True,
            enable_uqpp_stability=True,
            enable_sca=False,
            p_final_w_sup=1.0,  # Must sum to 1.0
            p_final_w_uqpp=0.0,
            p_final_w_sca=0.0,
        )

        result = score_retrieval_v2(
            query,
            similarities,
            topk_doc_embeddings=topk_embeddings,
            topk_doc_ids=topk_ids,
            config=config,
        )

        p_sup = result["confidence"]
        u_score = result["uqpp"]["u_score"]
        p_final = result["p_final"]

        # Without retriever, u_score is None, so p_final equals p_sup
        assert u_score is None
        assert p_final == pytest.approx(p_sup)

        # Should be bounded
        assert 0.0 <= p_final <= 1.0


class TestComputeBand:
    """Tests for band computation."""

    def test_high_band(self):
        """Confidence >= 0.90 should be HIGH."""
        assert _compute_band(0.95) == "HIGH"
        assert _compute_band(0.90) == "HIGH"

    def test_medium_band(self):
        """0.75 <= confidence < 0.90 should be MEDIUM."""
        assert _compute_band(0.85) == "MEDIUM"
        assert _compute_band(0.75) == "MEDIUM"

    def test_low_band(self):
        """0.50 <= confidence < 0.75 should be LOW."""
        assert _compute_band(0.60) == "LOW"
        assert _compute_band(0.50) == "LOW"

    def test_very_low_band(self):
        """confidence < 0.50 should be VERY_LOW."""
        assert _compute_band(0.40) == "VERY_LOW"
        assert _compute_band(0.10) == "VERY_LOW"


class TestComputePFinal:
    """Tests for p_final computation."""

    def test_p_final_only_p_sup(self):
        """When no UQPP/SCA, p_final should equal p_sup."""
        config = V2Config()

        p_final = _compute_p_final(
            p_sup=0.8,
            u_score=None,
            p_suff=None,
            config=config,
        )

        assert p_final == 0.8

    def test_p_final_with_u_score(self):
        """p_final should combine p_sup and u_score."""
        # Weights must sum to 1.0 (w_sup + w_uqpp + w_sca)
        config = V2Config(
            p_final_w_sup=0.7,
            p_final_w_uqpp=0.2,
            p_final_w_sca=0.1,
        )

        p_final = _compute_p_final(
            p_sup=0.8,
            u_score=0.6,
            p_suff=None,
            config=config,
        )

        # When u_score is available but p_suff is None, the formula uses w_sup and w_uqpp
        # Normalized: (0.7*0.8 + 0.2*0.6) / (0.7+0.2) = (0.56 + 0.12) / 0.9
        expected = (0.7 * 0.8 + 0.2 * 0.6) / (0.7 + 0.2)
        assert p_final == pytest.approx(expected)

    def test_p_final_sca_override(self):
        """SCA p_suff should override if higher."""
        config = V2Config()

        # p_sup is low, but p_suff is high
        p_final = _compute_p_final(
            p_sup=0.5,
            u_score=None,
            p_suff=0.9,
            config=config,
        )

        # p_suff overrides because it's higher
        assert p_final == 0.9

    def test_p_final_sca_no_override(self):
        """SCA p_suff should not override if lower."""
        config = V2Config()

        # p_sup is high, p_suff is lower
        p_final = _compute_p_final(
            p_sup=0.9,
            u_score=None,
            p_suff=0.6,
            config=config,
        )

        # p_sup stays because it's higher
        assert p_final == 0.9

    def test_p_final_clipping(self):
        """p_final should be clipped to [0, 1]."""
        config = V2Config()

        # Even with extreme values, should be clipped
        p_final = _compute_p_final(
            p_sup=1.5,  # Invalid high
            u_score=None,
            p_suff=None,
            config=config,
        )

        assert p_final == 1.0


class TestV2Config:
    """Tests for V2Config."""

    def test_default_config(self):
        """Default config should have sensible values.

        NOTE: UQPP is disabled by default after Dense-QPP evaluation
        showed it degrades AUROC by 0.008-0.055.
        """
        config = V2Config()

        # UQPP disabled by default (Dense-QPP showed no improvement)
        assert config.enable_uqpp is False
        assert config.enable_uqpp_stability is False
        assert config.enable_sca is False  # Off by default (costs money)
        assert config.T_HIGH == 0.90
        assert config.T_MED == 0.75
        assert config.T_LOW == 0.50
        # p_final weights (UQPP disabled, so w_sup=1.0)
        assert config.p_final_w_sup == 1.0
        assert config.p_final_w_uqpp == 0.0

    def test_config_validation(self):
        """Config should validate threshold ordering."""
        # Valid config should work
        config = V2Config(T_HIGH=0.9, T_MED=0.7, T_LOW=0.5)
        assert config.T_HIGH > config.T_MED > config.T_LOW

        # Invalid ordering should raise
        with pytest.raises(ValueError):
            V2Config(T_HIGH=0.5, T_MED=0.9, T_LOW=0.7)

    def test_config_weight_validation(self):
        """Config should validate weights sum to 1.0."""
        # Valid weights (sum to 1.0)
        config = V2Config(p_final_w_sup=0.7, p_final_w_uqpp=0.2, p_final_w_sca=0.1)
        assert config.p_final_w_sup == 0.7

        # Weights not summing to 1.0 should raise
        with pytest.raises(ValueError, match="must sum to 1.0"):
            V2Config(p_final_w_sup=0.5, p_final_w_uqpp=0.2, p_final_w_sca=0.1)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_corpus(self):
        """Should handle small corpus gracefully."""
        # QPP extractor needs at least 30 chunks for bimodal_gap calculation
        # Use minimum viable size for testing
        np.random.seed(42)
        query = "Test query"
        similarities = np.concatenate([
            np.array([0.85, 0.70, 0.60, 0.55, 0.50]),
            np.random.uniform(0.2, 0.4, 45),  # At least 50 total
        ])

        result = score_retrieval_v2(query, similarities)

        assert "confidence" in result
        assert "band" in result

    def test_moderate_corpus(self):
        """Should handle moderate corpus size."""
        np.random.seed(42)
        query = "Test query"
        similarities = np.concatenate([
            np.array([0.8, 0.75, 0.70, 0.65, 0.60]),
            np.random.uniform(0.2, 0.5, 95),  # 100 total
        ])

        result = score_retrieval_v2(query, similarities)

        assert "confidence" in result

    def test_very_high_similarities(self):
        """Should handle very high similarity scores."""
        np.random.seed(42)
        query = "Test query"
        similarities = np.concatenate([
            np.array([0.99, 0.98, 0.97, 0.96, 0.95]),
            np.random.uniform(0.8, 0.9, 95),
        ])

        result = score_retrieval_v2(query, similarities)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_very_low_similarities(self):
        """Should handle very low similarity scores."""
        np.random.seed(42)
        query = "Test query"
        similarities = np.random.uniform(0.1, 0.3, 100)

        result = score_retrieval_v2(query, similarities)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_no_embeddings_with_uqpp_enabled(self):
        """Should handle UQPP enabled but no embeddings/retriever provided."""
        np.random.seed(42)
        query = "Test query"
        similarities = np.random.uniform(0.3, 0.8, 100)

        config = V2Config(
            enable_uqpp=True,
            enable_uqpp_stability=True,
            p_final_w_sup=1.0,  # Must sum to 1.0
            p_final_w_uqpp=0.0,
            p_final_w_sca=0.0,
        )

        # No embeddings or retriever provided
        result = score_retrieval_v2(
            query,
            similarities,
            topk_doc_embeddings=None,
            query_embedding=None,
            retriever=None,
            config=config,
        )

        # Should still work, stability just won't be computed
        assert "uqpp" in result
        assert result["uqpp"]["u_stability"] is None
