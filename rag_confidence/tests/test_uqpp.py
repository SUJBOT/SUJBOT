"""
Unit tests for UQPP (Unsupervised Query Performance Prediction) module.

Tests coherence, stability, and combination functions.
"""

import numpy as np
import pytest

from rag_confidence._experimental.uqpp import UQPPExtractor, UQPPNormStats


class TestUQPPCoherence:
    """Tests for coherence computation."""

    def test_identical_embeddings_high_coherence(self):
        """Identical embeddings should produce high coherence (low dispersion)."""
        extractor = UQPPExtractor()

        # All identical embeddings
        embeddings = np.ones((10, 128))

        coherence = extractor.coherence(embeddings)

        # Identical embeddings = zero dispersion = max coherence
        assert coherence == 1.0

    def test_diverse_embeddings_lower_coherence(self):
        """Diverse embeddings should produce lower coherence."""
        extractor = UQPPExtractor()

        # Diverse embeddings (random)
        np.random.seed(42)
        embeddings = np.random.randn(10, 128)

        coherence = extractor.coherence(embeddings)

        # Should be less than 1.0 due to dispersion
        assert 0.0 <= coherence <= 1.0
        assert coherence < 1.0

    def test_coherence_increases_with_clustering(self):
        """Coherence should increase when embeddings are more clustered."""
        extractor = UQPPExtractor()
        np.random.seed(42)

        # Tightly clustered embeddings
        tight = np.random.randn(10, 128) * 0.1 + np.ones(128)

        # Widely spread embeddings
        spread = np.random.randn(10, 128) * 2.0

        coh_tight = extractor.coherence(tight)
        coh_spread = extractor.coherence(spread)

        assert coh_tight > coh_spread

    def test_coherence_single_embedding(self):
        """Single embedding should return 1.0 (no dispersion)."""
        extractor = UQPPExtractor()

        embeddings = np.array([[1.0, 2.0, 3.0]])

        coherence = extractor.coherence(embeddings)

        assert coherence == 1.0

    def test_coherence_with_normalization_stats(self):
        """Coherence should use normalization when stats provided."""
        stats = UQPPNormStats(
            coh_p05=0.5,
            coh_p95=0.9,
        )
        extractor = UQPPExtractor(norm_stats=stats)

        np.random.seed(42)
        embeddings = np.random.randn(10, 128)

        coherence = extractor.coherence(embeddings)

        # Should be normalized between 0 and 1
        assert 0.0 <= coherence <= 1.0


class TestUQPPStability:
    """Tests for stability computation."""

    def test_deterministic_retriever_perfect_stability(self):
        """Deterministic retriever should produce stability = 1.0."""
        extractor = UQPPExtractor()

        query_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        base_ids = ["doc_1", "doc_2", "doc_3"]

        # Deterministic retriever always returns same results
        def deterministic_retriever(emb, k):
            return base_ids[:k]

        stability = extractor.stability(
            query_embedding,
            deterministic_retriever,
            base_ids,
            M=3,
            k=3,
            sigma=0.05,
        )

        assert stability == 1.0

    def test_random_retriever_low_stability(self):
        """Completely random retriever should produce low stability."""
        extractor = UQPPExtractor()

        query_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        base_ids = ["doc_1", "doc_2", "doc_3"]
        all_ids = [f"doc_{i}" for i in range(100)]

        # Random retriever returns different results each time
        def random_retriever(emb, k):
            np.random.seed(None)  # Ensure randomness
            return list(np.random.choice(all_ids, k, replace=False))

        stability = extractor.stability(
            query_embedding,
            random_retriever,
            base_ids,
            M=5,
            k=3,
            sigma=0.1,
        )

        # Should be low due to randomness
        assert 0.0 <= stability <= 1.0
        assert stability < 0.5  # Expect low stability

    def test_stability_increases_with_more_overlap(self):
        """Stability should increase when perturbed results overlap more."""
        extractor = UQPPExtractor()

        query_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        base_ids = ["doc_1", "doc_2", "doc_3"]

        # High overlap retriever (returns base_ids + 1 random)
        call_count = [0]

        def high_overlap_retriever(emb, k):
            call_count[0] += 1
            result = base_ids[:k-1] + [f"random_{call_count[0]}"]
            return result[:k]

        stability = extractor.stability(
            query_embedding,
            high_overlap_retriever,
            base_ids,
            M=3,
            k=3,
            sigma=0.05,
        )

        # Should be reasonably high due to overlap
        assert stability >= 0.5


class TestUQPPCombine:
    """Tests for combining stability and coherence."""

    def test_combine_both_signals(self):
        """Combination should weight both signals."""
        extractor = UQPPExtractor()

        u_stability = 0.8
        u_coherence = 0.6

        # Default weights: w_stab=0.6, w_coh=0.4
        combined = extractor.combine(u_stability, u_coherence)

        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert combined == pytest.approx(expected)

    def test_combine_only_coherence(self):
        """When stability is None, should return coherence."""
        extractor = UQPPExtractor()

        combined = extractor.combine(None, 0.7)

        assert combined == 0.7

    def test_combine_only_stability(self):
        """When coherence is None, should return stability."""
        extractor = UQPPExtractor()

        combined = extractor.combine(0.8, None)

        assert combined == 0.8

    def test_combine_both_none(self):
        """When both are None, should return None."""
        extractor = UQPPExtractor()

        combined = extractor.combine(None, None)

        assert combined is None

    def test_combine_custom_weights(self):
        """Should respect custom weights."""
        extractor = UQPPExtractor()

        combined = extractor.combine(
            u_stability=0.5,
            u_coherence=0.9,
            w_stab=0.3,
            w_coh=0.7,
        )

        expected = 0.3 * 0.5 + 0.7 * 0.9
        assert combined == pytest.approx(expected)


class TestUQPPNormStats:
    """Tests for normalization statistics."""

    def test_load_save_roundtrip(self, tmp_path):
        """Stats should survive save/load cycle."""
        stats = UQPPNormStats(
            coh_p05=0.5,
            coh_p95=0.9,
        )

        path = tmp_path / "stats.json"
        stats.save(path)
        loaded = UQPPNormStats.load(path)

        assert loaded.coh_p05 == stats.coh_p05
        assert loaded.coh_p95 == stats.coh_p95

    def test_extractor_normalization(self):
        """Normalization should scale values to [0, 1]."""
        stats = UQPPNormStats(
            coh_p05=0.4,
            coh_p95=0.8,
        )
        extractor = UQPPExtractor(norm_stats=stats)

        # Value at p05 -> 0.0
        assert extractor._normalize_coherence(0.4) == pytest.approx(0.0)

        # Value at p95 -> 1.0
        assert extractor._normalize_coherence(0.8) == pytest.approx(1.0)

        # Value in middle -> 0.5
        assert extractor._normalize_coherence(0.6) == pytest.approx(0.5)

    def test_normalization_clipping(self):
        """Values outside range should be clipped."""
        stats = UQPPNormStats(
            coh_p05=0.4,
            coh_p95=0.8,
        )
        extractor = UQPPExtractor(norm_stats=stats)

        # Below p05 -> 0.0
        assert extractor._normalize_coherence(0.2) == 0.0

        # Above p95 -> 1.0
        assert extractor._normalize_coherence(0.95) == 1.0


class TestUQPPExtractor:
    """Integration tests for the full extractor."""

    def test_extractor_initialization(self):
        """Extractor should initialize with normalization stats (if available)."""
        # Pass explicit stats to avoid loading from default path
        stats = UQPPNormStats(coh_p05=0.0, coh_p95=1.0)
        extractor = UQPPExtractor(norm_stats=stats)

        assert extractor.norm_stats is not None

    def test_extractor_with_norm_stats_from_path(self, tmp_path):
        """Extractor should use normalization stats when provided via path."""
        stats = UQPPNormStats(
            coh_p05=0.5,
            coh_p95=0.9,
        )
        path = tmp_path / "stats.json"
        stats.save(path)

        extractor = UQPPExtractor(norm_stats_path=path)

        assert extractor.norm_stats is not None
        assert extractor.norm_stats.coh_p05 == 0.5

    def test_full_extraction_workflow(self):
        """Test complete extraction workflow."""
        # Use explicit stats to avoid auto-loading
        stats = UQPPNormStats(coh_p05=0.5, coh_p95=0.9)
        extractor = UQPPExtractor(norm_stats=stats)
        np.random.seed(42)

        # Create test embeddings
        embeddings = np.random.randn(10, 64)

        # Compute coherence
        coherence = extractor.coherence(embeddings)

        assert 0.0 <= coherence <= 1.0

        # Combine (stability not computed)
        combined = extractor.combine(None, coherence)

        assert combined == coherence
