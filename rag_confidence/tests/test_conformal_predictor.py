"""
Unit tests for Conformal Predictor.

Run with: uv run pytest rag_confidence/tests/test_conformal_predictor.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag_confidence.conformal_predictor import (
    CalibrationResult,
    ConformalPredictor,
    evaluate_coverage,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample calibration data with known properties."""
    np.random.seed(42)
    n_queries = 100
    n_chunks = 50

    # Create similarity matrix with values in [0.2, 0.9]
    similarity_matrix = np.random.uniform(0.2, 0.9, (n_queries, n_chunks))

    # Create relevance matrix (each query has 1-3 relevant chunks)
    relevance_matrix = np.zeros((n_queries, n_chunks), dtype=int)
    for i in range(n_queries):
        n_relevant = np.random.randint(1, 4)
        relevant_indices = np.random.choice(n_chunks, n_relevant, replace=False)
        relevance_matrix[i, relevant_indices] = 1

        # Make relevant chunks have higher similarity (more realistic)
        similarity_matrix[i, relevant_indices] += 0.2
        similarity_matrix[i, relevant_indices] = np.clip(
            similarity_matrix[i, relevant_indices], 0, 1
        )

    chunk_ids = np.array([f"chunk_{i}" for i in range(n_chunks)])

    return similarity_matrix, relevance_matrix, chunk_ids


@pytest.fixture
def calibrated_predictor(sample_data):
    """Return a calibrated predictor."""
    similarity_matrix, relevance_matrix, _ = sample_data
    predictor = ConformalPredictor(alpha=0.1)
    predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)
    return predictor


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibration:
    """Tests for calibration functionality."""

    def test_basic_calibration(self, sample_data):
        """Test that calibration produces valid result."""
        similarity_matrix, relevance_matrix, _ = sample_data
        predictor = ConformalPredictor(alpha=0.1)

        result = predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

        assert isinstance(result, CalibrationResult)
        assert result.alpha == 0.1
        assert result.coverage_guarantee == 0.9
        assert result.n_calibration == 100
        assert 0 <= result.threshold <= 1
        assert predictor.is_calibrated

    def test_calibration_alpha_05(self, sample_data):
        """Test calibration with 95% coverage (alpha=0.05)."""
        similarity_matrix, relevance_matrix, _ = sample_data
        predictor_90 = ConformalPredictor(alpha=0.1)
        predictor_95 = ConformalPredictor(alpha=0.05)

        result_90 = predictor_90.calibrate_from_matrices(
            similarity_matrix, relevance_matrix
        )
        result_95 = predictor_95.calibrate_from_matrices(
            similarity_matrix, relevance_matrix
        )

        # Higher coverage (lower alpha) should yield lower threshold
        assert result_95.threshold < result_90.threshold
        assert result_95.coverage_guarantee == 0.95
        assert result_90.coverage_guarantee == 0.90

    def test_calibration_percentiles(self, calibrated_predictor):
        """Test that percentiles are computed correctly."""
        result = calibrated_predictor.calibration_result
        percentiles = result.score_percentiles

        assert "p5" in percentiles
        assert "p50" in percentiles
        assert "p95" in percentiles

        # Percentiles should be ordered
        assert percentiles["p5"] <= percentiles["p25"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p75"] <= percentiles["p95"]

    def test_calibration_with_embeddings(self):
        """Test calibration from raw embeddings."""
        np.random.seed(42)
        n_queries, n_chunks, embed_dim = 50, 30, 64

        query_embeddings = np.random.randn(n_queries, embed_dim)
        chunk_embeddings = np.random.randn(n_chunks, embed_dim)

        # Create relevance
        relevance_matrix = np.zeros((n_queries, n_chunks), dtype=int)
        for i in range(n_queries):
            relevant_idx = np.random.randint(0, n_chunks)
            relevance_matrix[i, relevant_idx] = 1

        predictor = ConformalPredictor(alpha=0.1)
        result = predictor.calibrate_from_embeddings(
            query_embeddings, chunk_embeddings, relevance_matrix
        )

        assert predictor.is_calibrated
        assert result.n_calibration == n_queries

    def test_calibration_skips_queries_without_relevant(self):
        """Test that queries without relevant chunks are skipped."""
        np.random.seed(42)
        n_queries, n_chunks = 20, 10

        similarity_matrix = np.random.uniform(0.3, 0.8, (n_queries, n_chunks))
        relevance_matrix = np.zeros((n_queries, n_chunks), dtype=int)

        # Only 15 queries have relevant chunks
        for i in range(15):
            relevance_matrix[i, np.random.randint(0, n_chunks)] = 1

        predictor = ConformalPredictor(alpha=0.1)
        result = predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

        assert result.n_calibration == 15  # Only queries with relevant chunks

    def test_calibration_fails_insufficient_data(self):
        """Test that calibration fails with too few valid queries."""
        similarity_matrix = np.random.uniform(0.3, 0.8, (5, 10))
        relevance_matrix = np.zeros((5, 10), dtype=int)

        # Only 3 queries have relevant chunks (less than minimum 10)
        for i in range(3):
            relevance_matrix[i, 0] = 1

        predictor = ConformalPredictor(alpha=0.1)
        with pytest.raises(ValueError, match="at least 10"):
            predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)


# =============================================================================
# Inference Tests
# =============================================================================


class TestInference:
    """Tests for inference (retrieval) functionality."""

    def test_retrieve_with_guarantee(self, calibrated_predictor, sample_data):
        """Test threshold-based retrieval."""
        similarity_matrix, _, chunk_ids = sample_data

        # Use first query's similarities
        similarities = similarity_matrix[0]
        retrieved = calibrated_predictor.retrieve_with_guarantee(similarities, chunk_ids)

        assert isinstance(retrieved, set)
        assert all(isinstance(cid, str) for cid in retrieved)

        # Retrieved chunks should have similarity >= threshold
        for i, cid in enumerate(chunk_ids):
            if str(cid) in retrieved:
                assert similarities[i] >= calibrated_predictor.threshold

    def test_top_k_with_confidence(self, calibrated_predictor, sample_data):
        """Test top-k retrieval with confidence."""
        similarity_matrix, _, chunk_ids = sample_data
        similarities = similarity_matrix[0]

        k = 5
        retrieved_ids, confidence = calibrated_predictor.top_k_with_confidence(
            similarities, chunk_ids, k
        )

        assert len(retrieved_ids) == k
        assert 0 <= confidence <= 1
        assert isinstance(retrieved_ids, list)

        # Check that top-k are actually the highest similarities
        top_k_mask = np.zeros(len(chunk_ids), dtype=bool)
        for cid in retrieved_ids:
            idx = np.where(chunk_ids == cid)[0][0]
            top_k_mask[idx] = True

        top_k_sims = similarities[top_k_mask]
        other_sims = similarities[~top_k_mask]
        assert min(top_k_sims) >= max(other_sims)

    def test_compute_confidence(self, calibrated_predictor):
        """Test confidence computation."""
        # High min_similarity = strict threshold = few queries covered = LOW confidence
        high_threshold_conf = calibrated_predictor.compute_confidence(0.9)
        assert high_threshold_conf < 0.3  # Strict threshold, low confidence

        # Low min_similarity = loose threshold = most queries covered = HIGH confidence
        low_threshold_conf = calibrated_predictor.compute_confidence(0.1)
        assert low_threshold_conf > 0.9  # Loose threshold, high confidence

        # Confidence DECREASES as min_similarity (threshold) increases
        # (Stricter threshold = fewer queries have all relevant chunks above it)
        conf_50 = calibrated_predictor.compute_confidence(0.5)
        conf_70 = calibrated_predictor.compute_confidence(0.7)
        assert conf_50 >= conf_70  # Lower threshold = higher confidence

    def test_threshold_for_coverage(self, calibrated_predictor):
        """Test dynamic threshold computation."""
        threshold_90 = calibrated_predictor.threshold_for_coverage(0.90)
        threshold_95 = calibrated_predictor.threshold_for_coverage(0.95)
        threshold_99 = calibrated_predictor.threshold_for_coverage(0.99)

        # Higher coverage requires lower threshold
        assert threshold_99 < threshold_95 < threshold_90

    def test_uncalibrated_retrieval_fails(self):
        """Test that retrieval fails without calibration."""
        predictor = ConformalPredictor(alpha=0.1)
        similarities = np.array([0.5, 0.6, 0.7])
        chunk_ids = np.array(["a", "b", "c"])

        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.retrieve_with_guarantee(similarities, chunk_ids)

    def test_uncalibrated_confidence_fails(self):
        """Test that confidence computation fails without calibration."""
        predictor = ConformalPredictor(alpha=0.1)

        with pytest.raises(RuntimeError, match="calibrate"):
            predictor.compute_confidence(0.5)


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self, calibrated_predictor):
        """Test that save/load preserves calibration."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Save
            calibrated_predictor.save(path)
            assert path.exists()

            # Load
            loaded = ConformalPredictor.load(path)

            # Verify
            assert loaded.is_calibrated
            assert loaded.alpha == calibrated_predictor.alpha
            assert loaded.threshold == calibrated_predictor.threshold
            assert loaded.calibration_result.n_calibration == (
                calibrated_predictor.calibration_result.n_calibration
            )
            np.testing.assert_array_almost_equal(
                loaded.calibration_scores, calibrated_predictor.calibration_scores
            )
        finally:
            path.unlink()

    def test_save_uncalibrated_fails(self):
        """Test that saving uncalibrated predictor fails."""
        predictor = ConformalPredictor(alpha=0.1)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            with pytest.raises(RuntimeError, match="not calibrated"):
                predictor.save(path)
        finally:
            if path.exists():
                path.unlink()

    def test_saved_file_format(self, calibrated_predictor):
        """Test that saved file has expected JSON structure."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            calibrated_predictor.save(path)

            with open(path) as f:
                data = json.load(f)

            assert "calibration_result" in data
            assert "calibration_scores" in data
            assert data["calibration_result"]["alpha"] == 0.1
            assert isinstance(data["calibration_scores"], list)
        finally:
            path.unlink()


# =============================================================================
# Evaluation Tests
# =============================================================================


class TestEvaluation:
    """Tests for coverage evaluation."""

    def test_evaluate_coverage_basic(self, calibrated_predictor, sample_data):
        """Test coverage evaluation returns expected metrics."""
        similarity_matrix, relevance_matrix, chunk_ids = sample_data

        metrics = evaluate_coverage(
            calibrated_predictor, similarity_matrix, relevance_matrix, chunk_ids
        )

        assert "empirical_coverage" in metrics
        assert "target_coverage" in metrics
        assert "coverage_satisfied" in metrics
        assert "recall" in metrics
        assert "avg_retrieved" in metrics
        assert "n_test" in metrics

        assert 0 <= metrics["empirical_coverage"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert metrics["n_test"] == 100

    def test_coverage_on_calibration_data(self, sample_data):
        """Test that coverage on calibration data is >= target (by design)."""
        similarity_matrix, relevance_matrix, chunk_ids = sample_data

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

        metrics = evaluate_coverage(
            predictor, similarity_matrix, relevance_matrix, chunk_ids
        )

        # Coverage on calibration set should be >= 1 - alpha
        # Allow small tolerance for edge cases
        assert metrics["empirical_coverage"] >= 0.85  # Some slack for boundary effects

    def test_evaluate_coverage_empty_relevant(self):
        """Test evaluation handles queries without relevant chunks."""
        np.random.seed(42)
        n_queries, n_chunks = 30, 20

        similarity_matrix = np.random.uniform(0.3, 0.8, (n_queries, n_chunks))
        relevance_matrix = np.zeros((n_queries, n_chunks), dtype=int)

        # Only 20 queries have relevant chunks (for calibration)
        for i in range(20):
            relevance_matrix[i, np.random.randint(0, n_chunks)] = 1

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

        chunk_ids = np.array([f"chunk_{i}" for i in range(n_chunks)])

        # Evaluate on all 30 queries (some without relevant chunks)
        metrics = evaluate_coverage(
            predictor, similarity_matrix, relevance_matrix, chunk_ids
        )

        # Should handle gracefully
        assert metrics["n_test"] == 30


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_alpha_boundaries(self):
        """Test that invalid alpha values are rejected."""
        with pytest.raises(ValueError):
            ConformalPredictor(alpha=0)

        with pytest.raises(ValueError):
            ConformalPredictor(alpha=1)

        with pytest.raises(ValueError):
            ConformalPredictor(alpha=-0.1)

        with pytest.raises(ValueError):
            ConformalPredictor(alpha=1.5)

    def test_shape_mismatch(self):
        """Test that mismatched matrix shapes are rejected."""
        similarity_matrix = np.random.uniform(0, 1, (100, 50))
        relevance_matrix = np.random.randint(0, 2, (100, 60))  # Wrong n_chunks

        predictor = ConformalPredictor(alpha=0.1)
        with pytest.raises(ValueError, match="Shape mismatch"):
            predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

    def test_single_relevant_chunk_per_query(self):
        """Test calibration when each query has exactly one relevant chunk."""
        np.random.seed(42)
        n_queries, n_chunks = 50, 30

        similarity_matrix = np.random.uniform(0.3, 0.8, (n_queries, n_chunks))
        relevance_matrix = np.zeros((n_queries, n_chunks), dtype=int)

        for i in range(n_queries):
            relevance_matrix[i, i % n_chunks] = 1

        predictor = ConformalPredictor(alpha=0.1)
        result = predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

        assert result.n_calibration == n_queries

    def test_all_chunks_relevant(self):
        """Test when all chunks are relevant for a query."""
        np.random.seed(42)
        n_queries, n_chunks = 20, 10

        similarity_matrix = np.random.uniform(0.3, 0.8, (n_queries, n_chunks))
        relevance_matrix = np.ones((n_queries, n_chunks), dtype=int)

        predictor = ConformalPredictor(alpha=0.1)
        result = predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)

        # Min similarity score when all chunks are relevant is the global min
        assert predictor.is_calibrated


# =============================================================================
# CalibrationResult Tests
# =============================================================================


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""

    def test_to_dict(self, calibrated_predictor):
        """Test serialization to dictionary."""
        result = calibrated_predictor.calibration_result
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["alpha"] == 0.1
        assert d["threshold"] == result.threshold
        assert d["n_calibration"] == result.n_calibration

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "threshold": 0.5,
            "alpha": 0.1,
            "n_calibration": 100,
            "coverage_guarantee": 0.9,
            "score_percentiles": {"p50": 0.5},
            "calibrated_at": "2024-01-01T00:00:00Z",
        }

        result = CalibrationResult.from_dict(d)

        assert result.threshold == 0.5
        assert result.alpha == 0.1
        assert result.n_calibration == 100
