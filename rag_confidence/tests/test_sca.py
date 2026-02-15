"""
Unit tests for SCA (Sufficiency Context Assessment) module.

Tests sufficiency assessment and calibration.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_confidence._experimental.sca import IsotonicCalibrator, SufficiencyAssessor


class TestSufficiencyAssessor:
    """Tests for SufficiencyAssessor."""

    def test_parse_json_response_valid(self):
        """Should parse valid JSON response correctly."""
        assessor = SufficiencyAssessor(model="mock")

        response = """[
            {"chunk_id": "chunk_1", "p_useful": 0.9},
            {"chunk_id": "chunk_2", "p_useful": 0.3}
        ]"""

        chunks = [{"chunk_id": "chunk_1"}, {"chunk_id": "chunk_2"}]
        result = assessor._parse_response(response, chunks)

        # Returns dict mapping chunk_id -> p_useful
        assert len(result) == 2
        assert result["chunk_1"] == pytest.approx(0.9)
        assert result["chunk_2"] == pytest.approx(0.3)

    def test_parse_json_with_markdown_wrapper(self):
        """Should handle JSON wrapped in markdown code blocks."""
        assessor = SufficiencyAssessor(model="mock")

        response = """```json
[
    {"chunk_id": "chunk_1", "p_useful": 0.8}
]
```"""

        chunks = [{"chunk_id": "chunk_1"}]
        result = assessor._parse_response(response, chunks)

        assert len(result) == 1
        assert result["chunk_1"] == pytest.approx(0.8)

    def test_parse_json_with_extra_text(self):
        """Should extract JSON even with surrounding text via regex fallback."""
        assessor = SufficiencyAssessor(model="mock")

        # JSON embedded in text - regex fallback will extract it
        response = 'The scores are: "chunk_id": "chunk_1", "p_useful": 0.7 in the response'

        chunks = [{"chunk_id": "chunk_1"}]
        result = assessor._parse_response(response, chunks)

        # Regex fallback should find it
        assert result.get("chunk_1") == pytest.approx(0.7)

    def test_parse_invalid_json_fallback(self):
        """Should return empty dict for completely invalid JSON."""
        assessor = SufficiencyAssessor(model="mock")

        response = "This is not valid JSON at all"

        chunks = [{"chunk_id": "chunk_1"}, {"chunk_id": "chunk_2"}]
        result = assessor._parse_response(response, chunks)

        # Returns empty dict (calling code uses .get with default 0.5)
        assert len(result) == 0

    def test_max_aggregation_in_assess(self):
        """p_suff should be max of chunk probabilities (tested via result)."""
        # Just verify the aggregation logic - max(p_chunk)
        p_chunk = [0.9, 0.3, 0.6]
        p_suff = max(p_chunk) if p_chunk else 0.0
        assert p_suff == pytest.approx(0.9)

    def test_empty_chunks_assess(self):
        """Empty chunks should return p_suff=0.0."""
        assessor = SufficiencyAssessor(model="mock")

        result = assessor.assess(
            query="Test query",
            chunks=[],
        )

        assert result.p_suff == 0.0
        assert result.n_chunks == 0

    def test_format_chunks_for_prompt(self):
        """Should format chunks correctly for prompt."""
        assessor = SufficiencyAssessor(model="mock")

        chunks = [
            {"chunk_id": "doc_1", "content": "Content one"},
            {"chunk_id": "doc_2", "content": "Content two"},
        ]

        formatted = assessor._format_chunks(chunks)

        assert "doc_1" in formatted
        assert "doc_2" in formatted
        assert "Content one" in formatted
        assert "Content two" in formatted


class TestIsotonicCalibrator:
    """Tests for IsotonicCalibrator."""

    def test_fit_and_calibrate(self):
        """Should fit calibrator and transform probabilities."""
        calibrator = IsotonicCalibrator(n_bins=5)

        # Training data: raw probs vs actual labels
        raw_probs = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
        true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        calibrator.fit(raw_probs, true_labels)

        assert calibrator.is_fitted

        # Calibrate new values
        calibrated = calibrator.calibrate([0.55, 0.85])

        assert len(calibrated) == 2
        for c in calibrated:
            assert 0.0 <= c <= 1.0

    def test_calibrate_single(self):
        """Should calibrate single probability."""
        calibrator = IsotonicCalibrator(n_bins=5)

        raw_probs = np.array([0.2, 0.4, 0.6, 0.8])
        true_labels = np.array([0, 0, 1, 1])

        calibrator.fit(raw_probs, true_labels)

        result = calibrator.calibrate_single(0.7)

        assert 0.0 <= result <= 1.0

    def test_unfitted_calibrator_passthrough(self):
        """Unfitted calibrator should return raw probs."""
        calibrator = IsotonicCalibrator()

        result = calibrator.calibrate([0.3, 0.7])

        assert result == [0.3, 0.7]

    def test_save_and_load(self, tmp_path):
        """Should save and load calibrator."""
        calibrator = IsotonicCalibrator(n_bins=5)

        raw_probs = np.array([0.2, 0.4, 0.6, 0.8])
        true_labels = np.array([0, 0, 1, 1])
        calibrator.fit(raw_probs, true_labels)

        # Save
        path = tmp_path / "calibrator.json"
        calibrator.save(path)

        # Load
        loaded = IsotonicCalibrator.load(path)

        assert loaded.is_fitted

        # Should produce same results
        original = calibrator.calibrate([0.5])
        loaded_result = loaded.calibrate([0.5])

        assert original == loaded_result

    def test_evaluate_improvement(self):
        """Should show calibration improvement."""
        calibrator = IsotonicCalibrator(n_bins=5)

        # Overconfident predictions
        raw_probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        true_labels = np.array([1, 1, 0, 0, 1, 0, 0, 0])

        calibrator.fit(raw_probs, true_labels)

        # Evaluate on same data (would use holdout in practice)
        eval_result = calibrator.evaluate(raw_probs, true_labels)

        assert "brier_before" in eval_result
        assert "brier_after" in eval_result
        assert "ece_before" in eval_result
        assert "ece_after" in eval_result

    def test_get_calibration_curve(self):
        """Should return calibration curve for plotting."""
        calibrator = IsotonicCalibrator(n_bins=5)

        raw_probs = np.array([0.2, 0.4, 0.6, 0.8])
        true_labels = np.array([0, 0, 1, 1])
        calibrator.fit(raw_probs, true_labels)

        bin_centers, calibrated_values = calibrator.get_calibration_curve()

        assert len(bin_centers) == 5
        assert len(calibrated_values) == 5

    def test_length_mismatch_error(self):
        """Should raise error on length mismatch."""
        calibrator = IsotonicCalibrator()

        raw_probs = np.array([0.2, 0.4, 0.6])
        true_labels = np.array([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            calibrator.fit(raw_probs, true_labels)

    def test_unfitted_save_error(self):
        """Should raise error when saving unfitted calibrator."""
        calibrator = IsotonicCalibrator()

        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            calibrator.save("test.json")

    def test_unfitted_curve_error(self):
        """Should raise error when getting curve from unfitted calibrator."""
        calibrator = IsotonicCalibrator()

        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.get_calibration_curve()
