"""
Tests for adaptive-k score thresholding (Otsu / GMM).

TDD: these tests are written BEFORE the implementation.
"""

import pytest
from dataclasses import dataclass

from src.retrieval.adaptive_k import (
    AdaptiveKConfig,
    AdaptiveKResult,
    adaptive_k_filter,
    _otsu_threshold,
)


# ---------------------------------------------------------------------------
# Helper to create simple labeled items with scores
# ---------------------------------------------------------------------------


@dataclass
class FakeItem:
    id: str
    score: float


def _make_items(scores: list[float]) -> tuple[list[FakeItem], list[float]]:
    items = [FakeItem(id=f"item_{i}", score=s) for i, s in enumerate(scores)]
    return items, scores


# ---------------------------------------------------------------------------
# Bimodal clear separation
# ---------------------------------------------------------------------------


class TestBimodalSeparation:
    """When scores clearly split into high + low groups, keep only the high group."""

    def test_keeps_only_high_group(self):
        # 3 high (~0.8) + 3 low (~0.3) -> should keep only the 3 high
        scores = [0.85, 0.82, 0.78, 0.30, 0.25, 0.20]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 3
        assert all(item.score >= 0.7 for item in result.items)
        assert result.method_used == "otsu"
        assert result.original_count == 6

    def test_bimodal_with_wider_gap(self):
        scores = [0.95, 0.90, 0.88, 0.10, 0.08, 0.05]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 3
        assert result.items[0].score == 0.95


# ---------------------------------------------------------------------------
# Unimodal (all scores similar) -> fallback
# ---------------------------------------------------------------------------


class TestUnimodalFallback:
    """When all scores are similar (score range < threshold), return min_k."""

    def test_all_high_scores_returns_min_k(self):
        scores = [0.90, 0.89, 0.88, 0.87, 0.86]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=2, max_k=10, score_gap_threshold=0.05)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 2
        assert result.method_used == "unimodal_fallback"

    def test_all_low_scores_returns_min_k(self):
        scores = [0.12, 0.11, 0.10, 0.09]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10, score_gap_threshold=0.05)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 1
        assert result.method_used == "unimodal_fallback"


# ---------------------------------------------------------------------------
# Gradual decay -> Otsu splits roughly in the middle
# ---------------------------------------------------------------------------


class TestGradualDecay:
    """For gradually decaying scores, Otsu should find a reasonable split."""

    def test_linear_decay(self):
        scores = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        # Should keep roughly the top half
        assert 2 <= result.filtered_count <= 6
        assert result.method_used == "otsu"
        # All kept items should have higher scores than all removed items
        kept_scores = [item.score for item in result.items]
        assert min(kept_scores) >= result.threshold


# ---------------------------------------------------------------------------
# min_k / max_k bounds
# ---------------------------------------------------------------------------


class TestBoundsEnforcement:
    """min_k and max_k must always be respected."""

    def test_min_k_forces_minimum_results(self):
        # Even if Otsu says keep 1, min_k=3 should force 3
        scores = [0.90, 0.50, 0.48, 0.10, 0.08]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=3, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count >= 3

    def test_max_k_caps_results(self):
        # Even if all scores are above threshold, max_k=2 should cap at 2
        scores = [0.95, 0.94, 0.93, 0.92, 0.91]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=2)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count <= 2

    def test_min_k_greater_than_available(self):
        # If min_k=5 but only 3 items, return all 3
        scores = [0.80, 0.70, 0.60]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=5, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 3


# ---------------------------------------------------------------------------
# Passthrough for fewer than min_samples_for_adaptive
# ---------------------------------------------------------------------------


class TestPassthrough:
    """With <3 samples, skip analysis and return all items."""

    def test_two_items_passthrough(self):
        scores = [0.90, 0.10]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10, min_samples_for_adaptive=3)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 2
        assert result.method_used == "passthrough"

    def test_one_item_passthrough(self):
        scores = [0.50]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 1
        assert result.method_used == "passthrough"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Empty input should return empty result without error."""

    def test_empty_items_and_scores(self):
        config = AdaptiveKConfig()

        result = adaptive_k_filter([], [], config)

        assert result.filtered_count == 0
        assert result.items == []
        assert result.method_used == "passthrough"
        assert result.original_count == 0

    def test_score_range_for_empty(self):
        result = adaptive_k_filter([], [], AdaptiveKConfig())
        assert result.score_range == (0.0, 0.0)


# ---------------------------------------------------------------------------
# GMM method
# ---------------------------------------------------------------------------


class TestGMMMethod:
    """GMM method should also produce valid threshold splits."""

    def test_gmm_bimodal_separation(self):
        scores = [0.90, 0.88, 0.85, 0.82, 0.80, 0.30, 0.25, 0.20, 0.15, 0.10]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(method="gmm", min_k=1, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        # Should keep the 5 high scores
        assert 3 <= result.filtered_count <= 7
        assert result.method_used == "gmm"

    def test_gmm_fallback_to_unimodal(self):
        scores = [0.50, 0.49, 0.48, 0.47, 0.46]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(method="gmm", min_k=1, max_k=10, score_gap_threshold=0.05)

        result = adaptive_k_filter(items, score_list, config)

        assert result.method_used == "unimodal_fallback"


# ---------------------------------------------------------------------------
# Disabled config
# ---------------------------------------------------------------------------


class TestDisabledConfig:
    """When enabled=False, return all items (up to max_k)."""

    def test_disabled_returns_all_up_to_max_k(self):
        scores = [0.90, 0.10, 0.05]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(enabled=False, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 3
        assert result.method_used == "passthrough"

    def test_disabled_still_respects_max_k(self):
        scores = [0.90, 0.80, 0.70, 0.60, 0.50]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(enabled=False, max_k=3)

        result = adaptive_k_filter(items, score_list, config)

        assert result.filtered_count == 3


# ---------------------------------------------------------------------------
# Exception handling (graceful fallback)
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    """Internal errors should fall back gracefully, not crash."""

    def test_mismatched_lengths_raises(self):
        items = [FakeItem(id="a", score=0.5)]
        scores = [0.5, 0.3]
        config = AdaptiveKConfig()

        with pytest.raises(ValueError, match="length"):
            adaptive_k_filter(items, scores, config)


# ---------------------------------------------------------------------------
# AdaptiveKResult properties
# ---------------------------------------------------------------------------


class TestAdaptiveKResult:
    """Verify AdaptiveKResult is frozen and has expected fields."""

    def test_result_is_immutable(self):
        result = AdaptiveKResult(
            items=[],
            threshold=0.5,
            method_used="otsu",
            original_count=10,
            filtered_count=5,
            score_range=(0.1, 0.9),
        )
        with pytest.raises(AttributeError):
            result.threshold = 0.7

    def test_score_range_tuple(self):
        scores = [0.90, 0.50, 0.10]
        items, score_list = _make_items(scores)
        config = AdaptiveKConfig(min_k=1, max_k=10)

        result = adaptive_k_filter(items, score_list, config)

        assert result.score_range == (0.10, 0.90)


# ---------------------------------------------------------------------------
# Otsu threshold unit test
# ---------------------------------------------------------------------------


class TestOtsuThreshold:
    """Direct test of the Otsu threshold function."""

    def test_bimodal_threshold_between_groups(self):
        import numpy as np

        scores = np.array([0.85, 0.82, 0.78, 0.30, 0.25, 0.20])
        threshold = _otsu_threshold(scores)

        # Threshold should be between the two groups
        assert 0.30 <= threshold <= 0.78

    def test_three_scores(self):
        import numpy as np

        scores = np.array([0.9, 0.5, 0.1])
        threshold = _otsu_threshold(scores)

        # Should produce a valid threshold between min and max
        assert 0.1 <= threshold <= 0.9
