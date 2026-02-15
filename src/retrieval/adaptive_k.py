"""
Adaptive Retrieval-K via Otsu/GMM Score Thresholding.

Analyzes the similarity score distribution to find the natural cutoff
between relevant and irrelevant results, reducing wasted context budget.

Pure score analysis — no coupling to PostgreSQL, Jina, or embeddings.
Shared by both VL search and graph search.
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Literal, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveKConfig:
    """Configuration for adaptive-k score thresholding."""

    enabled: bool = True
    method: Literal["otsu", "gmm"] = "otsu"
    fetch_k: int = 20
    min_k: int = 1
    max_k: int = 10
    score_gap_threshold: float = 0.05
    min_samples_for_adaptive: int = 3


@dataclass(frozen=True)
class AdaptiveKResult:
    """Result of adaptive-k filtering."""

    items: list
    threshold: float
    method_used: str  # "otsu" | "gmm" | "passthrough" | "unimodal_fallback"
    original_count: int
    filtered_count: int
    score_range: Tuple[float, float]


def adaptive_k_filter(
    items: Sequence[Any],
    scores: Sequence[float],
    config: AdaptiveKConfig,
) -> AdaptiveKResult:
    """
    Filter items by adaptive score thresholding.

    Analyzes score distribution to find a natural cutoff between
    relevant and irrelevant results. Falls back gracefully on edge cases.

    Args:
        items: Ordered sequence of result items (highest score first).
        scores: Corresponding similarity scores (same order as items).
        config: Adaptive-k configuration.

    Returns:
        AdaptiveKResult with filtered items and diagnostics.

    Raises:
        ValueError: If items and scores have different lengths.
    """
    n = len(items)

    if len(scores) != n:
        raise ValueError(
            f"items and scores must have same length, got {n} and {len(scores)}"
        )

    # Empty input
    if n == 0:
        return AdaptiveKResult(
            items=[],
            threshold=0.0,
            method_used="passthrough",
            original_count=0,
            filtered_count=0,
            score_range=(0.0, 0.0),
        )

    score_arr = np.array(scores, dtype=np.float64)
    score_min, score_max = float(score_arr.min()), float(score_arr.max())
    score_range = (round(score_min, 4), round(score_max, 4))

    # Disabled → passthrough (respect max_k)
    if not config.enabled:
        capped = list(items[: config.max_k])
        return AdaptiveKResult(
            items=capped,
            threshold=0.0,
            method_used="passthrough",
            original_count=n,
            filtered_count=len(capped),
            score_range=score_range,
        )

    # Too few samples → passthrough (respect max_k)
    if n < config.min_samples_for_adaptive:
        capped = list(items[: config.max_k])
        return AdaptiveKResult(
            items=capped,
            threshold=0.0,
            method_used="passthrough",
            original_count=n,
            filtered_count=len(capped),
            score_range=score_range,
        )

    # Unimodal check — score range too narrow for meaningful split
    if (score_max - score_min) < config.score_gap_threshold:
        k = min(config.min_k, n)
        return AdaptiveKResult(
            items=list(items[:k]),
            threshold=score_min,
            method_used="unimodal_fallback",
            original_count=n,
            filtered_count=k,
            score_range=score_range,
        )

    # Apply thresholding method
    try:
        if config.method == "gmm":
            threshold = _gmm_threshold(score_arr)
        else:
            threshold = _otsu_threshold(score_arr)
    except Exception as e:
        logger.warning(
            "Adaptive-k %s failed, falling back to min_k=%d: %s",
            config.method,
            config.min_k,
            e,
        )
        k = min(config.min_k, n)
        return AdaptiveKResult(
            items=list(items[:k]),
            threshold=0.0,
            method_used="passthrough",
            original_count=n,
            filtered_count=k,
            score_range=score_range,
        )

    method_used = config.method

    # Filter: keep items with score >= threshold
    filtered = [item for item, s in zip(items, scores) if s >= threshold]

    # Enforce bounds
    if len(filtered) < config.min_k:
        # Take at least min_k (or all if fewer available)
        filtered = list(items[: min(config.min_k, n)])
    if len(filtered) > config.max_k:
        filtered = filtered[: config.max_k]

    return AdaptiveKResult(
        items=filtered,
        threshold=round(threshold, 4),
        method_used=method_used,
        original_count=n,
        filtered_count=len(filtered),
        score_range=score_range,
    )


def _otsu_threshold(scores: np.ndarray) -> float:
    """
    Otsu's method for 1-D score thresholding.

    For each possible split point in the sorted scores, compute
    inter-class variance: w0 * w1 * (mu0 - mu1)^2, and pick the
    split that maximizes it.

    Args:
        scores: 1-D array of similarity scores.

    Returns:
        Threshold value that maximizes inter-class variance.
    """
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)

    best_threshold = sorted_scores[0]
    best_variance = -1.0

    for i in range(1, n):
        # Split: group0 = sorted_scores[:i], group1 = sorted_scores[i:]
        w0 = i / n
        w1 = (n - i) / n
        mu0 = sorted_scores[:i].mean()
        mu1 = sorted_scores[i:].mean()
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            # Threshold is the midpoint between the last element of group0
            # and the first element of group1
            best_threshold = (sorted_scores[i - 1] + sorted_scores[i]) / 2.0

    return float(best_threshold)


def _gmm_threshold(scores: np.ndarray) -> float:
    """
    GMM-based threshold using 2-component Gaussian mixture.

    Fits a 2-component GMM and computes threshold at the weighted
    midpoint of the two component means.

    Args:
        scores: 1-D array of similarity scores.

    Returns:
        Threshold at the weighted midpoint of the two Gaussian means.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(scores.reshape(-1, 1))

    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()

    # Threshold at the weighted midpoint of the two means
    total_weight = weights.sum()
    threshold = (weights[0] * means[0] + weights[1] * means[1]) / total_weight

    # If means are on the same side of the threshold, use simple midpoint
    low_mean, high_mean = sorted(means)
    if threshold < low_mean or threshold > high_mean:
        threshold = (low_mean + high_mean) / 2.0

    return float(threshold)
