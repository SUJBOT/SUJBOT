"""
Conformal Prediction for RAG Retrieval.

Provides statistical coverage guarantees for chunk retrieval.
Guarantees that with probability >= 1-alpha, ALL relevant chunks
are retrieved for a given query.

Usage:
    # Calibration (offline, once)
    predictor = ConformalPredictor(alpha=0.1)
    predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)
    predictor.save("rag_confidence/data/calibration_result.json")

    # Inference (online, per query)
    predictor = ConformalPredictor.load("rag_confidence/data/calibration_result.json")
    chunk_ids = predictor.retrieve_with_guarantee(query_similarities, chunk_ids)
    # OR
    confidence = predictor.compute_confidence(min_similarity)
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of conformal calibration."""

    threshold: float  # Calibrated similarity threshold
    alpha: float  # Error rate (e.g., 0.1 for 90% coverage)
    n_calibration: int  # Number of calibration queries
    coverage_guarantee: float  # 1 - alpha
    score_percentiles: Dict[str, float]  # p5, p10, p25, p50, p75, p90, p95
    calibrated_at: str  # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationResult":
        return cls(**d)


class ConformalPredictor:
    """
    Conformal predictor for RAG retrieval with coverage guarantees.

    Uses Approach 1 (Minimum Similarity) from conformal_rag_retrieval_guide.md:
    - For each calibration query, compute min similarity to relevant chunks
    - Threshold = alpha-quantile of these min similarities
    - At inference: retrieve all chunks with similarity >= threshold

    Attributes:
        alpha: Error rate (default 0.1 = 90% coverage guarantee)
        calibration_result: Result of calibration (None until calibrated)
        calibration_scores: Sorted array of nonconformity scores
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Args:
            alpha: Desired error rate. 0.1 = 90% coverage, 0.05 = 95% coverage.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.calibration_result: Optional[CalibrationResult] = None
        self.calibration_scores: Optional[np.ndarray] = None

    @property
    def threshold(self) -> float:
        """Get calibrated threshold (raises if not calibrated)."""
        if self.calibration_result is None:
            raise RuntimeError("Must calibrate before accessing threshold")
        return self.calibration_result.threshold

    @property
    def is_calibrated(self) -> bool:
        """Check if predictor has been calibrated."""
        return self.calibration_result is not None

    # =========================================================================
    # Calibration Methods
    # =========================================================================

    def calibrate_from_matrices(
        self,
        similarity_matrix: np.ndarray,
        relevance_matrix: np.ndarray,
        query_ids: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """
        Calibrate threshold from precomputed similarity and relevance matrices.

        Args:
            similarity_matrix: (n_queries, n_chunks) cosine similarities
            relevance_matrix: (n_queries, n_chunks) binary relevance (0 or 1)
            query_ids: Optional query identifiers for logging

        Returns:
            CalibrationResult with threshold and statistics
        """
        n_queries, n_chunks = similarity_matrix.shape

        if relevance_matrix.shape != (n_queries, n_chunks):
            raise ValueError(
                f"Shape mismatch: similarity {similarity_matrix.shape} vs "
                f"relevance {relevance_matrix.shape}"
            )

        logger.info(f"Calibrating from matrices: {n_queries} queries, {n_chunks} chunks")

        # Compute nonconformity scores (min similarity to relevant chunks)
        scores = []
        skipped = 0

        for i in range(n_queries):
            relevant_mask = relevance_matrix[i] > 0
            if not relevant_mask.any():
                skipped += 1
                continue

            relevant_similarities = similarity_matrix[i, relevant_mask]
            min_sim = float(np.min(relevant_similarities))
            scores.append(min_sim)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} queries with no relevant chunks")

        scores = np.array(scores)
        return self._finalize_calibration(scores)

    def calibrate_from_embeddings(
        self,
        query_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
        relevance_matrix: np.ndarray,
    ) -> CalibrationResult:
        """
        Calibrate from raw embeddings (computes similarities on-the-fly).

        Args:
            query_embeddings: (n_queries, embed_dim) query vectors
            chunk_embeddings: (n_chunks, embed_dim) chunk vectors
            relevance_matrix: (n_queries, n_chunks) binary relevance

        Returns:
            CalibrationResult with threshold and statistics
        """
        logger.info("Computing similarity matrix from embeddings...")

        # Normalize embeddings
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

        query_embeddings = query_embeddings / np.where(query_norms > 0, query_norms, 1)
        chunk_embeddings = chunk_embeddings / np.where(chunk_norms > 0, chunk_norms, 1)

        # Compute similarity matrix (batch matrix multiplication)
        similarity_matrix = query_embeddings @ chunk_embeddings.T

        return self.calibrate_from_matrices(similarity_matrix, relevance_matrix)

    def _finalize_calibration(self, scores: np.ndarray) -> CalibrationResult:
        """Compute threshold from nonconformity scores."""
        n = len(scores)
        if n < 10:
            raise ValueError(f"Need at least 10 valid calibration queries, got {n}")

        # Compute alpha-quantile (lower tail)
        # For n=1000, alpha=0.1: index = floor(0.1 * 1001) - 1 = 99 (100th smallest)
        quantile_index = int(np.floor(self.alpha * (n + 1))) - 1
        quantile_index = max(0, min(quantile_index, n - 1))

        sorted_scores = np.sort(scores)
        threshold = float(sorted_scores[quantile_index])

        self.calibration_scores = sorted_scores

        # Compute percentiles for diagnostics
        percentiles = {
            "p5": float(np.percentile(scores, 5)),
            "p10": float(np.percentile(scores, 10)),
            "p25": float(np.percentile(scores, 25)),
            "p50": float(np.percentile(scores, 50)),
            "p75": float(np.percentile(scores, 75)),
            "p90": float(np.percentile(scores, 90)),
            "p95": float(np.percentile(scores, 95)),
        }

        self.calibration_result = CalibrationResult(
            threshold=threshold,
            alpha=self.alpha,
            n_calibration=n,
            coverage_guarantee=1 - self.alpha,
            score_percentiles=percentiles,
            calibrated_at=datetime.utcnow().isoformat() + "Z",
        )

        logger.info(
            f"Calibration complete: threshold={threshold:.4f}, "
            f"coverage={100 * (1 - self.alpha):.0f}%, n={n}"
        )
        logger.debug(f"Score distribution: {percentiles}")

        return self.calibration_result

    # =========================================================================
    # Inference Methods
    # =========================================================================

    def retrieve_with_guarantee(
        self,
        similarities: np.ndarray,
        chunk_ids: np.ndarray,
    ) -> Set[str]:
        """
        Retrieve chunks with coverage guarantee.

        Returns all chunks with similarity >= calibrated threshold.
        With probability >= 1-alpha, this set contains ALL relevant chunks.

        Args:
            similarities: (n_chunks,) cosine similarities to query
            chunk_ids: (n_chunks,) chunk identifiers

        Returns:
            Set of chunk_ids to retrieve
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before retrieval")

        mask = similarities >= self.threshold
        retrieved = set(str(cid) for cid in chunk_ids[mask])

        logger.debug(
            f"Conformal retrieval: {len(retrieved)}/{len(chunk_ids)} chunks "
            f"(threshold={self.threshold:.4f})"
        )

        return retrieved

    def compute_confidence(
        self,
        min_similarity: float,
    ) -> float:
        """
        Compute confidence that all relevant chunks are retrieved.

        Given the minimum similarity among retrieved chunks, compute the
        probability that all relevant chunks have been captured.

        Args:
            min_similarity: Minimum similarity in the retrieved set

        Returns:
            Confidence level (0.0 to 1.0)
        """
        if self.calibration_scores is None:
            raise RuntimeError("Must calibrate before computing confidence")

        # Confidence = fraction of calibration queries where min_relevant_sim >= min_similarity
        # A query is "covered" when all its relevant chunks have similarity >= threshold.
        # This happens when the query's min_relevant_sim (calibration score) >= threshold.
        # Higher k → lower min_similarity threshold → more queries covered → higher confidence.
        confidence = float(np.mean(self.calibration_scores >= min_similarity))

        return confidence

    def top_k_with_confidence(
        self,
        similarities: np.ndarray,
        chunk_ids: np.ndarray,
        k: int,
    ) -> Tuple[List[str], float]:
        """
        Retrieve top-k chunks and report confidence level.

        Useful when you have a fixed budget but want to know the confidence
        that all relevant chunks are included.

        Args:
            similarities: (n_chunks,) cosine similarities
            chunk_ids: (n_chunks,) chunk identifiers
            k: Number of chunks to retrieve

        Returns:
            Tuple of (list of top-k chunk_ids, confidence level)
        """
        if self.calibration_scores is None:
            raise RuntimeError("Must calibrate before retrieval")

        # Get top-k by similarity
        k = min(k, len(similarities))
        top_k_indices = np.argpartition(-similarities, k - 1)[:k]
        top_k_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]

        retrieved_ids = [str(chunk_ids[i]) for i in top_k_sorted]

        # Minimum similarity in top-k (effective threshold)
        min_sim = float(similarities[top_k_sorted[-1]])
        confidence = self.compute_confidence(min_sim)

        return retrieved_ids, confidence

    def threshold_for_coverage(self, target_coverage: float) -> float:
        """
        Get threshold needed for a specific coverage level.

        Args:
            target_coverage: Desired coverage (e.g., 0.95 for 95%)

        Returns:
            Similarity threshold for that coverage
        """
        if self.calibration_scores is None:
            raise RuntimeError("Must calibrate before computing threshold")

        alpha = 1 - target_coverage
        n = len(self.calibration_scores)
        quantile_index = int(np.floor(alpha * (n + 1))) - 1
        quantile_index = max(0, min(quantile_index, n - 1))

        return float(self.calibration_scores[quantile_index])

    # =========================================================================
    # Top-K Coverage Analysis
    # =========================================================================

    def coverage_for_k(
        self,
        k: int,
        similarity_matrix: np.ndarray,
        relevance_matrix: np.ndarray,
    ) -> float:
        """
        Compute expected coverage (Recall@k) for a given k.

        Coverage = P(relevant chunk is in top-k results)

        Args:
            k: Number of top results to consider
            similarity_matrix: (n_queries, n_chunks) similarity scores
            relevance_matrix: (n_queries, n_chunks) binary relevance labels

        Returns:
            Coverage probability (fraction of queries with relevant chunk in top-k)
        """
        n_queries = similarity_matrix.shape[0]
        n_covered = 0

        for i in range(n_queries):
            relevant_mask = relevance_matrix[i] == 1
            if not relevant_mask.any():
                continue

            # Get top-k indices
            top_k_indices = np.argpartition(-similarity_matrix[i], min(k, len(similarity_matrix[i])) - 1)[:k]

            # Check if any relevant chunk is in top-k
            if relevance_matrix[i, top_k_indices].sum() > 0:
                n_covered += 1

        return n_covered / n_queries if n_queries > 0 else 0.0

    def k_for_coverage(
        self,
        target_coverage: float,
        similarity_matrix: np.ndarray,
        relevance_matrix: np.ndarray,
        max_k: int = 500,
    ) -> int:
        """
        Find minimum k needed to achieve target coverage.

        Uses binary search to find the smallest k where Recall@k >= target_coverage.

        Args:
            target_coverage: Desired coverage (e.g., 0.9 for 90%)
            similarity_matrix: (n_queries, n_chunks) similarity scores
            relevance_matrix: (n_queries, n_chunks) binary relevance labels
            max_k: Maximum k to consider (default 500)

        Returns:
            Minimum k that achieves target coverage
        """
        # First, compute ranks of relevant chunks for each query
        n_queries = similarity_matrix.shape[0]
        relevant_ranks = []

        for i in range(n_queries):
            relevant_mask = relevance_matrix[i] == 1
            if not relevant_mask.any():
                continue

            # Get rank of the relevant chunk (0-indexed)
            relevant_indices = np.where(relevant_mask)[0]
            similarities = similarity_matrix[i]

            # Rank = number of chunks with higher similarity + 1
            for rel_idx in relevant_indices:
                rank = (similarities > similarities[rel_idx]).sum() + 1
                relevant_ranks.append(rank)

        relevant_ranks = np.array(relevant_ranks)

        # Find minimum k where coverage >= target
        # Coverage@k = fraction of relevant_ranks <= k
        for k in range(1, max_k + 1):
            coverage = (relevant_ranks <= k).mean()
            if coverage >= target_coverage:
                return k

        return max_k  # Target not achievable within max_k

    def analyze_top_k(
        self,
        similarity_matrix: np.ndarray,
        relevance_matrix: np.ndarray,
        k_values: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive top-k coverage analysis.

        Args:
            similarity_matrix: (n_queries, n_chunks) similarity scores
            relevance_matrix: (n_queries, n_chunks) binary relevance labels
            k_values: List of k values to evaluate (default: [1, 3, 5, 10, 20, 50, 100])

        Returns:
            Dict with coverage analysis results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20, 50, 100]

        # Compute coverage for each k
        coverage_by_k = {}
        for k in k_values:
            coverage_by_k[k] = self.coverage_for_k(k, similarity_matrix, relevance_matrix)

        # Find k for common coverage targets
        k_for_targets = {}
        for target in [0.80, 0.85, 0.90, 0.95, 0.99]:
            k_for_targets[f"{int(target*100)}%"] = self.k_for_coverage(
                target, similarity_matrix, relevance_matrix
            )

        return {
            "coverage_by_k": coverage_by_k,
            "k_for_coverage": k_for_targets,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration result to JSON file."""
        if self.calibration_result is None:
            raise RuntimeError("Nothing to save - not calibrated")

        path = Path(path)

        data = {
            "calibration_result": self.calibration_result.to_dict(),
            "calibration_scores": self.calibration_scores.tolist(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved calibration to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalPredictor":
        """Load calibrated predictor from JSON file."""
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        result = CalibrationResult.from_dict(data["calibration_result"])

        predictor = cls(alpha=result.alpha)
        predictor.calibration_result = result
        predictor.calibration_scores = np.array(data["calibration_scores"])

        logger.info(
            f"Loaded calibration from {path}: "
            f"threshold={result.threshold:.4f}, coverage={result.coverage_guarantee:.0%}"
        )

        return predictor


# =============================================================================
# Evaluation Utilities
# =============================================================================


def evaluate_coverage(
    predictor: ConformalPredictor,
    similarity_matrix: np.ndarray,
    relevance_matrix: np.ndarray,
    chunk_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate empirical coverage on a test set.

    Args:
        predictor: Calibrated ConformalPredictor
        similarity_matrix: (n_queries, n_chunks) test similarities
        relevance_matrix: (n_queries, n_chunks) test relevance
        chunk_ids: (n_chunks,) chunk identifiers

    Returns:
        Dict with coverage metrics
    """
    n_queries = similarity_matrix.shape[0]
    n_covered = 0
    total_relevant = 0
    total_retrieved_relevant = 0
    total_retrieved = 0

    for i in range(n_queries):
        retrieved = predictor.retrieve_with_guarantee(similarity_matrix[i], chunk_ids)
        relevant = set(str(cid) for cid in chunk_ids[relevance_matrix[i] > 0])

        # Full coverage check
        if relevant.issubset(retrieved):
            n_covered += 1

        # Recall stats
        total_relevant += len(relevant)
        total_retrieved_relevant += len(relevant & retrieved)
        total_retrieved += len(retrieved)

    return {
        "empirical_coverage": n_covered / n_queries if n_queries > 0 else 0,
        "target_coverage": predictor.calibration_result.coverage_guarantee,
        "coverage_satisfied": (n_covered / n_queries >= 1 - predictor.alpha) if n_queries > 0 else False,
        "recall": total_retrieved_relevant / total_relevant if total_relevant > 0 else 0,
        "avg_retrieved": total_retrieved / n_queries if n_queries > 0 else 0,
        "n_test": n_queries,
    }
