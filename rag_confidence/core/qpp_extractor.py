"""
QPP (Query Performance Prediction) Feature Extractor for RAG Confidence.

Extracts 14 features from similarity score distributions to predict retrieval quality
without requiring relevance labels at inference time.

Features are based on empirical correlation analysis (see IMPROVED_QPP_FEATURES.md).
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from scipy.stats import skew


@dataclass
class QPPFeatures:
    """Container for QPP features extracted from similarity scores.

    Total: 14 features organized into 3 categories:
    - Core (5): Strongest predictors (|r| > 0.35)
    - Extended (7): Moderate predictors (0.25 < |r| < 0.35)
    - Conformal (2): Features based on calibrated threshold
    """

    # Core features (Tier 1) - 5 features
    top1_minus_p99: float       # Best feature (r=+0.382): gap from corpus noise
    top1_vs_top10_gap: float    # r=+0.372: gap between rank 1 and 10
    sim_std_top10: float        # r=+0.366: score spread in top-10
    sim_slope: float            # r=-0.361: linear decay rate
    bimodal_gap: float          # r=+0.351: separation between top cluster and rest

    # Extended features (Tier 2) - 7 features
    exp_decay_rate: float       # r=-0.342: exponential decay rate
    n_above_08: int             # r=-0.324: count of chunks with sim >= 0.8
    n_above_07: int             # r=-0.307: count of chunks with sim >= 0.7
    top5_concentration: float   # r=+0.273: score mass in top-5
    percentile_99: float        # r=-0.333: 99th percentile (query genericity)
    skewness_top50: float       # r=+0.286: distribution asymmetry
    max_second_deriv: float     # r=+0.256: elbow sharpness

    # Conformal-informed features - 2 features
    n_above_tau: int            # count above conformal threshold
    top1_margin_over_tau: float # margin above threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array in canonical order."""
        return np.array([
            self.top1_minus_p99,
            self.top1_vs_top10_gap,
            self.sim_std_top10,
            self.sim_slope,
            self.bimodal_gap,
            self.exp_decay_rate,
            self.n_above_08,
            self.n_above_07,
            self.top5_concentration,
            self.percentile_99,
            self.skewness_top50,
            self.max_second_deriv,
            self.n_above_tau,
            self.top1_margin_over_tau,
        ])


class ImprovedQPPExtractor:
    """Extract QPP features for RAG confidence prediction.

    Takes raw similarity scores (query to all corpus chunks) and extracts
    14 features that predict retrieval quality (Recall@k).

    Features are designed based on empirical correlation analysis on
    1500 synthetic queries from the calibration dataset.

    Example:
        extractor = ImprovedQPPExtractor(tau=0.711)
        similarities = retriever.get_similarities(query)  # shape: (n_chunks,)
        features = extractor.extract(similarities)
        X = features.to_vector()  # shape: (14,)
    """

    # Canonical feature order for model compatibility
    FEATURE_NAMES = [
        "top1_minus_p99",
        "top1_vs_top10_gap",
        "sim_std_top10",
        "sim_slope",
        "bimodal_gap",
        "exp_decay_rate",
        "n_above_08",
        "n_above_07",
        "top5_concentration",
        "percentile_99",
        "skewness_top50",
        "max_second_deriv",
        "n_above_tau",
        "top1_margin_over_tau",
    ]

    def __init__(self, tau: float = 0.711, k: int = 10):
        """Initialize extractor.

        Args:
            tau: Conformal prediction threshold for 90% coverage guarantee.
                 Default 0.711 from calibration on synthetic dataset.
            k: Number of top results to consider for some features.
        """
        self.tau = tau
        self.k = k

    def extract(self, similarities: np.ndarray) -> QPPFeatures:
        """Extract QPP features from similarity scores.

        Args:
            similarities: Array of similarity scores between query and all chunks.
                          Shape: (n_chunks,). Can be unsorted.

        Returns:
            QPPFeatures dataclass with 14 extracted features.
        """
        # Sort descending (highest similarity first)
        sorted_sims = np.sort(similarities)[::-1]
        n = len(sorted_sims)

        # Precompute common values
        p99 = np.percentile(sorted_sims, 99)
        top10 = sorted_sims[:10]
        top50 = sorted_sims[:50] if n >= 50 else sorted_sims
        top100 = sorted_sims[:100] if n >= 100 else sorted_sims

        # Core features
        top1_minus_p99 = sorted_sims[0] - p99
        top1_vs_top10_gap = sorted_sims[0] - sorted_sims[min(9, n-1)]
        sim_std_top10 = float(np.std(top10))

        # Linear slope of top-10 scores vs rank
        if len(top10) >= 2:
            sim_slope = float(np.polyfit(np.arange(len(top10)), top10, 1)[0])
        else:
            sim_slope = 0.0

        # Bimodal gap: mean(top3) - mean(positions 10-30)
        top3_mean = np.mean(sorted_sims[:3])
        rest_mean = np.mean(sorted_sims[10:30]) if n >= 30 else np.mean(sorted_sims[10:])
        bimodal_gap = top3_mean - rest_mean

        # Extended features
        # Exponential decay rate (fit log-linear model)
        if len(top10) >= 2:
            log_scores = np.log(top10 + 1e-8)
            exp_decay_rate = float(np.polyfit(np.arange(len(top10)), log_scores, 1)[0])
        else:
            exp_decay_rate = 0.0

        n_above_08 = int((sorted_sims >= 0.8).sum())
        n_above_07 = int((sorted_sims >= 0.7).sum())

        # Top-5 concentration (score mass ratio)
        top5_sum = sorted_sims[:5].sum()
        top100_sum = top100.sum() + 1e-8
        top5_concentration = top5_sum / top100_sum

        percentile_99 = p99

        # Skewness of top-50 distribution
        if len(top50) >= 3:
            skewness_top50 = float(skew(top50))
        else:
            skewness_top50 = 0.0

        # Maximum second derivative (elbow detection)
        if n >= 32:
            second_deriv = np.diff(sorted_sims[:30], n=2)
            max_second_deriv = float(np.max(second_deriv))
        else:
            max_second_deriv = 0.0

        # Conformal features
        n_above_tau = int((sorted_sims >= self.tau).sum())
        top1_margin_over_tau = max(0.0, sorted_sims[0] - self.tau)

        return QPPFeatures(
            top1_minus_p99=top1_minus_p99,
            top1_vs_top10_gap=top1_vs_top10_gap,
            sim_std_top10=sim_std_top10,
            sim_slope=sim_slope,
            bimodal_gap=bimodal_gap,
            exp_decay_rate=exp_decay_rate,
            n_above_08=n_above_08,
            n_above_07=n_above_07,
            top5_concentration=top5_concentration,
            percentile_99=percentile_99,
            skewness_top50=skewness_top50,
            max_second_deriv=max_second_deriv,
            n_above_tau=n_above_tau,
            top1_margin_over_tau=top1_margin_over_tau,
        )

    def extract_batch(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Extract features for multiple queries.

        Args:
            similarity_matrix: Shape (n_queries, n_chunks).

        Returns:
            Feature matrix of shape (n_queries, 14).
        """
        n_queries = similarity_matrix.shape[0]
        features = np.zeros((n_queries, len(self.FEATURE_NAMES)))

        for i in range(n_queries):
            qpp = self.extract(similarity_matrix[i])
            features[i] = qpp.to_vector()

        return features

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered feature names for model coefficients."""
        return ImprovedQPPExtractor.FEATURE_NAMES.copy()

    def get_feature_importance_baseline(self) -> Dict[str, float]:
        """Return empirical correlation coefficients as feature importance baseline.

        These correlations were computed on the 1500-query calibration dataset.
        """
        return {
            "top1_minus_p99": 0.382,
            "top1_vs_top10_gap": 0.372,
            "sim_std_top10": 0.366,
            "sim_slope": -0.361,  # Negative correlation
            "bimodal_gap": 0.351,
            "exp_decay_rate": -0.342,  # Negative
            "n_above_08": -0.324,  # Negative
            "n_above_07": -0.307,  # Negative
            "top5_concentration": 0.273,
            "percentile_99": -0.333,  # Negative
            "skewness_top50": 0.286,
            "max_second_deriv": 0.256,
            "n_above_tau": -0.30,  # Approximate
            "top1_margin_over_tau": 0.25,  # Approximate
        }


def compute_recall_at_k(
    similarity_matrix: np.ndarray,
    relevance_matrix: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """Compute Recall@k for each query.

    Args:
        similarity_matrix: Shape (n_queries, n_chunks).
        relevance_matrix: Binary matrix, shape (n_queries, n_chunks).
        k: Number of top results to consider.

    Returns:
        Array of Recall@k values, shape (n_queries,).
    """
    n_queries = similarity_matrix.shape[0]
    recalls = np.zeros(n_queries)

    for i in range(n_queries):
        # Get top-k indices by similarity
        top_k_indices = np.argsort(similarity_matrix[i])[::-1][:k]

        # Count relevant in top-k
        relevant_in_topk = relevance_matrix[i, top_k_indices].sum()

        # Total relevant for this query
        total_relevant = relevance_matrix[i].sum()

        if total_relevant > 0:
            recalls[i] = relevant_in_topk / total_relevant
        else:
            recalls[i] = 0.0  # No relevant docs

    return recalls


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Simulate similarity scores (descending)
    good_retrieval = np.array([0.85, 0.72, 0.68, 0.65, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30] +
                               [0.2 + 0.1 * np.random.random() for _ in range(90)])
    bad_retrieval = np.array([0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62] +
                              [0.5 + 0.1 * np.random.random() for _ in range(90)])

    extractor = ImprovedQPPExtractor(tau=0.711)

    print("Good retrieval features:")
    good_features = extractor.extract(good_retrieval)
    for name, val in good_features.to_dict().items():
        print(f"  {name}: {val:.4f}")

    print("\nBad retrieval features:")
    bad_features = extractor.extract(bad_retrieval)
    for name, val in bad_features.to_dict().items():
        print(f"  {name}: {val:.4f}")

    print("\nExpected: Good retrieval should have higher gap/std features, lower threshold counts")
