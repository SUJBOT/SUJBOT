"""
Generalized QPP (Query Performance Prediction) Feature Extractor.

Language-agnostic and domain-agnostic feature extraction for RAG confidence prediction.
Designed for cross-domain and cross-language transferability.

Features: 23 total
- Distribution features (14): From similarity score distributions
- Query features (5): Language-agnostic text statistics
- Global similarity features (4): Corpus-level similarity statistics

Reference: Plan for QPP vs SCA Benchmark (domain-agnostic design)
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from scipy.stats import skew


@dataclass
class GeneralQPPFeatures:
    """Container for generalized QPP features.

    Total: 24 features organized into 3 tiers:
    - Tier 1 (15): Distribution features - strongest predictors
    - Tier 2 (5): Query text features - language-agnostic
    - Tier 3 (4): Global similarity features - corpus-level
    """

    # Tier 1: Distribution Features (15) - from similarity score distribution
    top1_minus_p99: float       # Gap from corpus noise (r=+0.382)
    top1_vs_top10_gap: float    # Gap between rank 1 and 10 (r=+0.372)
    sim_std_top10: float        # Score spread in top-10 (r=+0.366)
    sim_slope: float            # Linear decay rate (r=-0.361)
    bimodal_gap: float          # Separation between top cluster and rest (r=+0.351)
    exp_decay_rate: float       # Exponential decay rate (r=-0.342)
    percentile_99: float        # 99th percentile - query genericity (r=-0.333)
    n_above_08: int             # Count of chunks with sim >= 0.8 (r=-0.324)
    n_above_07: int             # Count of chunks with sim >= 0.7 (r=-0.307)
    skewness_top50: float       # Distribution asymmetry (r=+0.286)
    top5_concentration: float   # Score mass in top-5 (r=+0.273)
    max_second_deriv: float     # Elbow sharpness (r=+0.256)
    n_above_tau: int            # Count above conformal threshold (~0.30)
    top1_margin_over_tau: float # Margin above threshold (~0.25)
    gap_concentration: float    # gap(1-2) / total_gap(1-10) - how much top-1 stands out (+0.0129 AUROC)

    # Tier 2: Query Text Features (5) - language-agnostic
    query_char_len: int         # Character count
    query_word_count: int       # Word count
    has_numbers: int            # Contains digits (0/1)
    n_punctuation: int          # Total punctuation count
    avg_word_length: float      # Mean word length

    # Tier 3: Global Similarity Features (4) - corpus-level statistics
    sim_mean_all: float         # Mean similarity to corpus
    sim_median_all: float       # Median similarity
    sim_max: float              # Maximum similarity (top-1)
    sim_percentile_range: float # p90 - p10 spread

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array in canonical order."""
        return np.array([
            # Tier 1: Distribution features (15)
            self.top1_minus_p99,
            self.top1_vs_top10_gap,
            self.sim_std_top10,
            self.sim_slope,
            self.bimodal_gap,
            self.exp_decay_rate,
            self.percentile_99,
            self.n_above_08,
            self.n_above_07,
            self.skewness_top50,
            self.top5_concentration,
            self.max_second_deriv,
            self.n_above_tau,
            self.top1_margin_over_tau,
            self.gap_concentration,
            # Tier 2: Query text features (5)
            self.query_char_len,
            self.query_word_count,
            self.has_numbers,
            self.n_punctuation,
            self.avg_word_length,
            # Tier 3: Global similarity features (4)
            self.sim_mean_all,
            self.sim_median_all,
            self.sim_max,
            self.sim_percentile_range,
        ], dtype=np.float32)


class GeneralQPPExtractor:
    """
    Language-agnostic QPP feature extraction for RAG confidence.

    Extracts 23 features from query text and similarity distributions:
    - 14 distribution features (from similarity scores)
    - 5 query features (language-agnostic text statistics)
    - 4 global similarity features (corpus-level)

    Example:
        extractor = GeneralQPPExtractor(tau=0.711)
        features = extractor.extract(
            query="What are the safety requirements?",
            similarities=np.array([0.85, 0.72, 0.68, ...])  # (n_chunks,)
        )
        X = features.to_vector()  # shape: (23,)
    """

    # Canonical feature order for model compatibility
    FEATURE_NAMES = [
        # Tier 1: Distribution (15)
        "top1_minus_p99",
        "top1_vs_top10_gap",
        "sim_std_top10",
        "sim_slope",
        "bimodal_gap",
        "exp_decay_rate",
        "percentile_99",
        "n_above_08",
        "n_above_07",
        "skewness_top50",
        "top5_concentration",
        "max_second_deriv",
        "n_above_tau",
        "top1_margin_over_tau",
        "gap_concentration",
        # Tier 2: Query text (5)
        "query_char_len",
        "query_word_count",
        "has_numbers",
        "n_punctuation",
        "avg_word_length",
        # Tier 3: Global similarity (4)
        "sim_mean_all",
        "sim_median_all",
        "sim_max",
        "sim_percentile_range",
    ]

    # Punctuation characters to count
    PUNCTUATION = set(".,;:!?()[]{}\"'-/\\@#$%^&*+=<>|~`")

    def __init__(self, tau: float = 0.711, k: int = 10):
        """
        Initialize extractor.

        Args:
            tau: Conformal prediction threshold for 90% coverage guarantee.
                 Default 0.711 from calibration on synthetic dataset.
            k: Number of top results to consider for some features.
        """
        self.tau = tau
        self.k = k

    def extract(
        self,
        query: str,
        similarities: np.ndarray,
    ) -> GeneralQPPFeatures:
        """
        Extract all QPP features from query and similarities.

        Args:
            query: Query text string.
            similarities: Array of similarity scores between query and all chunks.
                          Shape: (n_chunks,). Can be unsorted.

        Returns:
            GeneralQPPFeatures dataclass with 23 extracted features.
        """
        # Extract each feature tier
        dist_features = self._extract_distribution_features(similarities)
        query_features = self._extract_query_features(query)
        global_features = self._extract_global_features(similarities)

        return GeneralQPPFeatures(
            # Distribution features
            **dist_features,
            # Query features
            **query_features,
            # Global features
            **global_features,
        )

    def _extract_distribution_features(self, similarities: np.ndarray) -> Dict[str, Any]:
        """
        Extract 14 distribution features from similarity scores.

        These features capture the shape of the similarity distribution
        and are the strongest predictors of retrieval quality.
        """
        # Sort descending (highest similarity first)
        sorted_sims = np.sort(similarities)[::-1]
        n = len(sorted_sims)

        # Precompute common values
        p99 = np.percentile(sorted_sims, 99)
        top10 = sorted_sims[:min(10, n)]
        top50 = sorted_sims[:min(50, n)]
        top100 = sorted_sims[:min(100, n)]

        # Core features
        top1_minus_p99 = sorted_sims[0] - p99
        top1_vs_top10_gap = sorted_sims[0] - sorted_sims[min(9, n - 1)]
        sim_std_top10 = float(np.std(top10))

        # Linear slope of top-10 scores vs rank
        if len(top10) >= 2:
            sim_slope = float(np.polyfit(np.arange(len(top10)), top10, 1)[0])
        else:
            sim_slope = 0.0

        # Bimodal gap: mean(top3) - mean(positions 10-30)
        top3_mean = np.mean(sorted_sims[:min(3, n)])
        if n >= 30:
            rest_mean = np.mean(sorted_sims[10:30])
        elif n > 10:
            rest_mean = np.mean(sorted_sims[10:])
        else:
            rest_mean = np.mean(sorted_sims)
        bimodal_gap = top3_mean - rest_mean

        # Exponential decay rate (fit log-linear model)
        if len(top10) >= 2:
            log_scores = np.log(np.maximum(top10, 1e-8))
            exp_decay_rate = float(np.polyfit(np.arange(len(top10)), log_scores, 1)[0])
        else:
            exp_decay_rate = 0.0

        n_above_08 = int((sorted_sims >= 0.8).sum())
        n_above_07 = int((sorted_sims >= 0.7).sum())

        # Top-5 concentration (score mass ratio)
        top5_sum = sorted_sims[:min(5, n)].sum()
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

        # Gap concentration: ratio of gap(1-2) to total gap in top-10
        # High value = top-1 stands out clearly; Low value = uniform distribution
        gap_1_2 = sorted_sims[0] - sorted_sims[1] if n >= 2 else 0.0
        total_gap = sorted_sims[0] - sorted_sims[min(9, n - 1)] if n >= 2 else 1e-8
        gap_concentration = gap_1_2 / (total_gap + 1e-8)

        return {
            "top1_minus_p99": top1_minus_p99,
            "top1_vs_top10_gap": top1_vs_top10_gap,
            "sim_std_top10": sim_std_top10,
            "sim_slope": sim_slope,
            "bimodal_gap": bimodal_gap,
            "exp_decay_rate": exp_decay_rate,
            "percentile_99": percentile_99,
            "n_above_08": n_above_08,
            "n_above_07": n_above_07,
            "skewness_top50": skewness_top50,
            "top5_concentration": top5_concentration,
            "max_second_deriv": max_second_deriv,
            "n_above_tau": n_above_tau,
            "top1_margin_over_tau": top1_margin_over_tau,
            "gap_concentration": gap_concentration,
        }

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """
        Extract 5 language-agnostic query features.

        These features are based on pure text statistics and work
        across any language without requiring language-specific knowledge.
        """
        words = query.split()
        word_count = len(words)

        # Character length
        query_char_len = len(query)

        # Word count
        query_word_count = word_count

        # Has numbers
        has_numbers = int(any(c.isdigit() for c in query))

        # Punctuation count
        n_punctuation = sum(1 for c in query if c in self.PUNCTUATION)

        # Average word length
        if word_count > 0:
            avg_word_length = sum(len(w) for w in words) / word_count
        else:
            avg_word_length = 0.0

        return {
            "query_char_len": query_char_len,
            "query_word_count": query_word_count,
            "has_numbers": has_numbers,
            "n_punctuation": n_punctuation,
            "avg_word_length": avg_word_length,
        }

    def _extract_global_features(self, similarities: np.ndarray) -> Dict[str, float]:
        """
        Extract 4 global similarity features.

        These features capture corpus-level statistics and
        indicate how the query relates to the entire corpus.
        """
        # Basic statistics
        sim_mean_all = float(np.mean(similarities))
        sim_median_all = float(np.median(similarities))
        sim_max = float(np.max(similarities))

        # Percentile range (p90 - p10) - spread measure
        p90 = np.percentile(similarities, 90)
        p10 = np.percentile(similarities, 10)
        sim_percentile_range = p90 - p10

        return {
            "sim_mean_all": sim_mean_all,
            "sim_median_all": sim_median_all,
            "sim_max": sim_max,
            "sim_percentile_range": sim_percentile_range,
        }

    def extract_batch(
        self,
        queries: List[str],
        similarity_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Extract features for multiple queries.

        Args:
            queries: List of query strings.
            similarity_matrix: Shape (n_queries, n_chunks).

        Returns:
            Feature matrix of shape (n_queries, 23).
        """
        n_queries = len(queries)
        features = np.zeros((n_queries, len(self.FEATURE_NAMES)), dtype=np.float32)

        for i, (query, similarities) in enumerate(zip(queries, similarity_matrix)):
            qpp = self.extract(query, similarities)
            features[i] = qpp.to_vector()

        return features

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered feature names for model coefficients."""
        return GeneralQPPExtractor.FEATURE_NAMES.copy()

    @staticmethod
    def n_features() -> int:
        """Return number of features."""
        return len(GeneralQPPExtractor.FEATURE_NAMES)

    def get_feature_tiers(self) -> Dict[str, List[str]]:
        """Return feature names organized by tier."""
        return {
            "distribution": self.FEATURE_NAMES[:15],
            "query_text": self.FEATURE_NAMES[15:20],
            "global_similarity": self.FEATURE_NAMES[20:24],
        }

    def get_feature_importance_baseline(self) -> Dict[str, float]:
        """
        Return empirical correlation coefficients as feature importance baseline.

        These correlations were computed on the 1500-query calibration dataset.
        Note: gap_concentration delta is from ViDoRe ablation study (+0.0129 AUROC).
        """
        return {
            # Distribution features
            "top1_minus_p99": 0.382,
            "top1_vs_top10_gap": 0.372,
            "sim_std_top10": 0.366,
            "sim_slope": -0.361,
            "bimodal_gap": 0.351,
            "exp_decay_rate": -0.342,
            "percentile_99": -0.333,
            "n_above_08": -0.324,
            "n_above_07": -0.307,
            "skewness_top50": 0.286,
            "top5_concentration": 0.273,
            "max_second_deriv": 0.256,
            "n_above_tau": -0.30,
            "top1_margin_over_tau": 0.25,
            "gap_concentration": 0.30,  # New feature: +1.29% AUROC on ViDoRe
            # Query features (estimated, weaker correlations)
            "query_char_len": 0.05,
            "query_word_count": 0.04,
            "has_numbers": 0.02,
            "n_punctuation": 0.01,
            "avg_word_length": 0.03,
            # Global features
            "sim_mean_all": -0.16,
            "sim_median_all": -0.15,
            "sim_max": 0.20,
            "sim_percentile_range": 0.10,
        }


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Simulate similarity scores
    good_retrieval = np.concatenate([
        np.array([0.85, 0.72, 0.68, 0.65, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]),
        0.2 + 0.1 * np.random.random(90)
    ])
    bad_retrieval = np.concatenate([
        np.array([0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62]),
        0.5 + 0.1 * np.random.random(90)
    ])

    extractor = GeneralQPPExtractor(tau=0.711)

    print("Good retrieval features:")
    good_features = extractor.extract("What are the safety requirements for nuclear facilities?", good_retrieval)
    for name, val in good_features.to_dict().items():
        print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    print("\nBad retrieval features:")
    bad_features = extractor.extract("What are the safety requirements for nuclear facilities?", bad_retrieval)
    for name, val in bad_features.to_dict().items():
        print(f"  {name}: {val:.4f}" if isinstance(val, float) else f"  {name}: {val}")

    print(f"\nTotal features: {extractor.n_features()}")
    print(f"Feature tiers: {extractor.get_feature_tiers()}")
