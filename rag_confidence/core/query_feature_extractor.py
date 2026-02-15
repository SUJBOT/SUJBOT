"""
Query Feature Extractor for RAG Confidence Prediction.

Extracts features from query text and similarity distributions to augment
the 14 handcrafted QPP features. These features capture query-level properties
that may affect retrieval quality.

Features Categories:
1. Query Text Features (~10): Length, question type markers, domain indicators
2. Similarity-Derived Features (~5): Global similarity statistics

Usage:
    extractor = QueryFeatureExtractor()
    features = extractor.extract(query_text, similarities)
    # or batch:
    features = extractor.extract_batch(query_texts, similarity_matrix)
"""

import numpy as np
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Any, Optional


@dataclass
class QueryFeatures:
    """Container for query-derived features.

    Total: ~15 features organized into 2 categories:
    - Query Text (10): Extracted from query string
    - Similarity-Derived (5): Computed from full similarity distribution
    """

    # Query Text Features (10)
    query_char_len: int           # Character count
    query_word_count: int         # Word count
    starts_with_co: int           # "Co" questions (what)
    starts_with_jake: int         # "Jak/Jaké/Jaký" questions (how/what kind)
    starts_with_proc: int         # "Proč" questions (why)
    starts_with_kdo: int          # "Kdo" questions (who)
    starts_with_kdy: int          # "Kdy" questions (when)
    starts_with_kolik: int        # "Kolik" questions (how much)
    has_numbers: int              # Contains digits
    n_commas: int                 # Punctuation complexity
    has_paragraph: int            # Contains § or "odstavec" (legal marker)
    has_vyhlasky: int             # Contains "vyhlášk" (decree marker)
    has_zakon: int                # Contains "zákon" (law marker)

    # Similarity-Derived Features (5)
    sim_mean_all: float           # Mean similarity to entire corpus
    sim_median_all: float         # Median similarity
    sim_max: float                # Maximum similarity (top-1)
    frac_above_05: float          # Fraction of corpus with sim > 0.5
    frac_above_06: float          # Fraction of corpus with sim > 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array in canonical order."""
        return np.array([
            # Query text features
            self.query_char_len,
            self.query_word_count,
            self.starts_with_co,
            self.starts_with_jake,
            self.starts_with_proc,
            self.starts_with_kdo,
            self.starts_with_kdy,
            self.starts_with_kolik,
            self.has_numbers,
            self.n_commas,
            self.has_paragraph,
            self.has_vyhlasky,
            self.has_zakon,
            # Similarity-derived features
            self.sim_mean_all,
            self.sim_median_all,
            self.sim_max,
            self.frac_above_05,
            self.frac_above_06,
        ], dtype=np.float32)


class QueryFeatureExtractor:
    """Extract query-level features for RAG confidence augmentation.

    Combines:
    1. Query text analysis (length, question type, domain markers)
    2. Similarity distribution statistics (mean, median, thresholds)

    Example:
        extractor = QueryFeatureExtractor()
        features = extractor.extract(
            "Jaké podmínky platí pro jaderná zařízení?",
            similarities  # shape: (n_chunks,)
        )
        X = features.to_vector()  # shape: (18,)
    """

    # Czech question starters (case-insensitive matching)
    QUESTION_STARTERS = {
        "co": "starts_with_co",
        "jak": "starts_with_jake",  # Covers Jak, Jaké, Jaký, Jakou, etc.
        "proč": "starts_with_proc",
        "kdo": "starts_with_kdo",
        "kdy": "starts_with_kdy",
        "kolik": "starts_with_kolik",
    }

    # Czech legal domain markers
    LEGAL_MARKERS = {
        "paragraph": ["§", "odstavec", "odst."],
        "vyhlasky": ["vyhlášk", "vyhláška", "vyhlášce", "vyhláškou"],
        "zakon": ["zákon", "zákona", "zákoně", "zákonem"],
    }

    # Canonical feature order
    FEATURE_NAMES = [
        "query_char_len",
        "query_word_count",
        "starts_with_co",
        "starts_with_jake",
        "starts_with_proc",
        "starts_with_kdo",
        "starts_with_kdy",
        "starts_with_kolik",
        "has_numbers",
        "n_commas",
        "has_paragraph",
        "has_vyhlasky",
        "has_zakon",
        "sim_mean_all",
        "sim_median_all",
        "sim_max",
        "frac_above_05",
        "frac_above_06",
    ]

    def __init__(self):
        """Initialize extractor."""
        pass

    def extract_text_features(self, query_text: str) -> Dict[str, Any]:
        """Extract features from query text only.

        Args:
            query_text: The query string.

        Returns:
            Dictionary of text-based features.
        """
        query_lower = query_text.lower()
        words = query_text.split()

        # Basic length features
        features = {
            "query_char_len": len(query_text),
            "query_word_count": len(words),
        }

        # Question type markers (one-hot encoded)
        for starter, feature_name in self.QUESTION_STARTERS.items():
            features[feature_name] = int(query_lower.startswith(starter))

        # Complexity indicators
        features["has_numbers"] = int(any(c.isdigit() for c in query_text))
        features["n_commas"] = query_text.count(",")

        # Legal domain markers
        features["has_paragraph"] = int(
            any(marker in query_lower for marker in self.LEGAL_MARKERS["paragraph"])
        )
        features["has_vyhlasky"] = int(
            any(marker in query_lower for marker in self.LEGAL_MARKERS["vyhlasky"])
        )
        features["has_zakon"] = int(
            any(marker in query_lower for marker in self.LEGAL_MARKERS["zakon"])
        )

        return features

    def extract_similarity_features(self, similarities: np.ndarray) -> Dict[str, float]:
        """Extract features from full similarity distribution.

        These features capture the "query genericness" - how similar the query
        is to the overall corpus, which affects retrieval quality.

        Args:
            similarities: Array of similarities to all chunks, shape (n_chunks,).

        Returns:
            Dictionary of similarity-derived features.
        """
        return {
            "sim_mean_all": float(np.mean(similarities)),
            "sim_median_all": float(np.median(similarities)),
            "sim_max": float(np.max(similarities)),
            "frac_above_05": float(np.mean(similarities > 0.5)),
            "frac_above_06": float(np.mean(similarities > 0.6)),
        }

    def extract(
        self,
        query_text: str,
        similarities: Optional[np.ndarray] = None
    ) -> QueryFeatures:
        """Extract all query features.

        Args:
            query_text: The query string.
            similarities: Optional array of similarities to all chunks.
                          If None, similarity features will be 0.

        Returns:
            QueryFeatures dataclass with all extracted features.
        """
        # Extract text features
        text_features = self.extract_text_features(query_text)

        # Extract similarity features (if provided)
        if similarities is not None:
            sim_features = self.extract_similarity_features(similarities)
        else:
            sim_features = {
                "sim_mean_all": 0.0,
                "sim_median_all": 0.0,
                "sim_max": 0.0,
                "frac_above_05": 0.0,
                "frac_above_06": 0.0,
            }

        # Combine all features
        return QueryFeatures(
            # Text features
            query_char_len=text_features["query_char_len"],
            query_word_count=text_features["query_word_count"],
            starts_with_co=text_features["starts_with_co"],
            starts_with_jake=text_features["starts_with_jake"],
            starts_with_proc=text_features["starts_with_proc"],
            starts_with_kdo=text_features["starts_with_kdo"],
            starts_with_kdy=text_features["starts_with_kdy"],
            starts_with_kolik=text_features["starts_with_kolik"],
            has_numbers=text_features["has_numbers"],
            n_commas=text_features["n_commas"],
            has_paragraph=text_features["has_paragraph"],
            has_vyhlasky=text_features["has_vyhlasky"],
            has_zakon=text_features["has_zakon"],
            # Similarity features
            sim_mean_all=sim_features["sim_mean_all"],
            sim_median_all=sim_features["sim_median_all"],
            sim_max=sim_features["sim_max"],
            frac_above_05=sim_features["frac_above_05"],
            frac_above_06=sim_features["frac_above_06"],
        )

    def extract_batch(
        self,
        query_texts: List[str],
        similarity_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract features for multiple queries.

        Args:
            query_texts: List of query strings.
            similarity_matrix: Optional matrix of shape (n_queries, n_chunks).

        Returns:
            Feature matrix of shape (n_queries, 18).
        """
        n_queries = len(query_texts)
        features = np.zeros((n_queries, len(self.FEATURE_NAMES)), dtype=np.float32)

        for i, query_text in enumerate(query_texts):
            similarities = similarity_matrix[i] if similarity_matrix is not None else None
            qf = self.extract(query_text, similarities)
            features[i] = qf.to_vector()

        return features

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered feature names."""
        return QueryFeatureExtractor.FEATURE_NAMES.copy()

    @staticmethod
    def n_features() -> int:
        """Return number of features."""
        return len(QueryFeatureExtractor.FEATURE_NAMES)


def extract_raw_similarity_features(
    similarity_row: np.ndarray,
    top_k: int = 1000
) -> np.ndarray:
    """Extract top-k sorted similarity scores as features.

    This is used for the Raw Similarity MLP that learns features
    directly from the similarity distribution.

    Args:
        similarity_row: Array of similarities to all chunks, shape (n_chunks,).
        top_k: Number of top similarities to use as features.

    Returns:
        Array of top-k sorted similarities, shape (top_k,).
    """
    sorted_sims = np.sort(similarity_row)[::-1]  # Descending
    n = len(sorted_sims)

    if n >= top_k:
        return sorted_sims[:top_k].astype(np.float32)
    else:
        # Pad with zeros if fewer than top_k chunks
        result = np.zeros(top_k, dtype=np.float32)
        result[:n] = sorted_sims
        return result


def extract_raw_similarity_batch(
    similarity_matrix: np.ndarray,
    top_k: int = 1000
) -> np.ndarray:
    """Extract top-k similarities for multiple queries.

    Args:
        similarity_matrix: Shape (n_queries, n_chunks).
        top_k: Number of top similarities per query.

    Returns:
        Feature matrix of shape (n_queries, top_k).
    """
    n_queries = similarity_matrix.shape[0]
    features = np.zeros((n_queries, top_k), dtype=np.float32)

    for i in range(n_queries):
        features[i] = extract_raw_similarity_features(similarity_matrix[i], top_k)

    return features


if __name__ == "__main__":
    # Quick test with sample data
    print("Testing QueryFeatureExtractor...")

    extractor = QueryFeatureExtractor()

    # Sample Czech legal queries
    test_queries = [
        "Co obsahuje seznam zkratek v technické zprávě?",
        "Jaké podmínky platí pro umístění jaderných zařízení?",
        "Proč je důležité dodržovat § 15 odstavec 3 atomového zákona?",
        "Kolik činí maximální pokuta podle vyhlášky 379/2016 Sb.?",
        "Kdy nabývá účinnosti zákon o jaderné bezpečnosti?",
    ]

    # Simulate similarity scores
    np.random.seed(42)
    n_chunks = 5704
    sim_matrix = np.random.uniform(0.3, 0.9, (len(test_queries), n_chunks))

    print("\nQuery Features:")
    print("-" * 60)

    for i, query in enumerate(test_queries):
        features = extractor.extract(query, sim_matrix[i])
        print(f"\nQuery: {query[:50]}...")
        print(f"  char_len: {features.query_char_len}")
        print(f"  word_count: {features.query_word_count}")
        print(f"  starts_with_co: {features.starts_with_co}")
        print(f"  starts_with_jake: {features.starts_with_jake}")
        print(f"  starts_with_proc: {features.starts_with_proc}")
        print(f"  starts_with_kolik: {features.starts_with_kolik}")
        print(f"  starts_with_kdy: {features.starts_with_kdy}")
        print(f"  has_paragraph: {features.has_paragraph}")
        print(f"  has_vyhlasky: {features.has_vyhlasky}")
        print(f"  has_zakon: {features.has_zakon}")
        print(f"  sim_mean: {features.sim_mean_all:.4f}")
        print(f"  sim_max: {features.sim_max:.4f}")

    # Test batch extraction
    print("\n\nBatch Extraction:")
    print("-" * 60)
    batch_features = extractor.extract_batch(test_queries, sim_matrix)
    print(f"Feature matrix shape: {batch_features.shape}")
    print(f"Feature names: {extractor.feature_names()}")

    # Test raw similarity extraction
    print("\n\nRaw Similarity Extraction:")
    print("-" * 60)
    for top_k in [100, 500, 1000]:
        raw_features = extract_raw_similarity_batch(sim_matrix, top_k=top_k)
        print(f"Top-{top_k}: shape={raw_features.shape}, "
              f"range=[{raw_features.min():.3f}, {raw_features.max():.3f}]")

    print("\nAll tests passed!")
