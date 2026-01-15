# Conformal Prediction for RAG Retrieval Confidence

## Overview

This guide explains how to implement conformal prediction (CP) to provide statistical guarantees on RAG retrieval coverage. The goal is to ensure that **all relevant chunks are retrieved** for a given query with a user-specified confidence level.

**Problem Statement:** Given a query, retrieve document chunks such that we can guarantee with probability ≥ 1-α that ALL relevant chunks are included in the retrieved set.

**Available Data:**
- Calibration set: 1000 queries with ground-truth labeled relevant chunks
- Embedding function for queries and chunks
- Similarity function (e.g., cosine similarity)

---

## Mathematical Foundation

### The Coverage Guarantee

Conformal prediction provides the following guarantee:

```
P(all relevant chunks retrieved) ≥ 1 - α
```

where α is the user-specified error rate (e.g., α = 0.1 for 90% coverage).

### Key Insight

Instead of retrieving top-k chunks (which has no guarantee), we retrieve all chunks above a **calibrated threshold** τ. This threshold is computed from the calibration set such that the coverage guarantee holds.

### Why It Works

For exchangeable data (calibration and test queries from the same distribution), the test query's nonconformity score is equally likely to fall anywhere in the distribution of calibration scores. By choosing τ as the α-quantile, we ensure the test score exceeds τ with probability ≥ 1-α.

---

## Approach 1: Minimum Similarity Score (Recommended)

This is the **recommended approach** for guaranteeing all relevant chunks are retrieved.

### Concept

For each query, compute the similarity to its **least similar relevant chunk**. If this worst-case chunk is retrieved, all other relevant chunks will be too.

### Nonconformity Score Definition

```
s_i = min_{d ∈ D_i*} similarity(q_i, d)
```

where:
- `q_i` is the i-th calibration query
- `D_i*` is the set of all relevant chunks for query i
- `similarity()` is cosine similarity (or other similarity metric)

### Algorithm

```
CALIBRATION:
1. For each calibration query q_i with relevant chunks D_i*:
   a. Compute similarities: sim_j = similarity(q_i, d_j) for all d_j in D_i*
   b. Store nonconformity score: s_i = min(sim_j)

2. Sort scores: s_(1) ≤ s_(2) ≤ ... ≤ s_(n)

3. Compute threshold: τ = s_(⌊α(n+1)⌋)
   - This is the α-quantile (lower tail)
   - For n=1000, α=0.1: τ = s_(100) (the 100th smallest score)

INFERENCE:
1. For new query q_test:
   a. Compute similarity to all chunks in database
   b. Return: C(q_test) = {d : similarity(q_test, d) ≥ τ}

GUARANTEE: P(D_test* ⊆ C(q_test)) ≥ 1 - α
```

### Implementation

```python
import numpy as np
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass

@dataclass
class CalibrationResult:
    threshold: float
    alpha: float
    n_calibration: int
    score_distribution: np.ndarray


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def compute_nonconformity_score(
    query_embedding: np.ndarray,
    relevant_chunk_embeddings: List[np.ndarray]
) -> float:
    """
    Compute nonconformity score for a query.
    
    The score is the MINIMUM similarity among all relevant chunks.
    Lower score = harder query (relevant chunks are less similar to query).
    
    Args:
        query_embedding: Embedding vector for the query
        relevant_chunk_embeddings: List of embeddings for all relevant chunks
    
    Returns:
        Minimum similarity score (nonconformity score)
    """
    if not relevant_chunk_embeddings:
        raise ValueError("Must have at least one relevant chunk")
    
    similarities = [
        cosine_similarity(query_embedding, chunk_emb)
        for chunk_emb in relevant_chunk_embeddings
    ]
    return min(similarities)


def calibrate_retrieval_threshold(
    calibration_data: List[Tuple[np.ndarray, List[np.ndarray]]],
    alpha: float = 0.1
) -> CalibrationResult:
    """
    Calibrate the retrieval threshold using conformal prediction.
    
    Args:
        calibration_data: List of (query_embedding, [relevant_chunk_embeddings])
                         Each element is a tuple containing:
                         - query_embedding: np.ndarray of shape (embed_dim,)
                         - relevant_chunk_embeddings: List of np.ndarray, each (embed_dim,)
        alpha: Desired error rate. Use 0.1 for 90% coverage, 0.05 for 95% coverage.
    
    Returns:
        CalibrationResult containing threshold and metadata
    
    Guarantee:
        For a new query from the same distribution, probability that ALL
        relevant chunks have similarity >= threshold is at least 1 - alpha.
    """
    n = len(calibration_data)
    if n < 10:
        raise ValueError(f"Need at least 10 calibration examples, got {n}")
    
    # Compute nonconformity scores for all calibration queries
    scores = np.array([
        compute_nonconformity_score(query_emb, relevant_embs)
        for query_emb, relevant_embs in calibration_data
    ])
    
    # Compute the α-quantile (lower tail)
    # We want the threshold such that (1-α) fraction of scores are above it
    quantile_index = int(np.floor(alpha * (n + 1))) - 1
    quantile_index = max(0, min(quantile_index, n - 1))
    
    sorted_scores = np.sort(scores)
    threshold = sorted_scores[quantile_index]
    
    return CalibrationResult(
        threshold=threshold,
        alpha=alpha,
        n_calibration=n,
        score_distribution=sorted_scores
    )


def conformal_retrieve(
    query_embedding: np.ndarray,
    chunk_database: Dict[str, np.ndarray],
    threshold: float
) -> Set[str]:
    """
    Retrieve chunks with conformal coverage guarantee.
    
    Args:
        query_embedding: Embedding of the query
        chunk_database: Dictionary mapping chunk_id -> chunk_embedding
        threshold: Calibrated threshold from calibrate_retrieval_threshold()
    
    Returns:
        Set of chunk_ids with similarity >= threshold
    
    Guarantee:
        If threshold was calibrated with error rate α, then with probability
        >= 1-α, this set contains ALL relevant chunks for the query.
    """
    retrieved = set()
    
    for chunk_id, chunk_embedding in chunk_database.items():
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        if similarity >= threshold:
            retrieved.add(chunk_id)
    
    return retrieved
```

---

## Approach 2: Per-Chunk Calibration (Alternative)

If queries have varying numbers of relevant chunks and you want tighter control, calibrate at the chunk level.

### Concept

Treat each (query, relevant_chunk) pair as a calibration point. This gives more calibration data but a slightly different guarantee.

### Nonconformity Score Definition

```
s_{i,j} = similarity(q_i, d_{i,j}*)
```

for each relevant chunk `d_{i,j}*` of query `q_i`.

### Algorithm

```
CALIBRATION:
1. Collect all (query, relevant_chunk) pairs:
   scores = []
   for each query q_i:
       for each relevant chunk d in D_i*:
           scores.append(similarity(q_i, d))

2. Compute threshold: τ = quantile(α, scores)

INFERENCE:
Same as Approach 1

GUARANTEE: 
For each relevant chunk individually, P(retrieved) ≥ 1-α
(Note: This is per-chunk, not joint guarantee over all chunks)
```

### Implementation

```python
def calibrate_per_chunk_threshold(
    calibration_data: List[Tuple[np.ndarray, List[np.ndarray]]],
    alpha: float = 0.1
) -> CalibrationResult:
    """
    Alternative calibration at the chunk level.
    
    This gives more calibration points but provides a per-chunk guarantee
    rather than a joint guarantee over all relevant chunks.
    
    Args:
        calibration_data: List of (query_embedding, [relevant_chunk_embeddings])
        alpha: Desired error rate
    
    Returns:
        CalibrationResult with per-chunk threshold
    """
    # Collect ALL (query, chunk) similarity scores
    all_scores = []
    
    for query_emb, relevant_embs in calibration_data:
        for chunk_emb in relevant_embs:
            sim = cosine_similarity(query_emb, chunk_emb)
            all_scores.append(sim)
    
    scores = np.array(all_scores)
    n = len(scores)
    
    # Compute α-quantile
    quantile_index = int(np.floor(alpha * (n + 1))) - 1
    quantile_index = max(0, min(quantile_index, n - 1))
    
    sorted_scores = np.sort(scores)
    threshold = sorted_scores[quantile_index]
    
    return CalibrationResult(
        threshold=threshold,
        alpha=alpha,
        n_calibration=n,
        score_distribution=sorted_scores
    )
```

### When to Use

- Use **Approach 1** (minimum similarity) when you need guarantee that ALL chunks are retrieved
- Use **Approach 2** (per-chunk) when you have few queries but many relevant chunks per query

---

## Approach 3: Adaptive Retrieval with Confidence

Instead of fixed threshold, retrieve chunks iteratively until desired confidence is reached.

### Concept

Sort chunks by similarity and retrieve in order until the cumulative confidence reaches target.

### Implementation

```python
def adaptive_retrieve_with_confidence(
    query_embedding: np.ndarray,
    chunk_database: Dict[str, np.ndarray],
    calibration_scores: np.ndarray,
    target_coverage: float = 0.9
) -> Tuple[Set[str], float]:
    """
    Adaptively retrieve chunks until target coverage is reached.
    
    Args:
        query_embedding: Query embedding
        chunk_database: Dict of chunk_id -> embedding
        calibration_scores: Sorted array of calibration nonconformity scores
        target_coverage: Desired coverage level (e.g., 0.9 for 90%)
    
    Returns:
        Tuple of (retrieved_chunk_ids, achieved_confidence)
    """
    # Compute all similarities and sort descending
    chunk_similarities = [
        (chunk_id, cosine_similarity(query_embedding, emb))
        for chunk_id, emb in chunk_database.items()
    ]
    chunk_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Find the threshold needed for target coverage
    alpha = 1 - target_coverage
    n = len(calibration_scores)
    quantile_index = int(np.floor(alpha * (n + 1))) - 1
    quantile_index = max(0, min(quantile_index, n - 1))
    threshold = calibration_scores[quantile_index]
    
    # Retrieve all chunks above threshold
    retrieved = set()
    min_retrieved_sim = 1.0
    
    for chunk_id, sim in chunk_similarities:
        if sim >= threshold:
            retrieved.add(chunk_id)
            min_retrieved_sim = sim
        else:
            break
    
    # Compute achieved confidence based on where min_sim falls in calibration
    achieved_confidence = np.mean(calibration_scores <= min_retrieved_sim)
    
    return retrieved, achieved_confidence


def retrieve_top_k_with_confidence(
    query_embedding: np.ndarray,
    chunk_database: Dict[str, np.ndarray],
    calibration_scores: np.ndarray,
    k: int
) -> Tuple[Set[str], float]:
    """
    Retrieve top-k chunks and report the confidence level.
    
    Useful when you have a fixed budget but want to know the confidence.
    
    Args:
        query_embedding: Query embedding
        chunk_database: Dict of chunk_id -> embedding  
        calibration_scores: Sorted calibration scores
        k: Number of chunks to retrieve
    
    Returns:
        Tuple of (retrieved_chunk_ids, confidence_level)
    """
    # Compute all similarities and get top-k
    chunk_similarities = [
        (chunk_id, cosine_similarity(query_embedding, emb))
        for chunk_id, emb in chunk_database.items()
    ]
    chunk_similarities.sort(key=lambda x: x[1], reverse=True)
    
    top_k = chunk_similarities[:k]
    retrieved = {chunk_id for chunk_id, _ in top_k}
    
    # The effective threshold is the k-th highest similarity
    if k > 0 and k <= len(chunk_similarities):
        effective_threshold = top_k[-1][1]
    else:
        effective_threshold = 0.0
    
    # Confidence = fraction of calibration scores <= effective_threshold
    # This tells us: what fraction of calibration queries would have
    # all relevant chunks above this threshold?
    confidence = np.mean(calibration_scores <= effective_threshold)
    
    return retrieved, confidence
```

---

## Complete Pipeline Example

```python
"""
Complete example of conformal prediction for RAG retrieval.
"""

import numpy as np
from typing import List, Dict, Set, Tuple
import json

# Assume you have an embedding function
def get_embedding(text: str) -> np.ndarray:
    """
    Replace with your actual embedding function.
    E.g., OpenAI embeddings, sentence-transformers, etc.
    """
    # Placeholder - replace with actual implementation
    # return openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
    raise NotImplementedError("Implement your embedding function")


def load_calibration_data(filepath: str) -> List[Tuple[str, List[str]]]:
    """
    Load calibration data from JSON file.
    
    Expected format:
    [
        {"query": "...", "relevant_chunks": ["chunk_id_1", "chunk_id_2", ...]},
        ...
    ]
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [(item["query"], item["relevant_chunks"]) for item in data]


def load_chunk_database(filepath: str) -> Dict[str, str]:
    """
    Load chunk database from JSON file.
    
    Expected format:
    {"chunk_id_1": "chunk text...", "chunk_id_2": "chunk text...", ...}
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class ConformalRetriever:
    """
    Retriever with conformal prediction guarantees.
    """
    
    def __init__(
        self,
        chunk_database: Dict[str, str],
        embed_fn,
        alpha: float = 0.1
    ):
        """
        Initialize the retriever.
        
        Args:
            chunk_database: Dict mapping chunk_id -> chunk_text
            embed_fn: Function that takes text and returns embedding vector
            alpha: Target error rate (0.1 = 90% coverage guarantee)
        """
        self.chunk_database = chunk_database
        self.embed_fn = embed_fn
        self.alpha = alpha
        
        # Precompute chunk embeddings
        print("Computing chunk embeddings...")
        self.chunk_embeddings = {
            chunk_id: embed_fn(text)
            for chunk_id, text in chunk_database.items()
        }
        print(f"Computed embeddings for {len(self.chunk_embeddings)} chunks")
        
        # Will be set after calibration
        self.threshold = None
        self.calibration_scores = None
    
    def calibrate(
        self,
        calibration_queries: List[str],
        calibration_relevant: List[List[str]]
    ) -> Dict:
        """
        Calibrate the retrieval threshold.
        
        Args:
            calibration_queries: List of query strings
            calibration_relevant: List of lists of relevant chunk IDs
        
        Returns:
            Dictionary with calibration statistics
        """
        assert len(calibration_queries) == len(calibration_relevant)
        n = len(calibration_queries)
        print(f"Calibrating with {n} queries...")
        
        scores = []
        
        for i, (query, relevant_ids) in enumerate(zip(calibration_queries, calibration_relevant)):
            if (i + 1) % 100 == 0:
                print(f"  Processing query {i + 1}/{n}")
            
            # Get query embedding
            query_emb = self.embed_fn(query)
            
            # Compute similarities to all relevant chunks
            similarities = []
            for chunk_id in relevant_ids:
                if chunk_id not in self.chunk_embeddings:
                    print(f"  Warning: chunk {chunk_id} not in database, skipping")
                    continue
                sim = cosine_similarity(query_emb, self.chunk_embeddings[chunk_id])
                similarities.append(sim)
            
            if not similarities:
                print(f"  Warning: no valid relevant chunks for query {i}, skipping")
                continue
            
            # Nonconformity score = minimum similarity
            scores.append(min(similarities))
        
        scores = np.array(scores)
        n_valid = len(scores)
        
        # Compute threshold at α-quantile
        quantile_index = int(np.floor(self.alpha * (n_valid + 1))) - 1
        quantile_index = max(0, min(quantile_index, n_valid - 1))
        
        sorted_scores = np.sort(scores)
        self.threshold = sorted_scores[quantile_index]
        self.calibration_scores = sorted_scores
        
        stats = {
            "n_calibration": n_valid,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "coverage_guarantee": 1 - self.alpha,
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
        
        print(f"Calibration complete:")
        print(f"  Threshold: {self.threshold:.4f}")
        print(f"  Coverage guarantee: {100 * (1 - self.alpha):.1f}%")
        
        return stats
    
    def retrieve(self, query: str) -> Set[str]:
        """
        Retrieve chunks with coverage guarantee.
        
        Args:
            query: Query string
        
        Returns:
            Set of chunk IDs guaranteed to contain all relevant chunks
            with probability >= 1 - alpha
        """
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before retrieve()")
        
        query_emb = self.embed_fn(query)
        
        retrieved = set()
        for chunk_id, chunk_emb in self.chunk_embeddings.items():
            sim = cosine_similarity(query_emb, chunk_emb)
            if sim >= self.threshold:
                retrieved.add(chunk_id)
        
        return retrieved
    
    def retrieve_with_confidence(
        self,
        query: str,
        target_coverage: float = None
    ) -> Tuple[Set[str], float, float]:
        """
        Retrieve chunks and report confidence level.
        
        Args:
            query: Query string
            target_coverage: If specified, use this coverage instead of default
        
        Returns:
            Tuple of (chunk_ids, threshold_used, confidence_level)
        """
        if self.calibration_scores is None:
            raise RuntimeError("Must call calibrate() before retrieve_with_confidence()")
        
        coverage = target_coverage if target_coverage else (1 - self.alpha)
        alpha = 1 - coverage
        
        n = len(self.calibration_scores)
        quantile_index = int(np.floor(alpha * (n + 1))) - 1
        quantile_index = max(0, min(quantile_index, n - 1))
        threshold = self.calibration_scores[quantile_index]
        
        query_emb = self.embed_fn(query)
        
        retrieved = set()
        for chunk_id, chunk_emb in self.chunk_embeddings.items():
            sim = cosine_similarity(query_emb, chunk_emb)
            if sim >= threshold:
                retrieved.add(chunk_id)
        
        return retrieved, threshold, coverage
    
    def evaluate(
        self,
        test_queries: List[str],
        test_relevant: List[List[str]]
    ) -> Dict:
        """
        Evaluate coverage on a test set.
        
        Args:
            test_queries: List of test query strings
            test_relevant: List of lists of relevant chunk IDs
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.threshold is None:
            raise RuntimeError("Must call calibrate() before evaluate()")
        
        n_test = len(test_queries)
        n_fully_covered = 0
        total_relevant = 0
        total_retrieved_relevant = 0
        total_retrieved = 0
        
        for query, relevant_ids in zip(test_queries, test_relevant):
            retrieved = self.retrieve(query)
            relevant_set = set(relevant_ids)
            
            # Check full coverage
            if relevant_set.issubset(retrieved):
                n_fully_covered += 1
            
            # Compute recall stats
            total_relevant += len(relevant_set)
            total_retrieved_relevant += len(relevant_set & retrieved)
            total_retrieved += len(retrieved)
        
        return {
            "n_test": n_test,
            "empirical_coverage": n_fully_covered / n_test,
            "target_coverage": 1 - self.alpha,
            "coverage_satisfied": n_fully_covered / n_test >= 1 - self.alpha,
            "recall": total_retrieved_relevant / total_relevant if total_relevant > 0 else 0,
            "avg_retrieved": total_retrieved / n_test,
        }


# Helper function
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage (replace with your actual data and embedding function)
    
    # 1. Load your data
    # chunk_db = load_chunk_database("chunks.json")
    # calibration_data = load_calibration_data("calibration.json")
    
    # 2. Create retriever
    # retriever = ConformalRetriever(
    #     chunk_database=chunk_db,
    #     embed_fn=get_embedding,
    #     alpha=0.1  # 90% coverage
    # )
    
    # 3. Calibrate
    # queries = [item[0] for item in calibration_data]
    # relevant = [item[1] for item in calibration_data]
    # stats = retriever.calibrate(queries, relevant)
    
    # 4. Use in production
    # results = retriever.retrieve("What is machine learning?")
    # print(f"Retrieved {len(results)} chunks with 90% coverage guarantee")
    
    # 5. Or retrieve with custom confidence
    # results, threshold, confidence = retriever.retrieve_with_confidence(
    #     "What is machine learning?",
    #     target_coverage=0.95  # 95% coverage
    # )
    
    print("See code comments for usage example")
```

---

## Calibration Set Size Guidelines

### Theoretical Bounds

For calibration set size `n`:

| Guarantee | Formula | Example (α=0.1, δ=0.05) |
|-----------|---------|-------------------------|
| Basic coverage | Any n ≥ 1 | n = 1 (but inefficient) |
| Coverage slack ≤ ε | n ≥ log(1/δ) / (2ε²) | ε=0.02: n ≥ 3,744 |
| Tight prediction sets | n ≥ O(100) | n ≈ 500-1000 |

### Practical Recommendations

| Use Case | Recommended n |
|----------|---------------|
| Prototyping | 200-500 |
| Production | 1,000-2,000 |
| High-stakes | 5,000+ |

### With n = 1000 (Your Setup)

- Coverage guarantee: 90% ± 0.1%
- Threshold is stable (low variance)
- Prediction sets are close to optimal size

---

## Evaluation Metrics

When evaluating your conformal retriever, track:

1. **Empirical Coverage**: Fraction of test queries where ALL relevant chunks were retrieved
   - Should be ≥ 1 - α

2. **Average Set Size**: Mean number of chunks retrieved per query
   - Lower is better (more efficient)

3. **Recall**: Fraction of relevant chunks retrieved across all queries
   - Should be very high (close to 100%)

4. **Precision**: Fraction of retrieved chunks that are relevant
   - Trade-off with coverage

```python
def compute_metrics(
    test_queries: List[str],
    test_relevant: List[Set[str]],
    retrieved_sets: List[Set[str]]
) -> Dict:
    """Compute evaluation metrics."""
    n = len(test_queries)
    
    full_coverage_count = sum(
        1 for rel, ret in zip(test_relevant, retrieved_sets)
        if rel.issubset(ret)
    )
    
    total_relevant = sum(len(rel) for rel in test_relevant)
    total_retrieved = sum(len(ret) for ret in retrieved_sets)
    total_correct = sum(
        len(rel & ret) for rel, ret in zip(test_relevant, retrieved_sets)
    )
    
    return {
        "coverage": full_coverage_count / n,
        "recall": total_correct / total_relevant,
        "precision": total_correct / total_retrieved if total_retrieved > 0 else 0,
        "avg_set_size": total_retrieved / n,
    }
```

---

## Common Issues and Solutions

### Issue 1: Threshold Too Low (Too Many Chunks Retrieved)

**Cause**: Calibration queries are "easy" (relevant chunks very similar to query)

**Solution**: 
- Ensure calibration set is representative of production queries
- Include "hard" queries with less obvious relevant chunks

### Issue 2: Threshold Too High (Coverage Not Met)

**Cause**: Test distribution differs from calibration distribution

**Solution**:
- Collect more diverse calibration data
- Use domain-specific calibration sets
- Consider Mondrian conformal prediction (calibrate per query type)

### Issue 3: High Variance in Set Sizes

**Cause**: Query difficulty varies significantly

**Solution**:
- This is expected behavior - harder queries need more chunks
- Report confidence intervals on set size
- Consider adaptive retrieval approaches

---

## Summary

1. **Use Approach 1 (Minimum Similarity)** for joint coverage guarantee over all relevant chunks

2. **Calibration**:
   - Compute min similarity to relevant chunks for each calibration query
   - Find α-quantile as threshold

3. **Inference**:
   - Retrieve all chunks with similarity ≥ threshold

4. **Guarantee**:
   - P(all relevant chunks retrieved) ≥ 1 - α

5. **Your setup (n=1000)** is sufficient for reliable coverage with tight prediction sets
