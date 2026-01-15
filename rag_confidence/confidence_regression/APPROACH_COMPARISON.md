# RAG Confidence: Approach Comparison

This document compares two approaches for RAG retrieval confidence estimation:

1. **Conformal Prediction** (implemented in `conformal_predictor.py`)
2. **Regression-based QPP** (proposed in `confidence_regression/`)

---

## Overview

| Aspect | Conformal Prediction | Regression-based QPP |
|--------|---------------------|---------------------|
| **Core idea** | Calibrate threshold from similarity scores | Train ML model on retrieval features |
| **Output** | Binary (above/below threshold) or coverage % | Continuous recall@10 prediction |
| **Guarantees** | Statistical (PAC-style) | Empirical (correlation-based) |
| **Training data** | Query-chunk similarities + relevance labels | Retrieval candidates with scores + labels |
| **Features used** | 1 (similarity score) | 6 (score statistics, entropy, etc.) |
| **Model** | Quantile-based threshold | Ridge/ElasticNet regression |

---

## Approach 1: Conformal Prediction

### Method

1. **Calibration phase:**
   - For each query, find similarity to relevant chunk(s)
   - Collect min-similarity scores across all queries
   - Set threshold τ at α-quantile (e.g., p10 for 90% coverage)

2. **Inference phase:**
   - Retrieve all chunks with similarity ≥ τ
   - OR: Given top-k, compute confidence = P(calibration_score ≥ min_top_k_sim)

### Implementation (our current system)

```python
# Calibration
predictor = ConformalPredictor(alpha=0.1)  # 90% coverage
predictor.calibrate_from_matrices(similarity_matrix, relevance_matrix)
# Result: threshold τ = 0.711

# Inference Option 1: Threshold-based
chunks = predictor.retrieve_with_guarantee(similarities, chunk_ids)
# Returns variable number of chunks (avg 26.2)

# Inference Option 2: Top-k with confidence
chunks, confidence = predictor.top_k_with_confidence(similarities, chunk_ids, k=10)
# Returns fixed 10 chunks + confidence score
```

### Results (from our calibration)

| Metric | Value |
|--------|-------|
| Threshold (τ) | 0.711 |
| Coverage @ τ | 90% |
| Avg chunks retrieved | 26.2 |
| k for 90% coverage | 9 |

### Strengths

- **Statistical guarantees**: Provable coverage bounds (PAC-learning style)
- **No ML training**: Just quantile computation
- **Simple**: Single parameter (α) to tune
- **Interpretable**: Threshold has clear meaning

### Weaknesses

- **Single feature**: Only uses similarity score
- **Fixed threshold**: Same τ for all queries (doesn't adapt to query difficulty)
- **Requires relevance labels**: Needs labeled calibration set

---

## Approach 2: Regression-based QPP

### Method

1. **Feature extraction** (6 regressors from top-10 candidates):
   - `std_norm_top10`: Score standard deviation (spread)
   - `entropy_norm_top10`: Score entropy (uncertainty)
   - `slope_norm_top10`: Score decay slope
   - `top_vs_rest_ratio`: Top-1 vs rest ratio
   - `section_diversity_top10`: Document diversity
   - `query_token_len`: Query length

2. **Training phase:**
   - Compute recall@10 labels for training queries
   - Train Ridge/ElasticNet regression: features → recall@10

3. **Inference phase:**
   - Extract features from retrieval results
   - Predict recall@10
   - Map to confidence bands (HIGH/MEDIUM/LOW)

### Proposed Implementation

```python
# Feature extraction
extractor = ConfidenceFeatureExtractor(k=10)
features = extractor.extract_features(query_example)
# Returns: {std_norm_top10: 0.12, entropy_norm_top10: 2.1, ...}

# Prediction
model = load_model("rag_confidence_model_v1.pkl")
predicted_recall = model.predict([features])
# Returns: 0.85 (predicted recall@10)

# Interpretation
if predicted_recall >= 0.9:
    confidence = "HIGH"
elif predicted_recall >= 0.75:
    confidence = "MEDIUM"
else:
    confidence = "LOW"
```

### Strengths

- **Multi-feature**: Captures multiple aspects of retrieval quality
- **Adaptive**: Different predictions for different query types
- **Interpretable features**: Each regressor has IR/QPP intuition
- **Continuous output**: Fine-grained confidence scores

### Weaknesses

- **No statistical guarantees**: Only empirical correlation
- **Requires training**: Need to fit and tune ML model
- **Feature engineering**: 6 features to compute and maintain
- **Calibration needed**: Thresholds for bands need tuning

---

## Feature Comparison

### Conformal Prediction Features

| Feature | Description |
|---------|-------------|
| `similarity` | Cosine similarity between query and chunk embeddings |

### QPP Regression Features

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `std_norm_top10` | σ(s₁...s₁₀) | Score spread - higher = more confident |
| `entropy_norm_top10` | -Σ pᵢ log pᵢ | Score entropy - lower = more confident |
| `slope_norm_top10` | linear fit slope | Score decay - steeper = more confident |
| `top_vs_rest_ratio` | s₁ / mean(s₂...s₁₀) | Top dominance - higher = more confident |
| `section_diversity` | |unique sections| / 10 | Document spread |
| `query_token_len` | len(query.split()) | Query complexity |

---

## When to Use Each Approach

### Use Conformal Prediction when:

- You need **statistical guarantees** on coverage
- You want **simplicity** (no ML model to maintain)
- You have a **fixed retrieval budget** and want to know confidence
- You're using **threshold-based retrieval** (variable k)

### Use Regression QPP when:

- You need **per-query adaptive confidence**
- You want to **predict recall** before making decisions
- You have **rich retrieval metadata** (scores, ranks, IDs)
- You're building a **confidence-aware UI** (HIGH/MEDIUM/LOW bands)

---

## Hybrid Approach (Recommended)

Combine both approaches for maximum benefit:

```python
class HybridConfidenceEstimator:
    def __init__(self, conformal_predictor, qpp_model):
        self.conformal = conformal_predictor
        self.qpp = qpp_model

    def estimate_confidence(self, query, candidates, similarities, chunk_ids):
        # 1. Conformal: Get coverage guarantee for top-k
        _, conformal_confidence = self.conformal.top_k_with_confidence(
            similarities, chunk_ids, k=10
        )

        # 2. QPP: Get predicted recall@10
        features = extract_qpp_features(query, candidates)
        qpp_predicted_recall = self.qpp.predict(features)

        # 3. Combine (e.g., weighted average or min)
        combined_confidence = 0.5 * conformal_confidence + 0.5 * qpp_predicted_recall

        return {
            "conformal_coverage": conformal_confidence,  # Statistical guarantee
            "qpp_predicted_recall": qpp_predicted_recall,  # ML prediction
            "combined_confidence": combined_confidence,
            "interpretation": interpret(combined_confidence),
        }
```

### Benefits of Hybrid

1. **Conformal** provides floor guarantee (worst-case bound)
2. **QPP** provides refined estimate (average-case prediction)
3. **Combined** balances both perspectives

---

## Practical Recommendations

### For Your Current Setup

Given your current implementation:

1. **Keep Conformal Prediction** for:
   - Setting retrieval k (use `k_for_coverage(target)`)
   - Reporting confidence to users
   - Threshold-based retrieval when recall is critical

2. **Add QPP Regression** for:
   - Per-query difficulty estimation
   - UI confidence bands (HIGH/MEDIUM/LOW)
   - Flagging queries for human review

### Implementation Priority

| Priority | Task | Effort |
|----------|------|--------|
| ✅ Done | Conformal prediction calibration | - |
| ✅ Done | Top-k coverage analysis | - |
| 🔜 Next | QPP feature extractor | Low |
| 🔜 Next | QPP model training | Medium |
| 🔜 Later | Hybrid confidence estimator | Low |
| 🔜 Later | Production integration | Medium |

---

## Summary

| Criterion | Conformal | QPP Regression | Winner |
|-----------|-----------|----------------|--------|
| Statistical guarantees | ✅ Yes | ❌ No | Conformal |
| Per-query adaptivity | ❌ No | ✅ Yes | QPP |
| Implementation complexity | Low | Medium | Conformal |
| Feature richness | 1 feature | 6 features | QPP |
| Interpretability | High | High | Tie |
| Maintenance burden | Low | Medium | Conformal |

**Bottom line:** Use Conformal for guarantees, QPP for refinement. The hybrid approach gives you the best of both worlds.

---

*Generated by Claude Code on 2026-01-08*
