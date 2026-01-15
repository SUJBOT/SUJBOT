# RAG Confidence: Comprehensive Model Comparison Report

**Date:** 2026-01-08
**Dataset:** 1,500 synthetic queries, 5,704 chunks
**Embedding Model:** Qwen/Qwen3-Embedding-8B (4096 dims)

---

## Executive Summary

This report compares two approaches for RAG confidence estimation:

1. **Conformal Prediction (CP):** Statistical guarantees on retrieval coverage
2. **Query Performance Prediction (QPP):** ML models predicting Recall@10

| Approach | Best Method | Key Metric | Use Case |
|----------|-------------|------------|----------|
| **CP** | Threshold τ=0.711 | 90% coverage guarantee | Conservative retrieval |
| **QPP** | MLP Improved | R²=0.246, Corr=0.50 | Confidence scoring |

**Recommendation:** Use **hybrid approach** - QPP for confidence scoring + CP threshold as safety net.

---

## Part 1: Conformal Prediction (CP) Approach

### Overview

Conformal Prediction provides statistical guarantees on retrieval coverage without requiring a trained model at inference time.

### Key Results

| Parameter | Value |
|-----------|-------|
| Coverage Target | 90% |
| Calibrated Threshold (τ) | **0.7112** |
| Actual Coverage | 90.1% |
| Method | Quantile of conformity scores |

### Top-K Coverage Analysis

| k | Recall@k |
|---|----------|
| 1 | 68.2% |
| 3 | 82.9% |
| 5 | 87.3% |
| **10** | **91.5%** |
| 20 | 94.0% |

### Strengths & Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Statistical guarantees | No per-query confidence |
| No model needed at runtime | Binary (above/below threshold) |
| Simple to implement | Doesn't predict retrieval quality |
| Adapts to query difficulty | Requires calibration dataset |

---

## Part 2: Query Performance Prediction (QPP) Approach

### Overview

QPP uses 14 engineered features from similarity score distributions to predict Recall@10 via machine learning.

### Feature Set (14 Features)

| Rank | Feature | Correlation | Category |
|------|---------|-------------|----------|
| 1 | `top1_minus_p99` | +0.382 | Core (Novel) |
| 2 | `top1_vs_top10_gap` | +0.372 | Core |
| 3 | `sim_std_top10` | +0.366 | Core |
| 4 | `sim_slope` | -0.361 | Core |
| 5 | `bimodal_gap` | +0.351 | Core |
| 6 | `exp_decay_rate` | -0.342 | Extended |
| 7 | `percentile_99` | -0.333 | Extended |
| 8 | `n_above_08` | -0.324 | Extended |
| 9 | `n_above_07` | -0.307 | Extended |
| 10 | `top5_concentration` | +0.273 | Extended |
| 11 | `skewness_top50` | +0.286 | Extended |
| 12 | `max_second_deriv` | +0.256 | Extended |
| 13 | `n_above_tau` | ~0.30 | Conformal |
| 14 | `top1_margin_over_tau` | ~0.25 | Conformal |

---

## Part 3: QPP Model Comparison

### Test Set Performance

| Rank | Model | Test R² | Test Corr | CV R² | RMSE |
|------|-------|---------|-----------|-------|------|
| **1** | **MLP Improved** | **0.2456** | **0.497** | 0.278 | 0.253 |
| 2 | XGBoost | 0.2380 | 0.489 | 0.280 | 0.254 |
| 3 | Ridge Regression | 0.2071 | 0.456 | 0.236 | 0.259 |
| 4 | Lasso Regression | 0.2064 | 0.456 | - | 0.259 |
| 5 | Poly + Ridge | 0.1955 | 0.453 | - | 0.261 |
| 6 | ElasticNet | 0.1929 | 0.449 | - | 0.261 |
| 7 | MLP Original | 0.1479 | 0.467 | 0.202 | 0.269 |

### Best Model Configuration

**Winner: MLP Improved**

```
Architecture:     (32,) - single hidden layer
Regularization:   alpha=1.0 (L2)
Activation:       ReLU
Optimizer:        Adam (adaptive LR)
Early Stopping:   Yes (patience=15)
```

**Key insight:** Massive regularization (alpha=0.001 → 1.0) was the critical improvement.

### Calibration Quality (Confidence Bands)

| Band | MLP Improved | XGBoost | Ridge |
|------|--------------|---------|-------|
| HIGH (≥0.90) | 99.0% | 98.6% | 99.3% |
| MEDIUM (0.75-0.90) | 85.9% | 85.1% | 86.0% |
| LOW (0.50-0.75) | 66.0% | 60.2% | 61.8% |
| VERY_LOW (<0.50) | 32.1% | 32.8% | 34.4% |

**Interpretation:** When model predicts HIGH confidence (≥0.90), actual recall is ~99%.

---

## Part 4: CP vs QPP Comparison

### Fundamental Differences

| Aspect | Conformal Prediction | QPP Regression |
|--------|---------------------|----------------|
| **Output** | Binary (pass/fail threshold) | Continuous confidence [0,1] |
| **Guarantee** | Statistical (90% coverage) | Empirical (R²=0.25) |
| **Granularity** | Query-level threshold | Per-query score |
| **Runtime Model** | None (just threshold) | Requires ML model |
| **Interpretability** | High (single threshold) | Medium (14 features) |
| **Adaptivity** | Adapts via threshold | Adapts via features |

### When to Use Each

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Need coverage guarantee | **CP** | Statistical bound |
| Flag uncertain queries | **QPP** | Continuous confidence |
| Simple implementation | **CP** | Just threshold check |
| Rich confidence metadata | **QPP** | Feature breakdown |
| Production safety net | **Both** | Hybrid approach |

### Hybrid Approach (Recommended)

```python
def get_confidence(similarities, tau=0.711, qpp_model=None):
    """Hybrid CP + QPP confidence scoring."""

    # 1. QPP: Get continuous confidence score
    features = qpp_extractor.extract(similarities)
    qpp_confidence = qpp_model.predict(features.to_vector())

    # 2. CP: Check threshold guarantee
    n_above_tau = (similarities >= tau).sum()
    cp_pass = similarities.max() >= tau

    # 3. Combine
    return {
        "confidence": float(qpp_confidence),
        "band": classify_band(qpp_confidence),
        "cp_guarantee": cp_pass,
        "n_above_threshold": n_above_tau,
        "should_flag": qpp_confidence < 0.5 or not cp_pass,
    }
```

---

## Part 5: Feature Importance Analysis

### XGBoost Feature Importance (Gain)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `exp_decay_rate` | 22.1% |
| 2 | `sim_std_top10` | 13.9% |
| 3 | `sim_slope` | 12.5% |
| 4 | `bimodal_gap` | 8.0% |
| 5 | `top1_minus_p99` | 7.0% |

### Ridge Regression Coefficients

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | `sim_slope` | -0.382 |
| 2 | `sim_std_top10` | -0.288 |
| 3 | `exp_decay_rate` | +0.179 |
| 4 | `percentile_99` | -0.176 |
| 5 | `skewness_top50` | +0.118 |

### Key Finding

Both models agree that **score distribution shape** (slope, std, decay) matters more than **absolute thresholds** (n_above_08, etc.).

---

## Part 6: Limitations & Future Work

### Current Limitations

1. **Dataset bias:** 91.5% of queries have Recall@10 = 1.0
   - Limited variance in target variable
   - Models struggle to predict rare failures

2. **Synthetic queries:** Generated by GPT-4o from chunks
   - May not represent real user queries
   - Real queries might be more diverse/challenging

3. **Single embedding model:** Qwen3-Embedding-8B
   - Results may differ with other models
   - Threshold τ=0.711 is model-specific

### Future Improvements

1. **Collect real user queries** for validation
2. **Active learning** on queries where model is uncertain
3. **Multi-task learning** with additional signals (answer quality, user feedback)
4. **Ensemble CP + QPP** with learned combination weights

---

## Part 7: Production Recommendations

### Recommended Configuration

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Primary Model** | MLP Improved | Best test R² (0.246) |
| **Backup Model** | XGBoost | Comparable, more robust |
| **CP Threshold** | τ = 0.711 | 90% coverage guarantee |
| **Flag Threshold** | QPP < 0.5 | 32% actual recall in this band |

### Confidence Band Interpretation

| Band | Predicted Range | Expected Recall | Action |
|------|-----------------|-----------------|--------|
| HIGH | ≥ 0.90 | ~99% | Trust retrieval |
| MEDIUM | 0.75 - 0.90 | ~86% | Likely OK |
| LOW | 0.50 - 0.75 | ~66% | Review recommended |
| VERY_LOW | < 0.50 | ~32% | Flag for human review |

### API Integration

```python
from rag_confidence.confidence_regression.qpp_extractor import ImprovedQPPExtractor
import pickle

# Load models
with open("qpp_mlp_improved_model.pkl", "rb") as f:
    qpp_model = pickle.load(f)

extractor = ImprovedQPPExtractor(tau=0.711)

# At inference
def score_retrieval(similarities):
    features = extractor.extract(similarities)
    confidence = float(qpp_model.predict([features.to_vector()])[0])
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.9:
        band = "HIGH"
    elif confidence >= 0.75:
        band = "MEDIUM"
    elif confidence >= 0.5:
        band = "LOW"
    else:
        band = "VERY_LOW"

    return {
        "confidence": confidence,
        "band": band,
        "should_flag": band in ["LOW", "VERY_LOW"],
    }
```

---

## Appendix: Files Generated

| File | Description |
|------|-------------|
| `qpp_extractor.py` | 14-feature QPP extractor |
| `qpp_xgboost_model.json` | Trained XGBoost (56KB) |
| `qpp_mlp_improved_model.pkl` | Best MLP model |
| `qpp_linear_model.pkl` | Ridge regression model |
| `calibration_result.json` | CP threshold (τ=0.711) |
| `IMPROVED_QPP_FEATURES.md` | Feature documentation |

---

## Summary Table

| Approach | Method | Test R² | Guarantee | Runtime Model |
|----------|--------|---------|-----------|---------------|
| CP | Threshold | N/A | 90% coverage | No |
| QPP | MLP Improved | **0.246** | Empirical | Yes |
| QPP | XGBoost | 0.238 | Empirical | Yes |
| QPP | Ridge | 0.207 | Empirical | Yes |

**Final Recommendation:** Deploy **MLP Improved** for confidence scoring with **CP threshold** as safety net.

---

*Report generated 2026-01-08 by Claude Code*
