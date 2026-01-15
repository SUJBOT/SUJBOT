# Improved QPP Features for RAG Confidence

**QPP = Query Performance Prediction** - predicting retrieval quality without relevance labels.

This document proposes improved features based on empirical analysis of our 1500-query calibration dataset.

---

## Key Finding: Relative Features Beat Absolute Features

| Feature Type | Example | Correlation | Verdict |
|--------------|---------|-------------|---------|
| **Relative** | top1 - top10 gap | +0.372 | ✅ Strong |
| **Absolute** | top1 similarity | -0.002 | ❌ Useless |

**Insight:** It's not about *how high* the top score is, but *how much it stands out*.

---

## Feature Ranking (by correlation with Recall@10)

### Tier 1: Strong Predictors (|r| > 0.35)

| Rank | Feature | Correlation | Description |
|------|---------|-------------|-------------|
| 1 | `top1_vs_top10_gap` | +0.372 | sim[0] - sim[9] |
| 2 | `sim_std_top10` | +0.366 | std(sim[0:10]) |
| 3 | `sim_slope` | -0.361 | Linear slope of top-10 scores |
| 4 | `bimodal_gap` | +0.351 | mean(top3) - mean(positions 10-30) |

### Tier 2: Moderate Predictors (0.25 < |r| < 0.35)

| Rank | Feature | Correlation | Description |
|------|---------|-------------|-------------|
| 5 | `exp_decay_rate` | -0.342 | Exponential decay rate of scores |
| 6 | `n_above_0.8` | -0.324 | Count of chunks with sim ≥ 0.8 |
| 7 | `n_above_0.7` | -0.307 | Count of chunks with sim ≥ 0.7 |
| 8 | `top5_concentration` | +0.273 | sum(top5) / sum(top100) |

### Tier 3: Weak Predictors (|r| < 0.25)

| Rank | Feature | Correlation | Description |
|------|---------|-------------|-------------|
| 9 | `clarity_score` | +0.165 | (top1 - rest_mean) / rest_std |
| 10 | `avg_corpus_sim` | -0.161 | Mean similarity to all chunks |
| 11 | `gap_ratio` | +0.143 | gap(1-5) / gap(5-10) |

### Tier 4: Non-Predictive (DO NOT USE)

| Feature | Correlation | Why It Fails |
|---------|-------------|--------------|
| `top1_sim` | -0.002 | Absolute value doesn't matter |
| `max_sim` | -0.002 | Same as top1_sim |
| `query_len` | +0.031 | Query length barely matters |
| `top10_mean` | -0.246 | Counter-intuitive, negative correlation |

---

## Novel Regressors (Extended Analysis)

Additional features discovered through empirical correlation analysis on the 1500-query dataset.

### New Tier 1: Strong Predictors (|r| > 0.35)

| Rank | Feature | Correlation | Description |
|------|---------|-------------|-------------|
| **1** | `top1_minus_p99` | **+0.382** | sim[0] - percentile_99(all_sims) |

This is our **best feature overall** - even stronger than `top1_vs_top10_gap`!

### New Tier 2: Moderate Predictors (0.25 < |r| < 0.35)

| Rank | Feature | Correlation | Description |
|------|---------|-------------|-------------|
| 2 | `percentile_99` | -0.333 | 99th percentile of all similarities |
| 3 | `skewness_top50` | +0.286 | Skewness of top-50 score distribution |
| 4 | `score_at_100` | -0.280 | Similarity score at position 100 |
| 5 | `max_second_deriv` | +0.256 | Maximum 2nd derivative (sharpest elbow) |

### Why These Novel Features Work

#### 1. `top1_minus_p99` (r = +0.382) - **BEST FEATURE**

**Intuition:** Measures how much the top result stands out from the corpus "background noise".

```python
def top1_minus_p99(sorted_sims: np.ndarray) -> float:
    """Gap between top-1 and 99th percentile of corpus."""
    p99 = np.percentile(sorted_sims, 99)  # Or sorted_sims[int(0.01 * len(sorted_sims))]
    return sorted_sims[0] - p99
```

```
Good retrieval:  top1=0.85, p99=0.52 → gap = 0.33 ✅
Bad retrieval:   top1=0.71, p99=0.65 → gap = 0.06 ❌
```

**Why it beats `top1_vs_top10_gap`:** Uses corpus-level context, not just top-10.

#### 2. `percentile_99` (r = -0.333)

**Intuition:** High p99 means many chunks are similar to the query → ambiguous/generic query.

```python
def percentile_99(sorted_sims: np.ndarray) -> float:
    """99th percentile of similarity distribution."""
    return np.percentile(sorted_sims, 99)
```

**Why negative correlation?** If even the 99th percentile is high (e.g., 0.65+), the query matches too many chunks → retrieval uncertainty.

#### 3. `skewness_top50` (r = +0.286)

**Intuition:** Positive skew = heavy right tail = few standout results = good retrieval.

```python
from scipy.stats import skew

def skewness_top50(sorted_sims: np.ndarray) -> float:
    """Skewness of top-50 score distribution."""
    return skew(sorted_sims[:50])
```

```
Good retrieval: skew = +1.2 (few high scores, many low) ✅
Bad retrieval:  skew = +0.3 (uniform-ish distribution) ❌
```

#### 4. `score_at_100` (r = -0.280)

**Intuition:** If the 100th result still has high similarity, the query is too generic.

```python
def score_at_100(sorted_sims: np.ndarray) -> float:
    """Score at rank 100 (tail indicator)."""
    return sorted_sims[99] if len(sorted_sims) >= 100 else sorted_sims[-1]
```

#### 5. `max_second_deriv` (r = +0.256)

**Intuition:** Detects the "elbow" in the score curve. Sharp elbow = clear cutoff between relevant and irrelevant.

```python
def max_second_deriv(sorted_sims: np.ndarray, k: int = 30) -> float:
    """Maximum second derivative (elbow sharpness)."""
    second_deriv = np.diff(sorted_sims[:k], n=2)
    return np.max(second_deriv)  # Most positive = sharpest elbow
```

```
Good retrieval: sharp elbow at position 3 → max_2nd_deriv = 0.08 ✅
Bad retrieval:  smooth decay → max_2nd_deriv = 0.01 ❌
```

### Features That DON'T Work (Tested & Rejected)

| Feature | Correlation | Why It Failed |
|---------|-------------|---------------|
| `query_type` (categorical) | < 0.05 | Question type doesn't predict difficulty |
| `query_word_count` | +0.031 | Same as query_len, not predictive |
| `same_doc_top2` (binary) | < 0.05 | Document clustering doesn't help |
| `kurtosis_top50` | ~0.10 | Weaker than skewness |
| `gini_coefficient` | ~0.15 | Weaker than std/concentration |

---

## Proposed Feature Set (Recommended 12 Features)

### Core Features (use all 5) - Updated with Novel Discovery

```python
def compute_core_features(sorted_sims: np.ndarray) -> dict:
    """Compute top 5 most predictive features."""
    return {
        # 1. NEW BEST: Gap from corpus background noise
        "top1_minus_p99": sorted_sims[0] - np.percentile(sorted_sims, 99),

        # 2. Gap between best and 10th result
        "top1_vs_top10_gap": sorted_sims[0] - sorted_sims[9],

        # 3. Score spread in top-10
        "sim_std_top10": np.std(sorted_sims[:10]),

        # 4. Score decay (steeper = better)
        "sim_slope": np.polyfit(np.arange(10), sorted_sims[:10], 1)[0],

        # 5. Bimodal separation (top cluster vs rest)
        "bimodal_gap": np.mean(sorted_sims[:3]) - np.mean(sorted_sims[10:30]),
    }
```

### Extended Features (optional, add for boost)

```python
from scipy.stats import skew

def compute_extended_features(sorted_sims: np.ndarray) -> dict:
    """Compute additional predictive features."""
    return {
        # 6. Exponential decay rate
        "exp_decay_rate": np.polyfit(
            np.arange(10),
            np.log(sorted_sims[:10] + 1e-8),
            1
        )[0],

        # 7-8. Threshold counts (calibrated from conformal analysis!)
        "n_above_0.8": (sorted_sims >= 0.8).sum(),
        "n_above_0.7": (sorted_sims >= 0.7).sum(),  # Our conformal threshold!

        # 9. Score concentration in top-5
        "top5_concentration": sorted_sims[:5].sum() / (sorted_sims[:100].sum() + 1e-8),

        # 10. NEW: 99th percentile (query genericity indicator)
        "percentile_99": np.percentile(sorted_sims, 99),

        # 11. NEW: Distribution skewness
        "skewness_top50": skew(sorted_sims[:50]),

        # 12. NEW: Elbow sharpness
        "max_second_deriv": np.max(np.diff(sorted_sims[:30], n=2)),
    }
```

---

## Why These Features Work

### 1. `top1_vs_top10_gap` (r = +0.372)

**Intuition:** When there's a clear winner, retrieval is confident.

```
Good retrieval:  [0.85, 0.72, 0.68, 0.65, ...] → gap = 0.20 ✅
Bad retrieval:   [0.71, 0.70, 0.69, 0.68, ...] → gap = 0.03 ❌
```

### 2. `sim_std_top10` (r = +0.366)

**Intuition:** High variance = differentiated results = confident ranking.

```
Good: std = 0.08 (scores spread out)
Bad:  std = 0.02 (scores all similar)
```

### 3. `sim_slope` (r = -0.361)

**Intuition:** Steeper negative slope = fast score decay = clear winner.

```
Good: slope = -0.025 (steep decay)
Bad:  slope = -0.005 (flat, uncertain)
```

### 4. `bimodal_gap` (r = +0.351)

**Intuition:** Good retrieval has a "cluster" of relevant results separated from noise.

```
Good: top3_mean=0.82, rest_mean=0.45 → gap = 0.37 ✅
Bad:  top3_mean=0.68, rest_mean=0.58 → gap = 0.10 ❌
```

### 5. `n_above_0.7` (r = -0.307)

**Why negative?** Counter-intuitive but makes sense:
- If MANY chunks score > 0.7, the query is probably generic
- Generic queries → harder to find THE relevant chunk

```
Specific query: "Jaké jsou limity pro alfa zářiče?" → 3 chunks > 0.7 ✅
Generic query:  "Jaké jsou požadavky?" → 200 chunks > 0.7 ❌
```

---

## Integration with Conformal Prediction

### Using Conformal Threshold as Feature

Our conformal calibration found τ = 0.711. Use this as a threshold feature:

```python
def compute_conformal_features(sorted_sims: np.ndarray, tau: float = 0.711) -> dict:
    """Features based on conformal threshold."""
    return {
        # How many results are above conformal threshold
        "n_above_tau": (sorted_sims >= tau).sum(),

        # Is top-1 above threshold? (binary)
        "top1_above_tau": int(sorted_sims[0] >= tau),

        # Margin above threshold
        "top1_margin_over_tau": max(0, sorted_sims[0] - tau),
    }
```

---

## Complete Feature Extractor (v2 with Novel Features)

```python
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.stats import skew

@dataclass
class QPPFeatures:
    # Core (Tier 1) - 5 features
    top1_minus_p99: float       # NEW: Best feature (r=+0.382)
    top1_vs_top10_gap: float
    sim_std_top10: float
    sim_slope: float
    bimodal_gap: float

    # Extended (Tier 2) - 7 features
    exp_decay_rate: float
    n_above_08: int
    n_above_07: int
    top5_concentration: float
    percentile_99: float        # NEW (r=-0.333)
    skewness_top50: float       # NEW (r=+0.286)
    max_second_deriv: float     # NEW (r=+0.256)

    # Conformal-informed - 2 features
    n_above_tau: int
    top1_margin_over_tau: float


class ImprovedQPPExtractor:
    """Extract QPP features for RAG confidence prediction.

    Total: 14 features (5 core + 7 extended + 2 conformal)
    """

    def __init__(self, tau: float = 0.711, k: int = 10):
        self.tau = tau  # Conformal threshold
        self.k = k

    def extract(self, similarities: np.ndarray) -> QPPFeatures:
        """Extract QPP features from similarity scores."""
        sorted_sims = np.sort(similarities)[::-1]  # Descending
        p99 = np.percentile(sorted_sims, 99)

        return QPPFeatures(
            # Core (strongest predictors)
            top1_minus_p99=sorted_sims[0] - p99,
            top1_vs_top10_gap=sorted_sims[0] - sorted_sims[9],
            sim_std_top10=np.std(sorted_sims[:10]),
            sim_slope=np.polyfit(np.arange(10), sorted_sims[:10], 1)[0],
            bimodal_gap=np.mean(sorted_sims[:3]) - np.mean(sorted_sims[10:30]),

            # Extended
            exp_decay_rate=np.polyfit(np.arange(10), np.log(sorted_sims[:10] + 1e-8), 1)[0],
            n_above_08=int((sorted_sims >= 0.8).sum()),
            n_above_07=int((sorted_sims >= 0.7).sum()),
            top5_concentration=sorted_sims[:5].sum() / (sorted_sims[:100].sum() + 1e-8),
            percentile_99=p99,
            skewness_top50=skew(sorted_sims[:50]),
            max_second_deriv=float(np.max(np.diff(sorted_sims[:30], n=2))),

            # Conformal
            n_above_tau=int((sorted_sims >= self.tau).sum()),
            top1_margin_over_tau=max(0, sorted_sims[0] - self.tau),
        )

    def to_vector(self, features: QPPFeatures) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array([
            features.top1_minus_p99,
            features.top1_vs_top10_gap,
            features.sim_std_top10,
            features.sim_slope,
            features.bimodal_gap,
            features.exp_decay_rate,
            features.n_above_08,
            features.n_above_07,
            features.top5_concentration,
            features.percentile_99,
            features.skewness_top50,
            features.max_second_deriv,
            features.n_above_tau,
            features.top1_margin_over_tau,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered feature names for model coefficients."""
        return [
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
```

---

## Comparison: Original vs Improved Features

| Original Feature | Correlation | Improved Alternative | Correlation | Δ |
|------------------|-------------|----------------------|-------------|---|
| `std_norm_top10` | +0.366 | `sim_std_top10` | +0.366 | = |
| `entropy_norm_top10` | -0.211 | `top5_concentration` | +0.273 | +29% |
| `slope_norm_top10` | -0.361 | `sim_slope` | -0.361 | = |
| `top_vs_rest_ratio` | ~0.30 | `bimodal_gap` | +0.351 | +17% |
| `section_diversity` | ~0.10 | **DROP** | - | - |
| `query_token_len` | +0.031 | **DROP** | - | - |
| - | - | `top1_minus_p99` | **+0.382** | **NEW BEST** |
| - | - | `top1_vs_top10_gap` | +0.372 | **NEW** |
| - | - | `percentile_99` | -0.333 | **NEW** |
| - | - | `skewness_top50` | +0.286 | **NEW** |
| - | - | `max_second_deriv` | +0.256 | **NEW** |
| - | - | `n_above_tau` | ~0.30 | **NEW** |

### Summary of Changes

1. **Keep:** `std_top10`, `slope`
2. **Replace:** entropy → concentration, top_vs_rest → bimodal_gap
3. **Drop:** section_diversity, query_length (weak predictors)
4. **Add (original):** gap features, threshold counts, conformal-informed features
5. **Add (novel):** `top1_minus_p99`, `percentile_99`, `skewness_top50`, `max_second_deriv`

---

## Expected Improvement

| Metric | Original 6 Features | Improved 14 Features |
|--------|---------------------|----------------------|
| Avg |r| | ~0.22 | ~0.33 |
| Max |r| | 0.37 | **0.382** |
| Features > 0.30 | 2 | **7** |
| Features > 0.25 | 3 | **10** |

**Estimated R² improvement: +20-35%**

---

## Final Feature Ranking (All 14 Features)

| Rank | Feature | |r| | Category |
|------|---------|-----|----------|
| 1 | `top1_minus_p99` | 0.382 | Core (Novel) |
| 2 | `top1_vs_top10_gap` | 0.372 | Core |
| 3 | `sim_std_top10` | 0.366 | Core |
| 4 | `sim_slope` | 0.361 | Core |
| 5 | `bimodal_gap` | 0.351 | Core |
| 6 | `exp_decay_rate` | 0.342 | Extended |
| 7 | `percentile_99` | 0.333 | Extended (Novel) |
| 8 | `n_above_0.8` | 0.324 | Extended |
| 9 | `n_above_0.7` | 0.307 | Extended |
| 10 | `n_above_tau` | ~0.30 | Conformal |
| 11 | `skewness_top50` | 0.286 | Extended (Novel) |
| 12 | `top5_concentration` | 0.273 | Extended |
| 13 | `max_second_deriv` | 0.256 | Extended (Novel) |
| 14 | `top1_margin_over_tau` | ~0.25 | Conformal |

---

## Next Steps

1. ✅ Design `ImprovedQPPExtractor` with 14 features
2. Train Ridge/ElasticNet on improved features
3. Compare with original 6-feature model
4. Validate on held-out test set
5. Consider feature selection (Lasso) to find minimal effective subset

---

*Analysis based on 1500 synthetic queries, 2026-01-08*
*Updated with novel regressors from extended correlation analysis*
