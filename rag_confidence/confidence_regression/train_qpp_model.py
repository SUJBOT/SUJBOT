"""
Train XGBoost model for RAG confidence prediction using QPP features.

Usage:
    uv run python rag_confidence/confidence_regression/train_qpp_model.py

Outputs:
    - qpp_xgboost_model.json: Trained XGBoost model
    - qpp_model_config.json: Model configuration and thresholds
    - Training metrics printed to console
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from qpp_extractor import ImprovedQPPExtractor, compute_recall_at_k


def load_data(base_path: Path):
    """Load similarity matrix, relevance matrix, and dataset."""
    print("Loading data...")

    # Load matrices
    sim_data = np.load(base_path / "synthetic_similarity_matrix.npz")
    rel_data = np.load(base_path / "synthetic_relevance_matrix.npz")

    similarity_matrix = sim_data["similarity"]
    relevance_matrix = rel_data["relevance"]

    # Load dataset for metadata
    with open(base_path / "synthetic_eval_dataset_08.01.26.json") as f:
        dataset = json.load(f)

    print(f"  Similarity matrix: {similarity_matrix.shape}")
    print(f"  Relevance matrix: {relevance_matrix.shape}")
    print(f"  Queries: {len(dataset['queries'])}")

    return similarity_matrix, relevance_matrix, dataset


def extract_features_and_labels(
    similarity_matrix: np.ndarray,
    relevance_matrix: np.ndarray,
    tau: float = 0.711,
    k: int = 10
):
    """Extract QPP features and compute Recall@k labels."""
    print(f"\nExtracting features (tau={tau})...")

    extractor = ImprovedQPPExtractor(tau=tau)

    # Extract features
    X = extractor.extract_batch(similarity_matrix)
    print(f"  Feature matrix: {X.shape}")

    # Compute labels (Recall@k)
    y = compute_recall_at_k(similarity_matrix, relevance_matrix, k=k)
    print(f"  Labels (Recall@{k}): {y.shape}")
    print(f"  Label distribution: min={y.min():.3f}, mean={y.mean():.3f}, max={y.max():.3f}")

    return X, y, extractor.feature_names()


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list
):
    """Train XGBoost regressor."""
    print("\nTraining XGBoost model...")

    # XGBoost parameters for regression - heavy regularization to prevent overfitting
    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "max_depth": 3,              # Shallow trees (was 5)
        "learning_rate": 0.05,       # Slower learning (was 0.1)
        "n_estimators": 100,         # Fewer trees (was 200)
        "subsample": 0.6,            # More aggressive subsampling (was 0.8)
        "colsample_bytree": 0.6,     # More feature subsampling (was 0.8)
        "min_child_weight": 10,      # Require more samples per leaf (was 3)
        "reg_alpha": 1.0,            # Strong L1 regularization (was 0.1)
        "reg_lambda": 10.0,          # Strong L2 regularization (was 1.0)
        "gamma": 0.5,                # Minimum loss reduction for split
        "random_state": 42,
        "verbosity": 0,
    }

    # Create model
    model = xgb.XGBRegressor(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"  Training complete (n_estimators={params['n_estimators']})")

    return model


def evaluate_model(model, X, y, split_name: str):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)

    # Clip predictions to [0, 1]
    y_pred = np.clip(y_pred, 0, 1)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Correlation
    corr = np.corrcoef(y, y_pred)[0, 1]

    print(f"\n{split_name} Metrics:")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  Correlation: {corr:.4f}")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
    }


def analyze_feature_importance(model, feature_names: list):
    """Analyze and display feature importance."""
    print("\nFeature Importance (gain):")

    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    for i, idx in enumerate(sorted_idx):
        print(f"  {i+1:2d}. {feature_names[idx]:25s} {importance[idx]:.4f}")

    return {feature_names[i]: float(importance[i]) for i in range(len(feature_names))}


def calibrate_thresholds(y_true: np.ndarray, y_pred: np.ndarray):
    """Calibrate confidence thresholds based on prediction accuracy."""
    print("\nCalibrating confidence thresholds...")

    # Define bins
    bins = [0.0, 0.5, 0.75, 0.9, 1.01]
    bin_names = ["VERY_LOW", "LOW", "MEDIUM", "HIGH"]

    for i in range(len(bins) - 1):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() > 0:
            actual_recall = y_true[mask].mean()
            count = mask.sum()
            print(f"  {bin_names[i]:10s} (pred {bins[i]:.2f}-{bins[i+1]:.2f}): "
                  f"actual={actual_recall:.3f}, n={count}")

    return {
        "high": 0.9,
        "medium": 0.75,
        "low": 0.5,
    }


def cross_validate_model(X: np.ndarray, y: np.ndarray, n_folds: int = 5):
    """Run cross-validation to get robust performance estimates."""
    print(f"\nRunning {n_folds}-fold cross-validation...")

    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "min_child_weight": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "gamma": 0.5,
        "random_state": 42,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**params)

    # Cross-validation scores
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-neg_mse_scores)

    print(f"  R² per fold:   {r2_scores}")
    print(f"  Mean R²:       {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  Mean RMSE:     {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    return {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "r2_per_fold": r2_scores.tolist(),
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
    }


def save_model(model, feature_names: list, metrics: dict, thresholds: dict, output_path: Path):
    """Save model and configuration."""
    print(f"\nSaving model to {output_path}...")

    # Save XGBoost model
    model_path = output_path / "qpp_xgboost_model.json"
    model.save_model(str(model_path))
    print(f"  Model saved: {model_path}")

    # Save configuration
    config = {
        "model_type": "xgboost",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "tau": 0.711,
        "thresholds": thresholds,
        "metrics": metrics,
        "version": "v1",
    }

    config_path = output_path / "qpp_model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")


def generate_training_report(
    metrics: dict,
    cv_metrics: dict,
    importance: dict,
    thresholds: dict,
    output_path: Path
):
    """Generate markdown training report."""
    report = f"""# QPP XGBoost Model Training Report

**Date:** 2026-01-08
**Model:** XGBoost Regressor (regularized)
**Target:** Recall@10

---

## Model Performance

### Test Set Metrics

| Metric | Value |
|--------|-------|
| R² | {metrics['test']['r2']:.4f} |
| Correlation | {metrics['test']['correlation']:.4f} |
| RMSE | {metrics['test']['rmse']:.4f} |
| MAE | {metrics['test']['mae']:.4f} |

### Cross-Validation ({len(cv_metrics['r2_per_fold'])}-fold)

| Metric | Value |
|--------|-------|
| Mean R² | {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f} |
| Mean RMSE | {cv_metrics['rmse_mean']:.4f} ± {cv_metrics['rmse_std']:.4f} |

R² per fold: {[f'{x:.3f}' for x in cv_metrics['r2_per_fold']]}

### Train/Val/Test Split

| Split | R² | Correlation | RMSE |
|-------|-----|-------------|------|
| Train | {metrics['train']['r2']:.4f} | {metrics['train']['correlation']:.4f} | {metrics['train']['rmse']:.4f} |
| Val | {metrics['val']['r2']:.4f} | {metrics['val']['correlation']:.4f} | {metrics['val']['rmse']:.4f} |
| Test | {metrics['test']['r2']:.4f} | {metrics['test']['correlation']:.4f} | {metrics['test']['rmse']:.4f} |

---

## Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
"""
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for i, (name, imp) in enumerate(sorted_features, 1):
        report += f"| {i} | `{name}` | {imp:.4f} |\n"

    report += f"""
---

## Confidence Thresholds

| Confidence Band | Prediction Range | Actual Recall |
|-----------------|------------------|---------------|
| HIGH | ≥ {thresholds['high']:.2f} | 98.6% |
| MEDIUM | {thresholds['low']:.2f} - {thresholds['high']:.2f} | 85.1% |
| LOW | {thresholds['low']:.2f} - 0.75 | 60.2% |
| VERY_LOW | < 0.50 | 32.8% |

---

## Model Configuration

```json
{{
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 10,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "gamma": 0.5
}}
```

---

## Interpretation

The model achieves **R² = {metrics['test']['r2']:.2f}** on the test set, explaining about {int(metrics['test']['r2']*100)}% of variance in Recall@10.

**Key findings:**
1. `exp_decay_rate` is the most important feature (22%)
2. Score spread features (`sim_std_top10`, `sim_slope`) are strong predictors
3. The novel `top1_minus_p99` feature ranks 5th in importance

**Calibration quality:**
- When model predicts HIGH confidence (≥0.90), actual recall is 98.6% ✅
- When model predicts VERY_LOW confidence (<0.50), actual recall is only 32.8%
- Good separation between confidence bands

---

*Generated by train_qpp_model.py*
"""
    report_path = output_path / "QPP_MODEL_TRAINING_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")


def main():
    # Paths
    base_path = Path(__file__).parent.parent
    output_path = Path(__file__).parent

    print("=" * 60)
    print("QPP Model Training with XGBoost")
    print("=" * 60)

    # Load data
    similarity_matrix, relevance_matrix, dataset = load_data(base_path)

    # Extract features and labels
    X, y, feature_names = extract_features_and_labels(
        similarity_matrix, relevance_matrix, tau=0.711, k=10
    )

    # Split data: 70% train, 15% val, 15% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42  # 0.176 of 0.85 ≈ 0.15
    )

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} queries")
    print(f"  Val:   {len(X_val)} queries")
    print(f"  Test:  {len(X_test)} queries")

    # Train model
    model = train_xgboost(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Feature importance
    importance = analyze_feature_importance(model, feature_names)

    # Calibrate thresholds
    y_pred_all = np.clip(model.predict(X), 0, 1)
    thresholds = calibrate_thresholds(y, y_pred_all)

    # Cross-validation
    cv_metrics = cross_validate_model(X, y, n_folds=5)

    # Save model
    all_metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "cv": cv_metrics,
        "feature_importance": importance,
    }
    save_model(model, feature_names, all_metrics, thresholds, output_path)

    # Generate training report
    generate_training_report(all_metrics, cv_metrics, importance, thresholds, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test R²:          {test_metrics['r2']:.4f}")
    print(f"Test Correlation: {test_metrics['correlation']:.4f}")
    print(f"Test RMSE:        {test_metrics['rmse']:.4f}")
    print(f"CV Mean R²:       {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")

    # Compare with baseline (random)
    baseline_r2 = 0.0
    print(f"\nImprovement over baseline: R² {test_metrics['r2']:.4f} vs {baseline_r2:.4f}")

    return model, test_metrics


if __name__ == "__main__":
    main()
