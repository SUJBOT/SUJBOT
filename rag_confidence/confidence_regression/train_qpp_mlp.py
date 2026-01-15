"""
Train MLP Neural Network for RAG confidence prediction using QPP features.

Usage:
    uv run python rag_confidence/confidence_regression/train_qpp_mlp.py

Outputs:
    - qpp_mlp_model.pkl: Trained MLP model + scaler
    - qpp_mlp_config.json: Model configuration
    - QPP_MLP_TRAINING_REPORT.md: Training report
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

from qpp_extractor import ImprovedQPPExtractor, compute_recall_at_k


def load_data(base_path: Path):
    """Load similarity matrix, relevance matrix, and dataset."""
    print("Loading data...")

    sim_data = np.load(base_path / "synthetic_similarity_matrix.npz")
    rel_data = np.load(base_path / "synthetic_relevance_matrix.npz")

    similarity_matrix = sim_data["similarity"]
    relevance_matrix = rel_data["relevance"]

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
    X = extractor.extract_batch(similarity_matrix)
    y = compute_recall_at_k(similarity_matrix, relevance_matrix, k=k)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Labels (Recall@{k}): mean={y.mean():.3f}, std={y.std():.3f}")

    return X, y, extractor.feature_names()


def create_mlp_pipeline(hidden_layers=(64, 32), alpha=0.01, learning_rate_init=0.001):
    """Create MLP pipeline with scaling."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=alpha,  # L2 regularization
            learning_rate='adaptive',
            learning_rate_init=learning_rate_init,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        ))
    ])


def grid_search_mlp(X_train: np.ndarray, y_train: np.ndarray):
    """Search for best MLP architecture."""
    print("\nGrid searching MLP architectures...")

    # Define parameter grid
    param_grid = {
        'mlp__hidden_layer_sizes': [
            (32,),           # Shallow
            (64,),           # Medium shallow
            (64, 32),        # Medium deep
            (128, 64),       # Deeper
            (64, 32, 16),    # 3 layers
        ],
        'mlp__alpha': [0.001, 0.01, 0.1],  # Regularization
        'mlp__learning_rate_init': [0.001, 0.01],
    }

    pipeline = create_mlp_pipeline()

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV R²: {grid_search.best_score_:.4f}")

    return grid_search.best_params_, grid_search.best_score_


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layers: tuple = (64, 32),
    alpha: float = 0.01,
    learning_rate_init: float = 0.001,
):
    """Train MLP with specified architecture."""
    print(f"\nTraining MLP: layers={hidden_layers}, alpha={alpha}, lr={learning_rate_init}")

    pipeline = create_mlp_pipeline(hidden_layers, alpha, learning_rate_init)
    pipeline.fit(X_train, y_train)

    mlp = pipeline.named_steps['mlp']
    print(f"  Converged: {mlp.n_iter_} iterations")
    print(f"  Final loss: {mlp.loss_:.6f}")

    return pipeline


def evaluate_model(model, X, y, split_name: str):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, 1)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    corr = np.corrcoef(y, y_pred)[0, 1]

    print(f"\n{split_name} Metrics:")
    print(f"  RMSE:        {rmse:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  Correlation: {corr:.4f}")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "correlation": float(corr),
    }


def cross_validate_mlp(X: np.ndarray, y: np.ndarray, best_params: dict, n_folds: int = 5):
    """Run cross-validation with best parameters."""
    print(f"\nRunning {n_folds}-fold cross-validation...")

    hidden_layers = best_params.get('mlp__hidden_layer_sizes', (64, 32))
    alpha = best_params.get('mlp__alpha', 0.01)
    lr = best_params.get('mlp__learning_rate_init', 0.001)

    pipeline = create_mlp_pipeline(hidden_layers, alpha, lr)

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    neg_mse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error")
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


def calibrate_thresholds(y_true: np.ndarray, y_pred: np.ndarray):
    """Calibrate confidence thresholds."""
    print("\nCalibrating confidence thresholds...")

    bins = [0.0, 0.5, 0.75, 0.9, 1.01]
    bin_names = ["VERY_LOW", "LOW", "MEDIUM", "HIGH"]
    calibration = {}

    for i in range(len(bins) - 1):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i+1])
        if mask.sum() > 0:
            actual_recall = y_true[mask].mean()
            count = mask.sum()
            print(f"  {bin_names[i]:10s} (pred {bins[i]:.2f}-{bins[i+1]:.2f}): "
                  f"actual={actual_recall:.3f}, n={count}")
            calibration[bin_names[i].lower()] = {
                "actual_recall": float(actual_recall),
                "count": int(count)
            }

    return {
        "high": 0.9,
        "medium": 0.75,
        "low": 0.5,
        "calibration_stats": calibration,
    }


def save_model(model, feature_names: list, metrics: dict, best_params: dict,
               thresholds: dict, output_path: Path):
    """Save model and configuration."""
    print(f"\nSaving model to {output_path}...")

    # Save pipeline (includes scaler + MLP)
    model_path = output_path / "qpp_mlp_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved: {model_path}")

    # Save configuration
    config = {
        "model_type": "mlp",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "tau": 0.711,
        "best_params": {k.replace('mlp__', ''): v for k, v in best_params.items()},
        "thresholds": thresholds,
        "metrics": metrics,
        "version": "v1",
    }

    config_path = output_path / "qpp_mlp_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved: {config_path}")


def generate_training_report(
    metrics: dict,
    cv_metrics: dict,
    best_params: dict,
    thresholds: dict,
    xgb_comparison: dict,
    output_path: Path
):
    """Generate markdown training report."""

    hidden_layers = best_params.get('mlp__hidden_layer_sizes', (64, 32))
    alpha = best_params.get('mlp__alpha', 0.01)
    lr = best_params.get('mlp__learning_rate_init', 0.001)

    report = f"""# QPP MLP Neural Network Training Report

**Date:** 2026-01-08
**Model:** Multi-Layer Perceptron (sklearn)
**Target:** Recall@10

---

## Model Performance

### Test Set Metrics

| Metric | MLP | XGBoost | Winner |
|--------|-----|---------|--------|
| R² | {metrics['test']['r2']:.4f} | {xgb_comparison['r2']:.4f} | {'MLP' if metrics['test']['r2'] > xgb_comparison['r2'] else 'XGBoost'} |
| Correlation | {metrics['test']['correlation']:.4f} | {xgb_comparison['correlation']:.4f} | {'MLP' if metrics['test']['correlation'] > xgb_comparison['correlation'] else 'XGBoost'} |
| RMSE | {metrics['test']['rmse']:.4f} | {xgb_comparison['rmse']:.4f} | {'MLP' if metrics['test']['rmse'] < xgb_comparison['rmse'] else 'XGBoost'} |
| MAE | {metrics['test']['mae']:.4f} | {xgb_comparison['mae']:.4f} | {'MLP' if metrics['test']['mae'] < xgb_comparison['mae'] else 'XGBoost'} |

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

## Best Architecture (Grid Search)

| Parameter | Value |
|-----------|-------|
| Hidden Layers | {hidden_layers} |
| L2 Regularization (alpha) | {alpha} |
| Learning Rate | {lr} |
| Activation | ReLU |
| Optimizer | Adam (adaptive) |
| Early Stopping | Yes (patience=20) |

---

## Confidence Thresholds

| Confidence Band | Prediction Range | Actual Recall | Count |
|-----------------|------------------|---------------|-------|
"""

    calib = thresholds.get('calibration_stats', {})
    for band in ['high', 'medium', 'low', 'very_low']:
        if band in calib:
            stats = calib[band]
            pred_range = {
                'high': '≥ 0.90',
                'medium': '0.75 - 0.90',
                'low': '0.50 - 0.75',
                'very_low': '< 0.50'
            }[band]
            report += f"| {band.upper()} | {pred_range} | {stats['actual_recall']*100:.1f}% | {stats['count']} |\n"

    report += f"""
---

## MLP vs XGBoost Comparison

| Aspect | MLP | XGBoost |
|--------|-----|---------|
| Test R² | {metrics['test']['r2']:.4f} | {xgb_comparison['r2']:.4f} |
| CV Mean R² | {cv_metrics['r2_mean']:.4f} | 0.2797 |
| Overfitting Gap | {metrics['train']['r2'] - metrics['test']['r2']:.4f} | 0.1608 |
| Model Size | ~50KB | 56KB |
| Inference Speed | Fast | Fast |

**Verdict:** {'MLP performs better' if metrics['test']['r2'] > xgb_comparison['r2'] else 'XGBoost performs better' if metrics['test']['r2'] < xgb_comparison['r2'] else 'Both perform similarly'}

---

## Interpretation

The MLP achieves **R² = {metrics['test']['r2']:.2f}** on the test set.

**Key observations:**
1. Architecture {hidden_layers} was selected via grid search
2. L2 regularization (alpha={alpha}) helps prevent overfitting
3. Early stopping prevents overtraining

**Practical use:**
- Model provides confidence scores that correlate {metrics['test']['correlation']:.1%} with actual recall
- Good separation between HIGH ({calib.get('high', {}).get('actual_recall', 0)*100:.0f}%) and VERY_LOW ({calib.get('very_low', {}).get('actual_recall', 0)*100:.0f}%) confidence

---

*Generated by train_qpp_mlp.py*
"""
    report_path = output_path / "QPP_MLP_TRAINING_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")


def main():
    # Paths
    base_path = Path(__file__).parent.parent
    output_path = Path(__file__).parent

    print("=" * 60)
    print("QPP Model Training with MLP Neural Network")
    print("=" * 60)

    # Load data
    similarity_matrix, relevance_matrix, dataset = load_data(base_path)

    # Extract features and labels
    X, y, feature_names = extract_features_and_labels(
        similarity_matrix, relevance_matrix, tau=0.711, k=10
    )

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42
    )

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} queries")
    print(f"  Val:   {len(X_val)} queries")
    print(f"  Test:  {len(X_test)} queries")

    # Grid search for best architecture
    best_params, best_cv_score = grid_search_mlp(X_train, y_train)

    # Train final model with best params
    hidden_layers = best_params.get('mlp__hidden_layer_sizes', (64, 32))
    alpha = best_params.get('mlp__alpha', 0.01)
    lr = best_params.get('mlp__learning_rate_init', 0.001)

    model = train_mlp(X_train, y_train, hidden_layers, alpha, lr)

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Cross-validation
    cv_metrics = cross_validate_mlp(X, y, best_params, n_folds=5)

    # Calibrate thresholds
    y_pred_all = np.clip(model.predict(X), 0, 1)
    thresholds = calibrate_thresholds(y, y_pred_all)

    # XGBoost comparison (from previous training)
    xgb_comparison = {
        "r2": 0.2380,
        "correlation": 0.4891,
        "rmse": 0.2539,
        "mae": 0.1209,
    }

    # Save model
    all_metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "cv": cv_metrics,
    }
    save_model(model, feature_names, all_metrics, best_params, thresholds, output_path)

    # Generate report
    generate_training_report(all_metrics, cv_metrics, best_params, thresholds,
                            xgb_comparison, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test R²:          {test_metrics['r2']:.4f}")
    print(f"Test Correlation: {test_metrics['correlation']:.4f}")
    print(f"Test RMSE:        {test_metrics['rmse']:.4f}")
    print(f"CV Mean R²:       {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")

    # Compare with XGBoost
    print(f"\n--- Comparison with XGBoost ---")
    print(f"MLP Test R²:     {test_metrics['r2']:.4f}")
    print(f"XGBoost Test R²: {xgb_comparison['r2']:.4f}")
    diff = test_metrics['r2'] - xgb_comparison['r2']
    print(f"Difference:      {diff:+.4f} ({'MLP better' if diff > 0 else 'XGBoost better'})")

    return model, test_metrics


if __name__ == "__main__":
    main()
