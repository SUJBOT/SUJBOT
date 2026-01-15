"""
Improved MLP Neural Network for RAG confidence prediction.

Improvements over baseline MLP:
1. Polynomial features for capturing interactions
2. Stronger regularization (higher alpha)
3. Simpler architectures to reduce overfitting
4. Ensemble of MLPs for variance reduction
5. Feature selection based on importance

Usage:
    uv run python rag_confidence/confidence_regression/train_qpp_mlp_improved.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')

from qpp_extractor import ImprovedQPPExtractor, compute_recall_at_k


class MLPEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble of MLPs with different random seeds."""

    def __init__(self, n_estimators=5, hidden_layer_sizes=(32,), alpha=0.1,
                 learning_rate_init=0.01, max_iter=500):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.models_ = []

    def fit(self, X, y):
        self.models_ = []
        for i in range(self.n_estimators):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=self.alpha,
                learning_rate='adaptive',
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42 + i,  # Different seed for each
                verbose=False,
            )
            mlp.fit(X, y)
            self.models_.append(mlp)
        return self

    def predict(self, X):
        predictions = np.array([m.predict(X) for m in self.models_])
        return predictions.mean(axis=0)


def load_data(base_path: Path):
    """Load data."""
    print("Loading data...")
    sim_data = np.load(base_path / "synthetic_similarity_matrix.npz")
    rel_data = np.load(base_path / "synthetic_relevance_matrix.npz")
    return sim_data["similarity"], rel_data["relevance"]


def extract_features_and_labels(similarity_matrix, relevance_matrix, tau=0.711, k=10):
    """Extract features and labels."""
    print(f"\nExtracting features...")
    extractor = ImprovedQPPExtractor(tau=tau)
    X = extractor.extract_batch(similarity_matrix)
    y = compute_recall_at_k(similarity_matrix, relevance_matrix, k=k)
    print(f"  Features: {X.shape}, Labels: mean={y.mean():.3f}")
    return X, y, extractor.feature_names()


def evaluate_model(model, X, y, split_name):
    """Evaluate and return metrics."""
    y_pred = np.clip(model.predict(X), 0, 1)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    corr = np.corrcoef(y, y_pred)[0, 1]
    print(f"{split_name}: R²={r2:.4f}, Corr={corr:.4f}, RMSE={rmse:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae, "correlation": corr}


def calibrate_thresholds(y_true, y_pred):
    """Calculate calibration stats."""
    bins = [(0.0, 0.5, "VERY_LOW"), (0.5, 0.75, "LOW"),
            (0.75, 0.9, "MEDIUM"), (0.9, 1.01, "HIGH")]
    stats = {}
    for low, high, name in bins:
        mask = (y_pred >= low) & (y_pred < high)
        if mask.sum() > 0:
            stats[name.lower()] = {
                "actual_recall": float(y_true[mask].mean()),
                "count": int(mask.sum())
            }
            print(f"  {name:10s}: actual={y_true[mask].mean():.3f}, n={mask.sum()}")
    return stats


def experiment_1_polynomial_features(X_train, y_train, X_val, y_val, X_test, y_test):
    """Experiment 1: Add polynomial feature interactions."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Polynomial Features (degree=2)")
    print("="*60)

    # Create pipeline with polynomial features
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(64, 32),
            alpha=0.1,  # Strong regularization
            learning_rate_init=0.01,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        ))
    ])

    pipeline.fit(X_train, y_train)

    n_poly_features = pipeline.named_steps['poly'].n_output_features_
    print(f"  Polynomial features: {X_train.shape[1]} → {n_poly_features}")

    train_metrics = evaluate_model(pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(pipeline, X_test, y_test, "  Test")

    return pipeline, test_metrics


def experiment_2_strong_regularization(X_train, y_train, X_val, y_val, X_test, y_test):
    """Experiment 2: Very strong regularization with simple architecture."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Strong Regularization + Simple Architecture")
    print("="*60)

    # Try different alpha values
    best_val_r2 = -np.inf
    best_config = None
    best_pipeline = None

    configs = [
        {"hidden": (16,), "alpha": 1.0},
        {"hidden": (32,), "alpha": 0.5},
        {"hidden": (32,), "alpha": 1.0},
        {"hidden": (16, 8), "alpha": 0.5},
        {"hidden": (32, 16), "alpha": 0.5},
    ]

    for cfg in configs:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=cfg["hidden"],
                alpha=cfg["alpha"],
                learning_rate_init=0.01,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42,
                verbose=False,
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred_val = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        print(f"  hidden={cfg['hidden']}, alpha={cfg['alpha']}: Val R²={val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_config = cfg
            best_pipeline = pipeline

    print(f"\n  Best config: {best_config}")
    train_metrics = evaluate_model(best_pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "  Test")

    return best_pipeline, test_metrics, best_config


def experiment_3_feature_selection(X_train, y_train, X_val, y_val, X_test, y_test):
    """Experiment 3: Select top-k most important features."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Feature Selection (top-k features)")
    print("="*60)

    best_val_r2 = -np.inf
    best_k = None
    best_pipeline = None

    for k in [5, 7, 10, 12]:
        pipeline = Pipeline([
            ('select', SelectKBest(f_regression, k=k)),
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(32,),
                alpha=0.5,
                learning_rate_init=0.01,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42,
                verbose=False,
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred_val = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        print(f"  k={k} features: Val R²={val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_k = k
            best_pipeline = pipeline

    print(f"\n  Best k: {best_k}")
    train_metrics = evaluate_model(best_pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "  Test")

    # Show selected features
    selector = best_pipeline.named_steps['select']
    feature_names = ImprovedQPPExtractor.feature_names()
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
    print(f"  Selected features: {selected_features}")

    return best_pipeline, test_metrics, best_k


def experiment_4_ensemble(X_train, y_train, X_val, y_val, X_test, y_test):
    """Experiment 4: Ensemble of MLPs."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: MLP Ensemble (5 models)")
    print("="*60)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', MLPEnsemble(
            n_estimators=5,
            hidden_layer_sizes=(32,),
            alpha=0.5,
            learning_rate_init=0.01,
            max_iter=500,
        ))
    ])

    pipeline.fit(X_train, y_train)
    print(f"  Trained {pipeline.named_steps['ensemble'].n_estimators} MLPs")

    train_metrics = evaluate_model(pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(pipeline, X_test, y_test, "  Test")

    return pipeline, test_metrics


def experiment_5_combined_best(X_train, y_train, X_val, y_val, X_test, y_test,
                                best_config, best_k):
    """Experiment 5: Combine best approaches."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Combined Best (Poly + Selection + Ensemble)")
    print("="*60)

    # Combine polynomial features, feature selection, and ensemble
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('select', SelectKBest(f_regression, k=min(50, best_k * 3))),  # More features after poly
        ('scaler', StandardScaler()),
        ('ensemble', MLPEnsemble(
            n_estimators=7,
            hidden_layer_sizes=best_config.get("hidden", (32,)),
            alpha=best_config.get("alpha", 0.5),
            learning_rate_init=0.01,
            max_iter=500,
        ))
    ])

    pipeline.fit(X_train, y_train)

    n_poly = pipeline.named_steps['poly'].n_output_features_
    print(f"  Poly features: {n_poly}, Selected: {min(50, best_k * 3)}")

    train_metrics = evaluate_model(pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(pipeline, X_test, y_test, "  Test")

    return pipeline, test_metrics


def cross_validate_best(X, y, best_pipeline_factory, n_folds=5):
    """Cross-validate the best model."""
    print(f"\nCross-validating best model ({n_folds} folds)...")

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = best_pipeline_factory()
        pipeline.fit(X_train, y_train)

        y_pred = np.clip(pipeline.predict(X_val), 0, 1)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

    r2_scores = np.array(r2_scores)
    rmse_scores = np.array(rmse_scores)

    print(f"  R² per fold: {r2_scores}")
    print(f"  Mean R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  Mean RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    return {
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "r2_per_fold": r2_scores.tolist(),
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
    }


def main():
    base_path = Path(__file__).parent.parent
    output_path = Path(__file__).parent

    print("=" * 60)
    print("IMPROVED MLP TRAINING")
    print("=" * 60)

    # Load and prepare data
    similarity_matrix, relevance_matrix = load_data(base_path)
    X, y, feature_names = extract_features_and_labels(similarity_matrix, relevance_matrix)

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42
    )

    print(f"\nSplits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Run experiments
    results = {}

    # Experiment 1: Polynomial features
    _, results["poly"] = experiment_1_polynomial_features(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 2: Strong regularization
    _, results["reg"], best_config = experiment_2_strong_regularization(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 3: Feature selection
    _, results["select"], best_k = experiment_3_feature_selection(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 4: Ensemble
    _, results["ensemble"] = experiment_4_ensemble(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 5: Combined best
    best_pipeline, results["combined"] = experiment_5_combined_best(
        X_train, y_train, X_val, y_val, X_test, y_test, best_config, best_k
    )

    # Find best experiment
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    baseline_xgb = 0.2380
    baseline_mlp = 0.1479

    print(f"\n{'Experiment':<20} {'Test R²':>10} {'vs XGBoost':>12} {'vs MLP':>10}")
    print("-" * 55)

    best_exp = None
    best_r2 = -np.inf

    for exp_name, metrics in results.items():
        r2 = metrics["r2"]
        diff_xgb = r2 - baseline_xgb
        diff_mlp = r2 - baseline_mlp

        print(f"{exp_name:<20} {r2:>10.4f} {diff_xgb:>+12.4f} {diff_mlp:>+10.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_exp = exp_name

    print("-" * 55)
    print(f"{'XGBoost baseline':<20} {baseline_xgb:>10.4f}")
    print(f"{'MLP baseline':<20} {baseline_mlp:>10.4f}")
    print(f"\nBest experiment: {best_exp} (R²={best_r2:.4f})")

    # Cross-validate best model
    def best_pipeline_factory():
        return Pipeline([
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('select', SelectKBest(f_regression, k=min(50, best_k * 3))),
            ('scaler', StandardScaler()),
            ('ensemble', MLPEnsemble(
                n_estimators=7,
                hidden_layer_sizes=best_config.get("hidden", (32,)),
                alpha=best_config.get("alpha", 0.5),
                learning_rate_init=0.01,
                max_iter=500,
            ))
        ])

    cv_metrics = cross_validate_best(X, y, best_pipeline_factory, n_folds=5)

    # Calibration of best model
    print("\nCalibrating best model...")
    y_pred_all = np.clip(best_pipeline.predict(X), 0, 1)
    calib_stats = calibrate_thresholds(y, y_pred_all)

    # Save best model
    print(f"\nSaving best model...")
    model_path = output_path / "qpp_mlp_improved_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"  Model saved: {model_path}")

    # Save config
    config = {
        "model_type": "mlp_improved",
        "best_experiment": best_exp,
        "experiments": {k: {"test_r2": v["r2"], "test_rmse": v["rmse"]} for k, v in results.items()},
        "cv_metrics": cv_metrics,
        "calibration": calib_stats,
        "best_config": best_config,
        "best_k": best_k,
    }

    config_path = output_path / "qpp_mlp_improved_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved: {config_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Test R²:     {best_r2:.4f}")
    print(f"CV Mean R²:       {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
    print(f"XGBoost baseline: {baseline_xgb:.4f}")
    print(f"Improvement:      {best_r2 - baseline_xgb:+.4f} ({(best_r2/baseline_xgb - 1)*100:+.1f}%)")

    return best_pipeline, results


if __name__ == "__main__":
    main()
