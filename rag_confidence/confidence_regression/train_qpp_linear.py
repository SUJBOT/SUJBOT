"""
Train Linear Regression models for RAG confidence prediction.

Tests Ridge, Lasso, and ElasticNet with various regularization strengths.

Usage:
    uv run python rag_confidence/confidence_regression/train_qpp_linear.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

from qpp_extractor import ImprovedQPPExtractor, compute_recall_at_k


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
    return {"r2": float(r2), "rmse": float(rmse), "mae": float(mae), "correlation": float(corr)}


def calibrate_thresholds(y_true, y_pred):
    """Calculate calibration stats."""
    bins = [(0.0, 0.5, "VERY_LOW"), (0.5, 0.75, "LOW"),
            (0.75, 0.9, "MEDIUM"), (0.9, 1.01, "HIGH")]
    stats = {}
    print("\nCalibration:")
    for low, high, name in bins:
        mask = (y_pred >= low) & (y_pred < high)
        if mask.sum() > 0:
            stats[name.lower()] = {
                "actual_recall": float(y_true[mask].mean()),
                "count": int(mask.sum())
            }
            print(f"  {name:10s}: actual={y_true[mask].mean():.3f}, n={mask.sum()}")
    return stats


def experiment_ridge(X_train, y_train, X_val, y_val, X_test, y_test):
    """Test Ridge regression with various alpha values."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Ridge Regression")
    print("="*60)

    best_val_r2 = -np.inf
    best_alpha = None
    best_pipeline = None

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for alpha in alphas:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred_val = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        print(f"  alpha={alpha:>6}: Val R²={val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            best_pipeline = pipeline

    print(f"\n  Best alpha: {best_alpha}")
    train_metrics = evaluate_model(best_pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "  Test")

    return best_pipeline, test_metrics, best_alpha


def experiment_lasso(X_train, y_train, X_val, y_val, X_test, y_test):
    """Test Lasso regression for feature selection."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Lasso Regression (L1)")
    print("="*60)

    best_val_r2 = -np.inf
    best_alpha = None
    best_pipeline = None

    alphas = [0.0001, 0.001, 0.01, 0.1]

    for alpha in alphas:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=alpha, random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)
        y_pred_val = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        # Count non-zero coefficients
        n_nonzero = np.sum(pipeline.named_steps['lasso'].coef_ != 0)
        print(f"  alpha={alpha:>6}: Val R²={val_r2:.4f}, features={n_nonzero}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            best_pipeline = pipeline

    print(f"\n  Best alpha: {best_alpha}")
    train_metrics = evaluate_model(best_pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "  Test")

    return best_pipeline, test_metrics, best_alpha


def experiment_elasticnet(X_train, y_train, X_val, y_val, X_test, y_test):
    """Test ElasticNet (L1 + L2)."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: ElasticNet (L1 + L2)")
    print("="*60)

    best_val_r2 = -np.inf
    best_params = None
    best_pipeline = None

    configs = [
        {"alpha": 0.01, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.5},
        {"alpha": 0.01, "l1_ratio": 0.2},
        {"alpha": 0.01, "l1_ratio": 0.8},
        {"alpha": 0.1, "l1_ratio": 0.2},
    ]

    for cfg in configs:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('elastic', ElasticNet(alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"],
                                   random_state=42, max_iter=10000))
        ])
        pipeline.fit(X_train, y_train)
        y_pred_val = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        print(f"  alpha={cfg['alpha']}, l1_ratio={cfg['l1_ratio']}: Val R²={val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_params = cfg
            best_pipeline = pipeline

    print(f"\n  Best params: {best_params}")
    train_metrics = evaluate_model(best_pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "  Test")

    return best_pipeline, test_metrics, best_params


def experiment_poly_ridge(X_train, y_train, X_val, y_val, X_test, y_test):
    """Test Ridge with polynomial features."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Ridge + Polynomial Features")
    print("="*60)

    best_val_r2 = -np.inf
    best_config = None
    best_pipeline = None

    configs = [
        {"degree": 2, "interaction_only": True, "alpha": 1.0},
        {"degree": 2, "interaction_only": True, "alpha": 10.0},
        {"degree": 2, "interaction_only": False, "alpha": 10.0},
        {"degree": 2, "interaction_only": False, "alpha": 100.0},
    ]

    for cfg in configs:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=cfg["degree"],
                                        interaction_only=cfg["interaction_only"],
                                        include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=cfg["alpha"], random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        n_features = pipeline.named_steps['poly'].n_output_features_
        y_pred_val = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, y_pred_val)

        int_only = "int" if cfg["interaction_only"] else "full"
        print(f"  deg={cfg['degree']}, {int_only}, alpha={cfg['alpha']}: "
              f"Val R²={val_r2:.4f} (n_feat={n_features})")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_config = cfg
            best_pipeline = pipeline

    print(f"\n  Best config: {best_config}")
    train_metrics = evaluate_model(best_pipeline, X_train, y_train, "  Train")
    val_metrics = evaluate_model(best_pipeline, X_val, y_val, "  Val")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, "  Test")

    return best_pipeline, test_metrics, best_config


def analyze_coefficients(pipeline, feature_names):
    """Analyze and display model coefficients."""
    print("\nFeature Coefficients:")

    # Get the regressor (could be ridge, lasso, or elasticnet)
    regressor = None
    for name in ['ridge', 'lasso', 'elastic']:
        if name in pipeline.named_steps:
            regressor = pipeline.named_steps[name]
            break

    if regressor is None:
        return {}

    coefs = regressor.coef_

    # Handle polynomial features
    if 'poly' in pipeline.named_steps:
        poly = pipeline.named_steps['poly']
        poly_names = poly.get_feature_names_out(feature_names)
        names = poly_names
    else:
        names = feature_names

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    coef_dict = {}
    for i, idx in enumerate(sorted_idx[:15]):  # Top 15
        name = names[idx] if idx < len(names) else f"feature_{idx}"
        coef = coefs[idx]
        coef_dict[name] = float(coef)
        sign = "+" if coef > 0 else ""
        print(f"  {i+1:2d}. {name:30s} {sign}{coef:.4f}")

    return coef_dict


def cross_validate_model(X, y, model_factory, n_folds=5):
    """Cross-validate a model."""
    print(f"\nCross-validating ({n_folds} folds)...")

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = np.clip(model.predict(X_val), 0, 1)
        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

    r2_scores = np.array(r2_scores)
    rmse_scores = np.array(rmse_scores)

    print(f"  R² per fold: {r2_scores}")
    print(f"  Mean R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

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
    print("LINEAR REGRESSION TRAINING")
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

    # Experiment 1: Ridge
    ridge_pipeline, results["ridge"], best_ridge_alpha = experiment_ridge(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 2: Lasso
    lasso_pipeline, results["lasso"], best_lasso_alpha = experiment_lasso(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 3: ElasticNet
    elastic_pipeline, results["elasticnet"], best_elastic_params = experiment_elasticnet(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Experiment 4: Polynomial Ridge
    poly_pipeline, results["poly_ridge"], best_poly_config = experiment_poly_ridge(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    baselines = {
        "xgboost": 0.2380,
        "mlp_improved": 0.2456,
        "mlp_original": 0.1479,
    }

    print(f"\n{'Model':<20} {'Test R²':>10} {'vs XGBoost':>12} {'vs MLP-Imp':>12}")
    print("-" * 56)

    best_exp = None
    best_r2 = -np.inf

    for exp_name, metrics in results.items():
        r2 = metrics["r2"]
        diff_xgb = r2 - baselines["xgboost"]
        diff_mlp = r2 - baselines["mlp_improved"]

        print(f"{exp_name:<20} {r2:>10.4f} {diff_xgb:>+12.4f} {diff_mlp:>+12.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_exp = exp_name

    print("-" * 56)
    print(f"{'XGBoost':<20} {baselines['xgboost']:>10.4f}")
    print(f"{'MLP Improved':<20} {baselines['mlp_improved']:>10.4f}")
    print(f"\nBest linear model: {best_exp} (R²={best_r2:.4f})")

    # Get best pipeline
    best_pipelines = {
        "ridge": ridge_pipeline,
        "lasso": lasso_pipeline,
        "elasticnet": elastic_pipeline,
        "poly_ridge": poly_pipeline,
    }
    best_pipeline = best_pipelines[best_exp]

    # Analyze coefficients
    coef_dict = analyze_coefficients(best_pipeline, feature_names)

    # Cross-validate best model
    def best_model_factory():
        if best_exp == "ridge":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=best_ridge_alpha, random_state=42))
            ])
        elif best_exp == "lasso":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=best_lasso_alpha, random_state=42, max_iter=10000))
            ])
        elif best_exp == "elasticnet":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('elastic', ElasticNet(**best_elastic_params, random_state=42, max_iter=10000))
            ])
        else:  # poly_ridge
            return Pipeline([
                ('poly', PolynomialFeatures(degree=best_poly_config["degree"],
                                            interaction_only=best_poly_config["interaction_only"],
                                            include_bias=False)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=best_poly_config["alpha"], random_state=42))
            ])

    cv_metrics = cross_validate_model(X, y, best_model_factory, n_folds=5)

    # Calibration
    y_pred_all = np.clip(best_pipeline.predict(X), 0, 1)
    calib_stats = calibrate_thresholds(y, y_pred_all)

    # Save best model
    print(f"\nSaving best linear model...")
    model_path = output_path / "qpp_linear_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"  Model saved: {model_path}")

    # Save config
    config = {
        "model_type": "linear",
        "best_model": best_exp,
        "experiments": {k: {"test_r2": v["r2"], "test_rmse": v["rmse"]} for k, v in results.items()},
        "cv_metrics": cv_metrics,
        "calibration": calib_stats,
        "coefficients": coef_dict,
        "feature_names": feature_names,
    }

    config_path = output_path / "qpp_linear_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"  Config saved: {config_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Linear Model:  {best_exp}")
    print(f"Best Test R²:       {best_r2:.4f}")
    print(f"CV Mean R²:         {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
    print(f"XGBoost:            {baselines['xgboost']:.4f}")
    print(f"MLP Improved:       {baselines['mlp_improved']:.4f}")

    return best_pipeline, results


if __name__ == "__main__":
    main()
