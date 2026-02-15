"""
Pluggable QPP Model Interface for RAG Confidence Prediction.

Provides a unified interface for different classifier models:
- LogisticRegression: Interpretable baseline
- MLP: Multi-layer perceptron (production default)
- XGBoost: Gradient boosted trees alternative

All models output probabilities in [0, 1] for binary classification
of retrieval success (Recall@k).

Example:
    model = QPPModelFactory.create("mlp")  # Uses optimized (128, 64), α=0.1
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class QPPModelProtocol(Protocol):
    """Protocol for QPP models - defines required interface."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model on training data."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for positive class."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        ...


class BaseQPPModel(ABC):
    """
    Abstract base class for QPP models.

    Handles common functionality:
    - Feature scaling (StandardScaler)
    - Model persistence (save/load)
    - Probability clipping
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self.scaler: Optional[StandardScaler] = None
        self.model: Any = None
        self._is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying sklearn/xgboost model."""
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_features: bool = True,
    ) -> "BaseQPPModel":
        """
        Fit model on training data.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Binary labels, shape (n_samples,)
            scale_features: Whether to standardize features (default: True)

        Returns:
            Self for method chaining.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Ensure binary labels
        y = (y > 0.5).astype(int)

        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            X_scaled = X

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        logger.info(f"{self.name} fitted on {X.shape[0]} samples, {X.shape[1]} features")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for positive class.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probabilities for positive class, shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)

        # Scale features if scaler was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # Get probabilities
        probas = self.model.predict_proba(X_scaled)

        # Return probability of positive class
        if probas.ndim == 2:
            probas = probas[:, 1]

        # Clip to [0, 1] for numerical stability
        return np.clip(probas, 0.0, 1.0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions, shape (n_samples,)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Save model and scaler to file."""
        path = Path(path)
        state = {
            "name": self.name,
            "model": self.model,
            "scaler": self.scaler,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved {self.name} to {path}")

    def load(self, path: Union[str, Path]) -> "BaseQPPModel":
        """Load model and scaler from file."""
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.name = state["name"]
        self.model = state["model"]
        self.scaler = state["scaler"]
        self._is_fitted = state["is_fitted"]
        logger.info(f"Loaded {self.name} from {path}")
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted


class LogisticRegressionModel(BaseQPPModel):
    """
    Logistic Regression wrapper for QPP.

    Pros:
    - Highly interpretable (feature coefficients)
    - Fast training and inference
    - Good baseline

    Cons:
    - Limited capacity for complex patterns
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
    ):
        """
        Initialize Logistic Regression model.

        Args:
            C: Inverse regularization strength
            max_iter: Maximum iterations for solver
            class_weight: Handle class imbalance ("balanced" or None)
        """
        super().__init__(name="LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight

    def _create_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
            random_state=42,
        )

    def get_coefficients(self) -> Dict[str, float]:
        """
        Get feature coefficients (for interpretability).

        Returns:
            Dict mapping feature index to coefficient.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")
        return {
            f"feature_{i}": coef
            for i, coef in enumerate(self.model.coef_[0])
        }


class MLPModel(BaseQPPModel):
    """
    Multi-Layer Perceptron wrapper for QPP.

    Pros:
    - Can learn complex non-linear patterns
    - Good balance of capacity and speed

    Cons:
    - Less interpretable than logistic regression
    - Requires more hyperparameter tuning
    """

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (128, 64),
        alpha: float = 0.1,
        max_iter: int = 500,
        early_stopping: bool = True,
        learning_rate_init: float = 0.001,
    ):
        """
        Initialize MLP model.

        Default configuration (wide_low_reg) optimized via architecture ablation:
        - hidden_sizes=(128, 64): Wider layers improve capacity
        - alpha=0.1: Lower regularization for better fit

        Ablation study results (2026-01-23):
        - ViDoRe: 0.8249 AUROC (+2.21% vs baseline 64,32,α=0.5)
        - SUJBOT: 0.9156 AUROC (+2.84% vs baseline)

        Args:
            hidden_sizes: Tuple of hidden layer sizes
            alpha: L2 regularization strength
            max_iter: Maximum training epochs
            early_stopping: Use early stopping with validation set
            learning_rate_init: Initial learning rate
        """
        super().__init__(name="MLP")
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.learning_rate_init = learning_rate_init

    def _create_model(self) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=0.1 if self.early_stopping else 0.0,
            learning_rate_init=self.learning_rate_init,
            random_state=42,
            verbose=False,
        )


class XGBoostModel(BaseQPPModel):
    """
    XGBoost wrapper for QPP.

    Pros:
    - Often best performance on tabular data
    - Handles feature interactions well
    - Built-in feature importance

    Cons:
    - Requires xgboost package
    - Can overfit on small datasets
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_child_weight: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            min_child_weight: Minimum sum of instance weight in child
            subsample: Subsample ratio for training
            colsample_bytree: Column subsample ratio per tree
            scale_pos_weight: Balance positive/negative weights
        """
        super().__init__(name="XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self._xgb_available = self._check_xgboost()

    def _check_xgboost(self) -> bool:
        """Check if xgboost is available."""
        try:
            import xgboost
            return True
        except ImportError:
            logger.warning("xgboost not installed. XGBoostModel will not work.")
            return False

    def _create_model(self) -> Any:
        if not self._xgb_available:
            raise ImportError("xgboost is required for XGBoostModel. Install with: pip install xgboost")

        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict mapping feature index to importance score.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted.")
        return {
            f"feature_{i}": imp
            for i, imp in enumerate(self.model.feature_importances_)
        }


class QPPModelFactory:
    """
    Factory for creating QPP models.

    Example:
        model = QPPModelFactory.create("mlp")  # Optimized defaults: (128, 64), α=0.1
        model = QPPModelFactory.create("logistic", C=0.5)
        model = QPPModelFactory.create("xgboost", max_depth=4)
    """

    AVAILABLE_MODELS = ["logistic", "mlp", "xgboost"]

    @staticmethod
    def create(model_type: str, **kwargs: Any) -> BaseQPPModel:
        """
        Create a QPP model by type.

        Args:
            model_type: One of "logistic", "mlp", "xgboost"
            **kwargs: Model-specific parameters

        Returns:
            Configured model instance

        Raises:
            ValueError: If model_type is unknown
        """
        model_type = model_type.lower()

        if model_type == "logistic":
            return LogisticRegressionModel(**kwargs)
        elif model_type == "mlp":
            return MLPModel(**kwargs)
        elif model_type == "xgboost":
            return XGBoostModel(**kwargs)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {QPPModelFactory.AVAILABLE_MODELS}"
            )

    @staticmethod
    def list_available() -> List[str]:
        """Return list of available model types."""
        return QPPModelFactory.AVAILABLE_MODELS.copy()

    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type.

        Args:
            model_type: Model type name

        Returns:
            Dict of default parameters
        """
        configs = {
            "logistic": {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
            },
            "mlp": {
                "hidden_sizes": (128, 64),
                "alpha": 0.1,
                "max_iter": 500,
                "early_stopping": True,
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "scale_pos_weight": None,
            },
        }
        return configs.get(model_type.lower(), {})


def load_model(path: Union[str, Path]) -> BaseQPPModel:
    """
    Load a QPP model from file.

    The model type is inferred from the saved state.

    Args:
        path: Path to saved model file

    Returns:
        Loaded model instance
    """
    path = Path(path)
    with open(path, "rb") as f:
        state = pickle.load(f)

    name = state.get("name", "")

    # Infer model type from name
    if "Logistic" in name:
        model = LogisticRegressionModel()
    elif "XGBoost" in name:
        model = XGBoostModel()
    else:
        model = MLPModel()

    model.load(path)
    return model


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Generate synthetic features and labels
    n_samples = 500
    n_features = 23

    X = np.random.randn(n_samples, n_features)
    # Create labels based on first few features
    y = (X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_samples) > 0).astype(int)

    # Train/test split
    train_idx = np.arange(400)
    test_idx = np.arange(400, 500)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Testing QPP Model Factory")
    print("=" * 50)

    for model_type in ["logistic", "mlp"]:
        print(f"\n{model_type.upper()}:")
        model = QPPModelFactory.create(model_type)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        preds = model.predict(X_test)

        accuracy = (preds == y_test).mean()
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Proba range: [{probas.min():.3f}, {probas.max():.3f}]")

    # Test XGBoost if available
    try:
        import xgboost
        print(f"\nXGBOOST:")
        model = QPPModelFactory.create("xgboost")
        model.fit(X_train, y_train)
        probas = model.predict_proba(X_test)
        preds = model.predict(X_test)
        accuracy = (preds == y_test).mean()
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Proba range: [{probas.min():.3f}, {probas.max():.3f}]")
    except ImportError:
        print("\nXGBoost not installed, skipping test.")

    print("\nAll tests passed!")
