"""
V2 Configuration for RAG Confidence System.

Defines configuration dataclasses for UQPP, SCA, and the main V2Config.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import json
from pathlib import Path


@dataclass
class StabilityConfig:
    """Hyperparameters for UQPP stability computation.

    Attributes:
        M: Number of perturbation trials (default: 3)
        k: Top-k for Jaccard comparison (default: 10)
        sigma: Noise scale relative to embedding norm (default: 0.02)
    """
    M: int = 3
    k: int = 10
    sigma: float = 0.02

    def __post_init__(self):
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0 < self.sigma < 1:
            raise ValueError(f"sigma must be in (0, 1), got {self.sigma}")


@dataclass
class DenseQPPConfig:
    """Hyperparameters for Dense-QPP (CIKM 2023).

    Dense-QPP uses AWGN perturbations and RBO comparison for query robustness.

    Attributes:
        gamma: ENR ratio - controls noise level (5-7% recommended). Default: 0.05.
        M: Number of perturbations (CLT recommends 30). Default: 30.
        k: Top-k for RBO comparison. Default: 100.
        p: RBO persistence parameter (0.9 = top-10 gets ~86% weight). Default: 0.9.
    """
    gamma: float = 0.05
    M: int = 30
    k: int = 100
    p: float = 0.9

    def __post_init__(self):
        if not 0 < self.gamma < 1:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0 < self.p < 1:
            raise ValueError(f"p must be in (0, 1), got {self.p}")


@dataclass
class V2Config:
    """Configuration for RAG Confidence v2.

    This configuration controls UQPP and SCA feature extraction,
    as well as the thresholds for confidence bands and p_final computation.

    Attributes:
        enable_uqpp: Enable UQPP features (coherence + stability)
        enable_uqpp_coherence: Enable coherence computation (cheap)
        enable_uqpp_stability: Enable stability computation (conditional)
        enable_sca: Enable SCA (Sufficiency Context Assessment)

        T_HIGH: High confidence threshold (default: 0.90)
        T_MED: Medium confidence threshold (default: 0.75)
        T_LOW: Low confidence threshold (default: 0.50)

        U_HIGH: High UQPP threshold - stable/coherent (default: 0.75)
        U_LOW: Low UQPP threshold - triggers stability (default: 0.35)

        stability: Stability computation hyperparameters
        uqpp_weights: Weights for combining UQPP signals

        p_final_weights: Weights for p_final computation
            - w_sup: weight for p_sup (default: 0.7)
            - w_uqpp: weight for u_score (default: 0.2)
            - w_sca: weight for p_suff (default: 0.1)

        force_stability: Always compute stability (for debugging)

    Example:
        >>> config = V2Config()
        >>> config.enable_uqpp
        True
        >>> config.T_HIGH
        0.9

        >>> # Custom config
        >>> config = V2Config(enable_sca=True, T_HIGH=0.85)
    """

    # Feature toggles
    # NOTE: UQPP disabled by default - Dense-QPP evaluation showed it degrades AUROC.
    # See rag_confidence/evaluation/compare_classification.py results.
    enable_uqpp: bool = False  # Disabled - evaluation showed no improvement
    enable_uqpp_stability: bool = False  # Disabled - Dense-QPP degraded AUROC by 0.008-0.055
    enable_sca: bool = False  # Off by default due to cost
    # NOTE: Coherence removed - redundant with QPP features (sim_std_top10, bimodal_gap, etc.)

    # Confidence band thresholds (v1 compatible)
    T_HIGH: float = 0.90
    T_MED: float = 0.75
    T_LOW: float = 0.50

    # UQPP thresholds (for stability)
    U_HIGH: float = 0.75
    U_LOW: float = 0.35

    # Sub-configs
    stability: StabilityConfig = field(default_factory=StabilityConfig)  # Legacy (Jaccard-based)
    dense_qpp: DenseQPPConfig = field(default_factory=DenseQPPConfig)  # Dense-QPP (RBO-based)

    # p_final computation weights
    # NOTE: UQPP disabled, so p_final = p_sup in practice
    p_final_w_sup: float = 1.0   # Weight for p_sup (v1 supervised confidence)
    p_final_w_uqpp: float = 0.0  # Weight for u_score (disabled - no improvement)
    p_final_w_sca: float = 0.0   # Weight for p_suff (disabled by default)

    # Debug options
    force_stability: bool = False

    # SCA model (for LLM-based assessment)
    sca_model: str = "claude-haiku-4-5"
    sca_max_chunks: int = 20

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate threshold ordering
        if not 0 <= self.T_LOW <= self.T_MED <= self.T_HIGH <= 1:
            raise ValueError(
                f"Thresholds must satisfy 0 <= T_LOW <= T_MED <= T_HIGH <= 1, "
                f"got T_LOW={self.T_LOW}, T_MED={self.T_MED}, T_HIGH={self.T_HIGH}"
            )

        # Validate UQPP thresholds
        if not 0 <= self.U_LOW <= self.U_HIGH <= 1:
            raise ValueError(
                f"UQPP thresholds must satisfy 0 <= U_LOW <= U_HIGH <= 1, "
                f"got U_LOW={self.U_LOW}, U_HIGH={self.U_HIGH}"
            )

        # Validate p_final weights sum to 1
        total = self.p_final_w_sup + self.p_final_w_uqpp + self.p_final_w_sca
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"p_final weights must sum to 1.0, got {total}")

    @classmethod
    def from_json(cls, path: Path) -> "V2Config":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            V2Config instance
        """
        with open(path) as f:
            data = json.load(f)

        # Parse nested configs
        stability_data = data.pop("stability", {})
        dense_qpp_data = data.pop("dense_qpp", {})
        # Remove legacy fields if present
        data.pop("uqpp_weights", None)
        data.pop("enable_uqpp_coherence", None)
        data.pop("invert_coherence", None)

        stability = StabilityConfig(**stability_data) if stability_data else StabilityConfig()
        dense_qpp = DenseQPPConfig(**dense_qpp_data) if dense_qpp_data else DenseQPPConfig()

        return cls(stability=stability, dense_qpp=dense_qpp, **data)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        data = {
            "enable_uqpp": self.enable_uqpp,
            "enable_uqpp_stability": self.enable_uqpp_stability,
            "enable_sca": self.enable_sca,
            "T_HIGH": self.T_HIGH,
            "T_MED": self.T_MED,
            "T_LOW": self.T_LOW,
            "U_HIGH": self.U_HIGH,
            "U_LOW": self.U_LOW,
            "stability": {
                "M": self.stability.M,
                "k": self.stability.k,
                "sigma": self.stability.sigma,
            },
            "dense_qpp": {
                "gamma": self.dense_qpp.gamma,
                "M": self.dense_qpp.M,
                "k": self.dense_qpp.k,
                "p": self.dense_qpp.p,
            },
            "p_final_w_sup": self.p_final_w_sup,
            "p_final_w_uqpp": self.p_final_w_uqpp,
            "p_final_w_sca": self.p_final_w_sca,
            "force_stability": self.force_stability,
            "sca_model": self.sca_model,
            "sca_max_chunks": self.sca_max_chunks,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "enable_uqpp": self.enable_uqpp,
            "enable_uqpp_stability": self.enable_uqpp_stability,
            "enable_sca": self.enable_sca,
            "T_HIGH": self.T_HIGH,
            "T_MED": self.T_MED,
            "T_LOW": self.T_LOW,
            "U_HIGH": self.U_HIGH,
            "U_LOW": self.U_LOW,
            "stability": {
                "M": self.stability.M,
                "k": self.stability.k,
                "sigma": self.stability.sigma,
            },
            "dense_qpp": {
                "gamma": self.dense_qpp.gamma,
                "M": self.dense_qpp.M,
                "k": self.dense_qpp.k,
                "p": self.dense_qpp.p,
            },
            "p_final_weights": {
                "w_sup": self.p_final_w_sup,
                "w_uqpp": self.p_final_w_uqpp,
                "w_sca": self.p_final_w_sca,
            },
        }


# Default config instance for convenience
DEFAULT_CONFIG = V2Config()
