"""
Centralized constants for backend configuration.

SSOT for agent variant configuration. Each variant maps to a single model
(no per-agent tiering — the single-agent system uses one model per variant).

SSOT Architecture:
- Primary source: config.json -> agent_variants section
- Fallback: Built-in defaults
- Pattern: Lazy loading with _ensure_config_loaded()
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Valid agent variants type hint
AgentVariant = Literal["remote", "local"]

# Module-level state (lazy loaded from config)
_config_loaded = False
_VARIANT_CONFIG: dict[str, dict[str, str]] = {}
_DEFAULT_VARIANT: str = "remote"
_DEEPINFRA_SUPPORTED_MODELS: frozenset[str] = frozenset()


# =============================================================================
# Built-in Defaults (fallback when config unavailable)
# =============================================================================

_BUILTIN_VARIANT_CONFIG = {
    "remote": {
        "display_name": "Remote (Haiku 4.5)",
        "model": "claude-haiku-4-5-20251001",
    },
    "local": {
        "display_name": "Local (Qwen3 VL 235B)",
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    },
}

_BUILTIN_DEFAULT_VARIANT = "remote"

_BUILTIN_DEEPINFRA_SUPPORTED_MODELS = frozenset({
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
})


# =============================================================================
# Config Loading Functions
# =============================================================================

def _load_builtin_defaults() -> None:
    """Load built-in fallback defaults."""
    global _VARIANT_CONFIG, _DEFAULT_VARIANT, _DEEPINFRA_SUPPORTED_MODELS
    _VARIANT_CONFIG = _BUILTIN_VARIANT_CONFIG.copy()
    _DEFAULT_VARIANT = _BUILTIN_DEFAULT_VARIANT
    _DEEPINFRA_SUPPORTED_MODELS = _BUILTIN_DEEPINFRA_SUPPORTED_MODELS


def _ensure_config_loaded() -> None:
    """
    Lazy load agent variant configuration from config.json.

    SSOT Source: config.json -> agent_variants section
    """
    global _config_loaded, _VARIANT_CONFIG, _DEFAULT_VARIANT, _DEEPINFRA_SUPPORTED_MODELS

    if _config_loaded:
        return

    try:
        from src.config import get_config
        config = get_config()

        if config.agent_variants is not None:
            _VARIANT_CONFIG = {}
            for variant_name, variant_config in config.agent_variants.variants.items():
                # Support both new format (single "model") and legacy ("opus_model"/"default_model")
                model = getattr(variant_config, "model", None)
                if not model:
                    model = getattr(variant_config, "default_model", "claude-haiku-4-5-20251001")
                _VARIANT_CONFIG[variant_name] = {
                    "display_name": variant_config.display_name,
                    "model": model,
                }

            _DEFAULT_VARIANT = config.agent_variants.default_variant

            logger.debug(
                "Agent variants loaded from config.json: %d variants",
                len(_VARIANT_CONFIG),
            )
        else:
            logger.debug("No agent_variants in config, using built-in defaults")
            _load_builtin_defaults()

        # Load DeepInfra supported models
        if hasattr(config.agent_variants, "deepinfra_supported_models"):
            _DEEPINFRA_SUPPORTED_MODELS = frozenset(config.agent_variants.deepinfra_supported_models)
        else:
            _DEEPINFRA_SUPPORTED_MODELS = _BUILTIN_DEEPINFRA_SUPPORTED_MODELS

        _config_loaded = True

    except ImportError as e:
        logger.info("Config module not available: %s. Using built-in defaults.", e)
        _load_builtin_defaults()
        _config_loaded = True

    except (KeyError, AttributeError, TypeError) as e:
        logger.warning(
            "Config schema error in agent_variants section: %s. Using built-in defaults.",
            e,
            exc_info=True,
        )
        _load_builtin_defaults()
        _config_loaded = True


def reload_constants() -> None:
    """Force reload of constants from config.json."""
    global _config_loaded
    _config_loaded = False
    _ensure_config_loaded()


# =============================================================================
# Public Getter Functions
# =============================================================================

def get_variant_config() -> dict[str, dict[str, str]]:
    """Get the variant configuration dictionary."""
    _ensure_config_loaded()
    return _VARIANT_CONFIG


def get_default_variant() -> str:
    """Get the default variant name."""
    _ensure_config_loaded()
    return _DEFAULT_VARIANT


def get_deepinfra_supported_models() -> frozenset[str]:
    """Get set of supported DeepInfra models."""
    _ensure_config_loaded()
    return _DEEPINFRA_SUPPORTED_MODELS


# =============================================================================
# Public Functions (main API)
# =============================================================================

def get_variant_model(variant: str) -> str:
    """
    Get model identifier for a variant.

    Args:
        variant: Agent variant ('remote' or 'local')

    Returns:
        Model identifier string

    Raises:
        KeyError: If variant is not found
    """
    _ensure_config_loaded()
    return _VARIANT_CONFIG[variant]["model"]


def is_valid_variant(variant: str) -> bool:
    """Check if variant is valid."""
    _ensure_config_loaded()
    return variant in _VARIANT_CONFIG


def get_all_variants() -> list[str]:
    """Get list of all available variant names."""
    _ensure_config_loaded()
    return list(_VARIANT_CONFIG.keys())


def get_variant_display_name(variant: str) -> str:
    """Get human-readable display name for a variant."""
    _ensure_config_loaded()
    return _VARIANT_CONFIG.get(variant, {}).get("display_name", variant)


# Initialize on first import
_ensure_config_loaded()
