"""
Model registry for SUJBOT pipeline.

SSOT: All model configuration is loaded from config.json (model_registry section).
This class provides helper methods for model resolution, provider detection, and
pricing lookup - all from config.json.

Features:
- Single source of truth for model configuration (config.json)
- Supports both OLD format (string) and NEW format (object with metadata)
- Provider detection from config (no pattern matching needed for new format)
- Pricing lookup from config (no hardcoded PRICING dict needed)
- Embedding dimensions from config
- Lazy loading - config loaded on first access
- Backward compatible - falls back to built-in defaults if config missing

NEW CONFIG FORMAT (preferred):
    "haiku": {
        "id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "pricing": {"input": 1.00, "output": 5.00},
        "context_window": 200000,
        "supports_caching": true
    }

OLD CONFIG FORMAT (still supported):
    "haiku": "claude-haiku-4-5-20251001"

Usage:
    from src.utils.model_registry import ModelRegistry

    # Resolve model alias
    model = ModelRegistry.resolve_llm("haiku")
    # Returns: "claude-haiku-4-5-20251001"

    # Get provider (from config, no pattern guessing)
    provider = ModelRegistry.get_provider("haiku", "llm")
    # Returns: "anthropic"

    # Get pricing (from config)
    pricing = ModelRegistry.get_pricing("haiku", "llm")
    # Returns: {"input": 1.00, "output": 5.00}

    # Get full model config
    config = ModelRegistry.get_model_config("haiku", "llm")
    # Returns: LLMModelConfig object with all metadata
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
# Data classes for model configuration (lightweight, no Pydantic dependency)
# ============================================================================

# Valid provider literals (keep in sync with config_schema.py)
LLM_PROVIDERS = frozenset({"anthropic", "openai", "google", "deepinfra", "local_llm", "local_llm_8b"})
EMBEDDING_PROVIDERS = frozenset({"openai", "deepinfra", "voyage", "huggingface"})


@dataclass
class ModelPricingData:
    """Pricing per 1M tokens."""
    input: float
    output: float = 0.0

    def __post_init__(self):
        if self.input < 0:
            raise ValueError(f"Input pricing cannot be negative: {self.input}")
        if self.output < 0:
            raise ValueError(f"Output pricing cannot be negative: {self.output}")


@dataclass
class LLMModelData:
    """LLM model configuration data."""
    id: str
    provider: str
    pricing: ModelPricingData
    context_window: int = 128000
    supports_caching: bool = False
    supports_extended_thinking: bool = False

    def __post_init__(self):
        if not self.id:
            raise ValueError("Model ID cannot be empty")
        if self.provider not in LLM_PROVIDERS:
            raise ValueError(f"Invalid LLM provider '{self.provider}'. Valid: {sorted(LLM_PROVIDERS)}")
        if self.context_window < 1000:
            raise ValueError(f"Context window must be >= 1000: {self.context_window}")


@dataclass
class EmbeddingModelData:
    """Embedding model configuration data."""
    id: str
    provider: str
    pricing: ModelPricingData
    dimensions: int
    is_local: bool = False

    def __post_init__(self):
        if not self.id:
            raise ValueError("Model ID cannot be empty")
        if self.provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"Invalid embedding provider '{self.provider}'. Valid: {sorted(EMBEDDING_PROVIDERS)}")
        if self.dimensions < 1:
            raise ValueError(f"Dimensions must be >= 1: {self.dimensions}")


@dataclass
class RerankerModelData:
    """Reranker model configuration data."""
    id: str
    is_local: bool = True

    def __post_init__(self):
        if not self.id:
            raise ValueError("Model ID cannot be empty")


# Union type for any model config
ModelConfigData = Union[LLMModelData, EmbeddingModelData, RerankerModelData]


# ============================================================================
# Global cache for config-loaded models (lazy initialization)
# ============================================================================

_config_loaded = False
# Raw config entries (can be string or dict)
_LLM_MODELS_RAW: Dict[str, Any] = {}
_EMBEDDING_MODELS_RAW: Dict[str, Any] = {}
_RERANKER_MODELS_RAW: Dict[str, Any] = {}
# Legacy dimensions dict (for backward compatibility)
_EMBEDDING_DIMENSIONS: Dict[str, int] = {}


def _parse_pricing(pricing_data: dict) -> ModelPricingData:
    """Parse pricing dict into ModelPricingData."""
    return ModelPricingData(
        input=float(pricing_data.get("input", 0.0)),
        output=float(pricing_data.get("output", 0.0)),
    )


def _parse_llm_config(alias: str, entry: Any) -> LLMModelData:
    """
    Parse LLM config entry (string, dict, or Pydantic model) into LLMModelData.

    For string entries (old format), we infer provider from patterns.
    For dict/Pydantic entries (new format), we read provider directly.
    """
    if isinstance(entry, str):
        # Old format: "haiku": "claude-haiku-4-5-20251001"
        model_id = entry
        provider = _infer_provider_from_model_id(model_id)
        return LLMModelData(
            id=model_id,
            provider=provider,
            pricing=ModelPricingData(input=0.0, output=0.0),  # Unknown pricing
        )
    elif isinstance(entry, dict):
        # New format from raw JSON (before Pydantic validation)
        return LLMModelData(
            id=entry["id"],
            provider=entry["provider"],
            pricing=_parse_pricing(entry.get("pricing", {})),
            context_window=entry.get("context_window", 128000),
            supports_caching=entry.get("supports_caching", False),
            supports_extended_thinking=entry.get("supports_extended_thinking", False),
        )
    elif hasattr(entry, "id") and hasattr(entry, "provider"):
        # Pydantic model object (LLMModelConfig) - after Pydantic validation
        pricing_data = ModelPricingData(
            input=entry.pricing.input if entry.pricing else 0.0,
            output=entry.pricing.output if entry.pricing else 0.0,
        )
        return LLMModelData(
            id=entry.id,
            provider=entry.provider,
            pricing=pricing_data,
            context_window=getattr(entry, "context_window", 128000),
            supports_caching=getattr(entry, "supports_caching", False),
            supports_extended_thinking=getattr(entry, "supports_extended_thinking", False),
        )
    else:
        raise ValueError(f"Invalid LLM config entry for '{alias}': {type(entry)}")


def _parse_embedding_config(alias: str, entry: Any) -> EmbeddingModelData:
    """Parse embedding config entry (string, dict, or Pydantic model) into EmbeddingModelData."""
    if isinstance(entry, str):
        # Old format
        model_id = entry
        provider = _infer_provider_from_model_id(model_id, "embedding")
        dims = _EMBEDDING_DIMENSIONS.get(model_id, 1024)
        return EmbeddingModelData(
            id=model_id,
            provider=provider,
            pricing=ModelPricingData(input=0.0),
            dimensions=dims,
            is_local=provider == "huggingface",
        )
    elif isinstance(entry, dict):
        # New format from raw JSON
        return EmbeddingModelData(
            id=entry["id"],
            provider=entry["provider"],
            pricing=_parse_pricing(entry.get("pricing", {})),
            dimensions=entry["dimensions"],
            is_local=entry.get("is_local", False),
        )
    elif hasattr(entry, "id") and hasattr(entry, "provider"):
        # Pydantic model object (EmbeddingModelConfig)
        pricing_data = ModelPricingData(
            input=entry.pricing.input if entry.pricing else 0.0,
            output=entry.pricing.output if entry.pricing else 0.0,
        )
        return EmbeddingModelData(
            id=entry.id,
            provider=entry.provider,
            pricing=pricing_data,
            dimensions=entry.dimensions,
            is_local=getattr(entry, "is_local", False),
        )
    else:
        raise ValueError(f"Invalid embedding config entry for '{alias}': {type(entry)}")


def _parse_reranker_config(alias: str, entry: Any) -> RerankerModelData:
    """Parse reranker config entry (string, dict, or Pydantic model) into RerankerModelData."""
    if isinstance(entry, str):
        return RerankerModelData(id=entry, is_local=True)
    elif isinstance(entry, dict):
        return RerankerModelData(
            id=entry["id"],
            is_local=entry.get("is_local", True),
        )
    elif hasattr(entry, "id"):
        # Pydantic model object (RerankerModelConfig)
        return RerankerModelData(
            id=entry.id,
            is_local=getattr(entry, "is_local", True),
        )
    else:
        raise ValueError(f"Invalid reranker config entry for '{alias}': {type(entry)}")


def _extract_model_id(entry: Any) -> str:
    """Extract model ID from entry (string, dict, or Pydantic object)."""
    if isinstance(entry, str):
        return entry
    elif isinstance(entry, dict):
        return entry["id"]
    elif hasattr(entry, "id"):
        return entry.id
    else:
        raise ValueError(f"Cannot extract model ID from entry: {type(entry)}")


def _infer_provider_from_model_id(model_id: str, model_type: str = "llm") -> str:
    """
    Infer provider from model ID patterns (for backward compatibility).

    Used only for old string-based config entries.
    New object entries have explicit provider field.
    """
    model_lower = model_id.lower()

    # LLM patterns
    if any(kw in model_lower for kw in ["claude", "haiku", "sonnet", "opus"]):
        return "anthropic"
    if any(kw in model_lower for kw in ["gpt-", "o1", "o3", "o4"]):
        return "openai"
    if "gemini" in model_lower:
        return "google"
    if any(kw in model_id for kw in ["Qwen/", "meta-llama/", "MiniMaxAI/"]):
        return "deepinfra"
    if any(kw in model_lower for kw in ["qwen", "llama", "minimax"]):
        return "deepinfra"

    # Embedding patterns
    if model_type == "embedding":
        if "text-embedding" in model_lower:
            return "openai"
        if "voyage" in model_lower:
            return "voyage"
        if any(p in model_id for p in ["BAAI/", "intfloat/", "sentence-transformers/"]):
            return "huggingface"

    # Default to unknown
    return "unknown"


def _load_builtin_defaults():
    """
    Load built-in defaults (backward compatibility).

    These are used when config.json doesn't have model_registry section.
    """
    global _LLM_MODELS_RAW, _EMBEDDING_MODELS_RAW, _RERANKER_MODELS_RAW, _EMBEDDING_DIMENSIONS

    _LLM_MODELS_RAW = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-5-20250929",
        "opus": "claude-opus-4-5-20251101",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o3-mini": "o3-mini",
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
        "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "minimax-m2": "MiniMaxAI/MiniMax-M2",
    }

    _EMBEDDING_MODELS_RAW = {
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-3-small": "text-embedding-3-small",
        "bge-m3": "BAAI/bge-m3",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "voyage-3-large": "voyage-3-large",
        "voyage-3": "voyage-3",
        "qwen3-embedding": "Qwen/Qwen3-Embedding-8B",
    }

    _RERANKER_MODELS_RAW = {
        "ms-marco-mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "bge-reranker-large": "BAAI/bge-reranker-large",
    }

    _EMBEDDING_DIMENSIONS = {
        "Qwen/Qwen3-Embedding-8B": 4096,
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "voyage-3-large": 1024,
        "voyage-3": 1024,
    }


def _ensure_config_loaded():
    """
    Load model definitions from config.json on first access.

    This function is called automatically before any registry access.
    Falls back to built-in defaults if config.json doesn't have model_registry.
    """
    global _config_loaded, _LLM_MODELS_RAW, _EMBEDDING_MODELS_RAW, _RERANKER_MODELS_RAW
    global _EMBEDDING_DIMENSIONS

    if _config_loaded:
        return

    try:
        from src.config import get_config
        config = get_config()

        if config.model_registry is not None:
            # Load raw entries from config.json (SSOT)
            # These can be strings (old format) or dicts (new format)
            _LLM_MODELS_RAW = dict(config.model_registry.llm_models)
            _EMBEDDING_MODELS_RAW = dict(config.model_registry.embedding_models)
            _RERANKER_MODELS_RAW = dict(config.model_registry.reranker_models)
            _EMBEDDING_DIMENSIONS = dict(config.model_registry.embedding_dimensions)

            # Count new format vs old format entries
            new_format_count = sum(
                1 for v in _LLM_MODELS_RAW.values() if isinstance(v, dict)
            )
            total_count = len(_LLM_MODELS_RAW)

            logger.info(
                f"ModelRegistry loaded from config.json: "
                f"{total_count} LLM ({new_format_count} with metadata), "
                f"{len(_EMBEDDING_MODELS_RAW)} embedding, "
                f"{len(_RERANKER_MODELS_RAW)} reranker models"
            )
        else:
            logger.warning(
                "model_registry section not found in config.json, using built-in defaults."
            )
            _load_builtin_defaults()

        _config_loaded = True

    except ImportError as e:
        logger.info(f"Config module not available: {e}. Using built-in model registry.")
        _load_builtin_defaults()
        _config_loaded = True

    except (KeyError, AttributeError, TypeError) as e:
        logger.warning(
            f"Model registry config schema error: {e}. Using built-in defaults.",
            exc_info=True,
        )
        _load_builtin_defaults()
        _config_loaded = True


def reload_registry():
    """
    Force reload of model registry from config.

    Use this after modifying config.json at runtime (rare).
    """
    global _config_loaded
    _config_loaded = False
    _ensure_config_loaded()


class ModelRegistry:
    """
    Central registry for all model configuration - reads from config.json.

    Provides:
    - Model alias resolution (alias → full model ID)
    - Provider detection (from config or pattern inference)
    - Pricing lookup (from config)
    - Embedding dimensions (from config)

    SSOT: All data comes from config.json model_registry section.
    Built-in fallbacks used only if config section is missing.
    """

    # ========================================================================
    # MODEL CONFIG ACCESS (NEW API)
    # ========================================================================

    @classmethod
    def get_model_config(cls, alias: str, model_type: str = "llm") -> ModelConfigData:
        """
        Get full model configuration including provider and pricing.

        This is the primary API for accessing model metadata.

        Args:
            alias: Model alias or full model ID
            model_type: "llm", "embedding", or "reranker"

        Returns:
            Model configuration dataclass (LLMModelData, EmbeddingModelData, etc.)

        Raises:
            KeyError: If alias not found in registry

        Example:
            >>> config = ModelRegistry.get_model_config("haiku", "llm")
            >>> print(config.provider)  # "anthropic"
            >>> print(config.pricing.input)  # 1.00
        """
        _ensure_config_loaded()

        if model_type == "llm":
            if alias in _LLM_MODELS_RAW:
                return _parse_llm_config(alias, _LLM_MODELS_RAW[alias])
            # Try to find by model ID (for full model names like "claude-haiku-4-5-20251001")
            for key, entry in _LLM_MODELS_RAW.items():
                model_id = _extract_model_id(entry)
                if model_id == alias:
                    return _parse_llm_config(key, entry)
            raise KeyError(f"LLM model not found: {alias}")

        elif model_type == "embedding":
            if alias in _EMBEDDING_MODELS_RAW:
                return _parse_embedding_config(alias, _EMBEDDING_MODELS_RAW[alias])
            for key, entry in _EMBEDDING_MODELS_RAW.items():
                model_id = _extract_model_id(entry)
                if model_id == alias:
                    return _parse_embedding_config(key, entry)
            raise KeyError(f"Embedding model not found: {alias}")

        elif model_type == "reranker":
            if alias in _RERANKER_MODELS_RAW:
                return _parse_reranker_config(alias, _RERANKER_MODELS_RAW[alias])
            for key, entry in _RERANKER_MODELS_RAW.items():
                model_id = _extract_model_id(entry)
                if model_id == alias:
                    return _parse_reranker_config(key, entry)
            raise KeyError(f"Reranker model not found: {alias}")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @classmethod
    def get_pricing(cls, model: str, model_type: str = "llm") -> Dict[str, float]:
        """
        Get pricing for a model from config.

        Args:
            model: Model alias or full ID
            model_type: "llm" or "embedding"

        Returns:
            Dict with "input" and "output" prices per 1M tokens

        Example:
            >>> pricing = ModelRegistry.get_pricing("haiku", "llm")
            >>> print(pricing)  # {"input": 1.00, "output": 5.00}
        """
        try:
            config = cls.get_model_config(model, model_type)
            return {"input": config.pricing.input, "output": config.pricing.output}
        except KeyError:
            logger.warning(f"No pricing found for {model_type}/{model}, returning $0")
            return {"input": 0.0, "output": 0.0}

    @classmethod
    def get_provider(cls, model: str, model_type: str = "llm") -> str:
        """
        Get provider name for a model.

        For new format entries, reads provider directly from config.
        For old format entries, infers provider from model ID patterns.

        Args:
            model: Model name or alias
            model_type: Type of model ("llm", "embedding", "reranker")

        Returns:
            Provider name ("anthropic", "openai", "google", "voyage", "huggingface", "deepinfra")

        Raises:
            ValueError: If provider cannot be determined

        Example:
            >>> ModelRegistry.get_provider("haiku", "llm")
            'anthropic'

            >>> ModelRegistry.get_provider("bge-m3", "embedding")
            'huggingface'
        """
        try:
            config = cls.get_model_config(model, model_type)
            if config.provider != "unknown":
                return config.provider
        except KeyError:
            pass

        # Fallback: infer from model ID patterns
        resolved = cls.resolve_llm(model) if model_type == "llm" else model
        provider = _infer_provider_from_model_id(resolved, model_type)

        if provider == "unknown":
            raise ValueError(
                f"Unable to auto-detect provider for model '{model}' (type: {model_type}).\n"
                f"Add model to config.json model_registry with explicit provider field."
            )

        return provider

    # ========================================================================
    # RESOLUTION METHODS (existing API)
    # ========================================================================

    @classmethod
    def resolve_llm(cls, alias: str) -> str:
        """
        Resolve LLM model alias to full model name.

        Args:
            alias: Model alias (e.g., "haiku", "sonnet")

        Returns:
            Full model name (e.g., "claude-haiku-4-5-20251001")
            Returns alias as-is if not found in registry.
        """
        _ensure_config_loaded()
        entry = _LLM_MODELS_RAW.get(alias)

        if entry is None:
            return alias  # Not found, return as-is

        resolved = _extract_model_id(entry)

        if resolved != alias:
            logger.debug(f"Resolved LLM alias: {alias} → {resolved}")

        return resolved

    @classmethod
    def resolve_embedding(cls, alias: str) -> str:
        """Resolve embedding model alias to full model name."""
        _ensure_config_loaded()
        entry = _EMBEDDING_MODELS_RAW.get(alias)

        if entry is None:
            return alias

        resolved = _extract_model_id(entry)

        if resolved != alias:
            logger.debug(f"Resolved embedding alias: {alias} → {resolved}")

        return resolved

    @classmethod
    def resolve_reranker(cls, alias: str) -> str:
        """Resolve reranker model alias to full model name."""
        _ensure_config_loaded()
        entry = _RERANKER_MODELS_RAW.get(alias)

        if entry is None:
            return alias

        resolved = _extract_model_id(entry)

        if resolved != alias:
            logger.debug(f"Resolved reranker alias: {alias} → {resolved}")

        return resolved

    @classmethod
    def get_embedding_dimensions(cls, model: str) -> int:
        """
        Get embedding dimensions for a model.

        Args:
            model: Model name or alias

        Returns:
            Embedding dimensions (default: 4096)
        """
        try:
            config = cls.get_model_config(model, "embedding")
            if isinstance(config, EmbeddingModelData):
                return config.dimensions
        except KeyError:
            pass

        # Fallback to legacy dimensions dict
        _ensure_config_loaded()
        resolved = cls.resolve_embedding(model)
        if resolved in _EMBEDDING_DIMENSIONS:
            return _EMBEDDING_DIMENSIONS[resolved]
        if model in _EMBEDDING_DIMENSIONS:
            return _EMBEDDING_DIMENSIONS[model]

        # Default
        logger.warning(f"Embedding dimensions not found for '{model}', defaulting to 4096")
        return 4096

    @classmethod
    def is_local_embedding(cls, model: str) -> bool:
        """
        Check if embedding model is local (no API key required).

        Args:
            model: Model name or alias

        Returns:
            True if model is local
        """
        try:
            config = cls.get_model_config(model, "embedding")
            if isinstance(config, EmbeddingModelData):
                return config.is_local
        except KeyError:
            pass

        # Fallback: infer from patterns
        resolved = cls.resolve_embedding(model)
        local_patterns = ["BAAI/", "sentence-transformers/", "intfloat/"]
        return any(resolved.startswith(pattern) for pattern in local_patterns)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @classmethod
    def list_models(cls, model_type: str = "all") -> Dict[str, Any]:
        """
        List all models of given type.

        Args:
            model_type: Type of models to list ("llm", "embedding", "reranker", "all")

        Returns:
            Dictionary of alias -> model_id
        """
        _ensure_config_loaded()

        if model_type == "llm":
            return {k: _extract_model_id(v) for k, v in _LLM_MODELS_RAW.items()}
        elif model_type == "embedding":
            return {k: _extract_model_id(v) for k, v in _EMBEDDING_MODELS_RAW.items()}
        elif model_type == "reranker":
            return {k: _extract_model_id(v) for k, v in _RERANKER_MODELS_RAW.items()}
        elif model_type == "all":
            return {
                "llm": cls.list_models("llm"),
                "embedding": cls.list_models("embedding"),
                "reranker": cls.list_models("reranker"),
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @classmethod
    def has_pricing(cls, model: str, model_type: str = "llm") -> bool:
        """
        Check if model has pricing data in config.

        Args:
            model: Model alias or ID
            model_type: "llm" or "embedding"

        Returns:
            True if pricing is available and non-zero
        """
        pricing = cls.get_pricing(model, model_type)
        return pricing["input"] > 0 or pricing["output"] > 0


# Example usage
if __name__ == "__main__":
    print("=== Model Registry Examples ===\n")

    # Example 1: Resolve LLM aliases
    print("1. LLM model resolution...")
    llm_aliases = ["haiku", "sonnet", "gpt-4o-mini", "minimax-m2"]
    for alias in llm_aliases:
        resolved = ModelRegistry.resolve_llm(alias)
        print(f"   {alias:20s} → {resolved}")

    # Example 2: Get provider (from config)
    print("\n2. Provider detection (from config)...")
    for alias in llm_aliases:
        try:
            provider = ModelRegistry.get_provider(alias, "llm")
            print(f"   {alias:20s} → {provider}")
        except ValueError as e:
            print(f"   {alias:20s} → ERROR: {e}")

    # Example 3: Get pricing (from config)
    print("\n3. Pricing lookup (from config)...")
    for alias in llm_aliases:
        pricing = ModelRegistry.get_pricing(alias, "llm")
        print(f"   {alias:20s} → ${pricing['input']:.2f} in / ${pricing['output']:.2f} out")

    # Example 4: Embedding dimensions
    print("\n4. Embedding dimensions...")
    embed_aliases = ["bge-m3", "text-embedding-3-large", "qwen3-embedding"]
    for alias in embed_aliases:
        dims = ModelRegistry.get_embedding_dimensions(alias)
        is_local = ModelRegistry.is_local_embedding(alias)
        local_str = " (local)" if is_local else " (cloud)"
        print(f"   {alias:25s} → dims={dims}{local_str}")

    print("\n=== All examples completed ===")
