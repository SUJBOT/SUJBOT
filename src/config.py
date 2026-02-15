"""
Unified configuration system for RAG pipeline.

## Configuration Philosophy (UPDATED 2025-11-10)

**Single Source of Truth: config.json file**
- ALL configuration in one JSON file
- Strict validation on startup - NO fallbacks, NO defaults
- If any required parameter is missing, application exits with error
- Hierarchical structure for better organization

## Migration from .env to config.json

Previous system used .env with fallback values. New system:
1. Copy config.json.example to config.json
2. Fill in ALL required values
3. Application validates config on startup
4. Missing values = immediate error, application stops

## Pipeline Phases

- PHASE 1: Document Extraction (Docling)
- PHASE 2: Summarization (Generic summaries, 150 chars) → Model from config.json
- PHASE 3: Chunking (Hierarchical with SAC, 512 tokens)
- PHASE 4: Embedding (Multi-layer embeddings) → Model from config.json

## Usage

All configuration is loaded from config.json automatically:

```python
from src.config import get_config, SummarizationConfig, EmbeddingConfig

# Load validated config (fails if config.json is invalid or missing)
config = get_config()

# Access specific sections
summary_config = SummarizationConfig.from_config(config.summarization)
embed_config = EmbeddingConfig.from_config(config.embedding)
```

All configuration classes can be imported by other modules.
"""

import logging
from typing import Optional
from dataclasses import dataclass

# Import JSON config loader and schema
from src.config_schema import (
    RootConfig,
    load_config as load_json_config,
)

logger = logging.getLogger(__name__)

# Global config instance - loaded once at module import
_CONFIG: Optional[RootConfig] = None


def get_config(reload: bool = False) -> RootConfig:
    """
    Get validated configuration from config.json.

    This function loads and validates config.json on first call,
    then caches the result for subsequent calls.

    Args:
        reload: Force reload config.json (default: False)

    Returns:
        Validated RootConfig instance

    Raises:
        FileNotFoundError: If config.json does not exist
        ValueError: If validation fails or required fields are missing
    """
    global _CONFIG

    if _CONFIG is None or reload:
        try:
            _CONFIG = load_json_config()
            logger.info("Configuration loaded and validated successfully from config.json")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    return _CONFIG


@dataclass
class ModelConfig:
    """Central model configuration loaded from config.json."""

    # LLM Configuration
    llm_provider: str  # "claude" or "openai" or "google"
    llm_model: str  # e.g., "claude-sonnet-4.5", "gpt-4o-mini"

    # Embedding Configuration
    embedding_provider: str  # "voyage", "openai", "huggingface"
    embedding_model: str  # e.g., "kanon-2", "text-embedding-3-large", "bge-m3"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    @classmethod
    def from_config(cls, config: RootConfig) -> "ModelConfig":
        """
        Load configuration from validated RootConfig.

        Args:
            config: Validated RootConfig instance

        Returns:
            ModelConfig instance
        """
        # Import ModelRegistry for provider detection
        from src.utils.model_registry import ModelRegistry

        # Auto-detect providers from model names if not explicitly set
        llm_provider = config.models.llm_provider
        if llm_provider is None:
            llm_provider = ModelRegistry.get_provider(config.models.llm_model, "llm")

        embedding_provider = config.models.embedding_provider
        if embedding_provider is None:
            embedding_provider = ModelRegistry.get_provider(config.models.embedding_model, "embedding")

        # Validate required API keys based on selected providers
        required_keys = {}
        if llm_provider in ("claude", "anthropic"):
            required_keys["ANTHROPIC_API_KEY"] = config.api_keys.anthropic_api_key
        elif llm_provider == "openai":
            required_keys["OPENAI_API_KEY"] = config.api_keys.openai_api_key
        elif llm_provider == "google":
            required_keys["GOOGLE_API_KEY"] = config.api_keys.google_api_key

        if embedding_provider == "voyage":
            required_keys["VOYAGE_API_KEY"] = config.api_keys.voyage_api_key
        elif embedding_provider == "openai" and llm_provider != "openai":
            required_keys["OPENAI_API_KEY"] = config.api_keys.openai_api_key

        # Check for missing keys
        missing_keys = [key for key, value in required_keys.items() if not value]
        if missing_keys:
            raise ValueError(
                f"Missing required API keys for selected providers:\n"
                f"  LLM Provider: {llm_provider}\n"
                f"  Embedding Provider: {embedding_provider}\n"
                f"  Missing keys: {', '.join(missing_keys)}\n\n"
                f"Please set these environment variables in your .env file:\n"
                f"  " + "\n  ".join(f"{key}=your_api_key_here" for key in missing_keys) + "\n\n"
                f"See .env.example for reference."
            )

        return cls(
            # LLM Configuration
            llm_provider=llm_provider,
            llm_model=config.models.llm_model,
            # Embedding Configuration
            embedding_provider=embedding_provider,
            embedding_model=config.models.embedding_model,
            # API Keys
            anthropic_api_key=config.api_keys.anthropic_api_key,
            openai_api_key=config.api_keys.openai_api_key,
            voyage_api_key=config.api_keys.voyage_api_key,
            google_api_key=config.api_keys.google_api_key,
        )

    def get_llm_config(self) -> dict:
        """Get LLM provider configuration."""
        if self.llm_provider in ("claude", "anthropic"):
            return {
                "provider": "claude",
                "model": self.llm_model,
                "api_key": self.anthropic_api_key,
            }
        elif self.llm_provider == "openai":
            return {"provider": "openai", "model": self.llm_model, "api_key": self.openai_api_key}
        elif self.llm_provider == "google":
            return {"provider": "google", "model": self.llm_model, "api_key": self.google_api_key}
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_embedding_config(self) -> dict:
        """Get embedding provider configuration."""
        if self.embedding_provider == "voyage":
            return {
                "provider": "voyage",
                "model": self.embedding_model,
                "api_key": self.voyage_api_key,
            }
        elif self.embedding_provider == "openai":
            return {
                "provider": "openai",
                "model": self.embedding_model,
                "api_key": self.openai_api_key,
            }
        elif self.embedding_provider == "huggingface":
            return {
                "provider": "huggingface",
                "model": self.embedding_model,
                "api_key": None,  # Local models
            }
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")


# Import ModelRegistry for centralized model management
from src.utils.model_registry import ModelRegistry


def resolve_model_alias(model_name: str) -> str:
    """
    Resolve model alias to full model name using centralized ModelRegistry.

    This function is kept for backward compatibility. All model aliases are now
    managed in utils.model_registry.ModelRegistry.

    Args:
        model_name: Model name or alias (e.g., "haiku", "sonnet", "gpt-4o-mini")

    Returns:
        Full model name (e.g., "claude-haiku-4-5-20251001")
    """
    # Try LLM models first
    if model_name in ModelRegistry.LLM_MODELS:
        return ModelRegistry.resolve_llm(model_name)
    # Then try embedding models
    elif model_name in ModelRegistry.EMBEDDING_MODELS:
        return ModelRegistry.resolve_embedding(model_name)
    # Fallback: return as-is
    else:
        return model_name



@dataclass
class EmbeddingConfig:
    """
    Unified configuration for embedding generation (PHASE 4).
    """

    # Model selection
    provider: Optional[str] = None
    model: Optional[str] = None

    # Research-backed parameters
    batch_size: int = 64
    normalize: bool = True

    # Multi-layer indexing
    enable_multi_layer: bool = True

    # Model metadata
    dimensions: Optional[int] = None

    # Performance optimization
    cache_enabled: bool = True
    cache_max_size: int = 1000

    def __post_init__(self):
        """Load model config from global config if not provided and validate."""
        if self.provider is None or self.model is None:
            config = get_config()
            model_config = ModelConfig.from_config(config)
            self.provider = model_config.embedding_provider
            self.model = model_config.embedding_model

        # Validate parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive if specified, got {self.dimensions}")
        if self.cache_max_size <= 0:
            raise ValueError(f"cache_max_size must be positive, got {self.cache_max_size}")
        if self.provider is not None and self.provider not in ["voyage", "openai", "huggingface", "deepinfra"]:
            raise ValueError(
                f"provider must be 'voyage', 'openai', 'huggingface', or 'deepinfra', got {self.provider}"
            )

    @classmethod
    def from_config(cls, embedding_config) -> "EmbeddingConfig":
        """
        Load configuration from validated EmbeddingConfig schema.

        Args:
            embedding_config: Validated embedding config from RootConfig

        Returns:
            EmbeddingConfig instance
        """
        return cls(
            batch_size=embedding_config.batch_size,
            normalize=embedding_config.normalize,
            enable_multi_layer=True,  # Always enabled
            cache_enabled=embedding_config.cache_enabled,
            cache_max_size=embedding_config.cache_size,
        )


