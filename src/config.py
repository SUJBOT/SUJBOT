"""
Unified configuration system for SUJBOT.

**Single Source of Truth: config.json file**
- ALL configuration in one JSON file
- Strict validation on startup
- Hierarchical structure for better organization

## Usage

```python
from src.config import get_config, ModelConfig

config = get_config()
model_config = ModelConfig.from_config(config)
```
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


