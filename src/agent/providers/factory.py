"""
Provider factory for creating appropriate provider instance.

SSOT: Provider detection now uses ModelRegistry.get_provider() which reads
from config.json model_registry section. Pattern matching is only used as
fallback for models not in config.

This simplifies adding new models - just add to config.json with explicit
provider field, no code changes needed.
"""

import logging
import os
from typing import Optional

from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .vllm_provider import VLLMProvider

# GeminiProvider uses lazy import to avoid breaking other providers
# when google-generativeai SDK is incompatible

logger = logging.getLogger(__name__)


def create_provider(
    model: str,
    api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
) -> BaseProvider:
    """
    Create appropriate provider based on model name.

    Automatically detects provider from model name and loads API key from
    environment if not provided.

    Args:
        model: Model name or alias (e.g., "claude-haiku-4-5", "gpt-4o-mini", "gemini-flash")
        api_key: API key for the provider (deprecated, use provider-specific keys)
        anthropic_api_key: Anthropic API key (optional, defaults to ANTHROPIC_API_KEY env var)
        openai_api_key: OpenAI API key (optional, defaults to OPENAI_API_KEY env var)
        google_api_key: Google API key (optional, defaults to GOOGLE_API_KEY env var)

    Returns:
        Provider instance (AnthropicProvider, OpenAIProvider, GeminiProvider, or VLLMProvider)

    Raises:
        ValueError: If provider cannot be determined or API key is missing

    Examples:
        >>> # Using model registry aliases
        >>> provider = create_provider("haiku")  # Uses ANTHROPIC_API_KEY from env
        >>> provider = create_provider("gpt-4o-mini")  # Uses OPENAI_API_KEY from env
        >>>
        >>> # Using full model names
        >>> provider = create_provider("claude-haiku-4-5-20251001")
        >>> provider = create_provider("gpt-4o-mini")
        >>>
        >>> # Providing API keys explicitly
        >>> provider = create_provider("haiku", anthropic_api_key="sk-ant-...")
        >>> provider = create_provider("gpt-4o-mini", openai_api_key="sk-...")
    """
    # Import here to avoid circular dependency
    try:
        from ...utils.model_registry import ModelRegistry
    except ImportError:
        # Fallback if model_registry not available
        logger.warning("ModelRegistry not available, using direct model name")
        resolved_model = model
        provider_name = detect_provider_from_model(model)
    else:
        # Resolve model alias to full model ID
        resolved_model = ModelRegistry.resolve_llm(model)
        # Look up provider from ORIGINAL alias first (handles same-ID, different-provider entries)
        try:
            provider_name = ModelRegistry.get_provider(model, "llm")
        except (ValueError, KeyError):
            provider_name = ModelRegistry.get_provider(resolved_model, "llm")

    logger.info(f"Creating provider: model={resolved_model}, provider={provider_name}")

    # Create provider instance
    if provider_name == "anthropic":
        # Get API key (priority: explicit param > env var)
        key = anthropic_api_key or api_key or os.getenv("ANTHROPIC_API_KEY")

        if not key:
            raise ValueError(
                "Anthropic API key required for Claude models.\n"
                "Set ANTHROPIC_API_KEY environment variable or pass anthropic_api_key parameter.\n"
                "Example: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        return AnthropicProvider(api_key=key, model=resolved_model)

    elif provider_name == "openai":
        # Get API key (priority: explicit param > env var)
        key = openai_api_key or api_key or os.getenv("OPENAI_API_KEY")

        if not key:
            raise ValueError(
                "OpenAI API key required for GPT models.\n"
                "Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.\n"
                "Example: export OPENAI_API_KEY=sk-..."
            )

        return OpenAIProvider(api_key=key, model=resolved_model)

    elif provider_name == "google":
        # Get API key (priority: explicit param > env var)
        key = google_api_key or api_key or os.getenv("GOOGLE_API_KEY")

        if not key:
            raise ValueError(
                "Google API key required for Gemini models.\n"
                "Set GOOGLE_API_KEY environment variable or pass google_api_key parameter.\n"
                "Example: export GOOGLE_API_KEY=AIza..."
            )

        # Lazy import to avoid breaking other providers when SDK is incompatible
        try:
            from .gemini_provider import GeminiProvider
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Gemini provider unavailable due to SDK incompatibility: {e}\n"
                "Install compatible google-generativeai version or use a different provider."
            ) from e

        return GeminiProvider(api_key=key, model=resolved_model)

    elif provider_name == "local_llm":
        # Local 30B LLM via vLLM (OpenAI-compatible API)
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:18080/v1")
        return VLLMProvider(
            base_url=base_url,
            model=resolved_model,
            provider_name="local_llm",
        )

    elif provider_name == "local_llm_8b":
        # Local 8B LLM (Qwen3-VL-8B-Instruct-FP8 on gx10-fa34, port 18082)
        base_url = os.getenv("LOCAL_LLM_8B_BASE_URL", "http://localhost:18082/v1")
        return VLLMProvider(
            base_url=base_url,
            model=resolved_model,
            provider_name="local_llm_8b",
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider_name} for model: {model}\n"
            f"Supported providers: anthropic (Claude), openai (GPT-4o/o-series), "
            f"google (Gemini), local_llm (local 30B vLLM), "
            f"local_llm_8b (local 8B vLLM)"
        )


def detect_provider_from_model(model: str) -> str:
    """
    Detect provider from model name.

    SSOT: Delegates to ModelRegistry.get_provider() which reads from config.json.
    Pattern matching is used as fallback for models not in config.

    Args:
        model: Model name or alias

    Returns:
        Provider name ("anthropic", "openai", "google", "local_llm", "local_llm_8b")

    Raises:
        ValueError: If provider cannot be determined
    """
    try:
        from ...utils.model_registry import ModelRegistry
        return ModelRegistry.get_provider(model, "llm")
    except ImportError as e:
        logger.debug(f"ModelRegistry unavailable, using pattern fallback: {e}")
    except (ValueError, KeyError) as e:
        logger.debug(f"ModelRegistry lookup failed for '{model}', using pattern fallback: {e}")

    # Fallback: pattern matching for models not in config
    model_lower = model.lower()

    # Anthropic patterns
    if any(pattern in model_lower for pattern in ["claude", "haiku", "sonnet", "opus"]):
        return "anthropic"

    # OpenAI patterns (includes o-series reasoning models)
    if any(pattern in model_lower for pattern in ["gpt-", "o1", "o3", "o4"]):
        return "openai"

    # Google Gemini patterns
    if "gemini" in model_lower:
        return "google"

    raise ValueError(
        f"Cannot determine provider for model: {model}\n"
        f"Add model to config.json model_registry with explicit 'provider' field,\n"
        f"or use a model name containing: 'claude', 'haiku', 'sonnet', 'opus' (Anthropic), "
        f"'gpt-', 'o1', 'o3', 'o4' (OpenAI), 'gemini' (Google), "
        f"or set provider to 'local_llm'/'local_llm_8b' for local vLLM servers"
    )
