"""
Base provider interface for LLM abstraction.

PRAGMATIC DESIGN:
- Minimal abstraction (only what's needed)
- Provider-specific features exposed via supports_feature()
- No over-engineering (no complex strategy patterns)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ProviderResponse:
    """
    Unified response format across providers.

    Uses Anthropic format as the canonical structure since our codebase
    was built around it. OpenAI responses are converted to this format.
    """

    content: List[Dict[str, Any]]  # Anthropic content blocks format
    stop_reason: str  # "end_turn", "tool_use", "max_tokens", etc.
    usage: Dict[str, int]  # {input_tokens, output_tokens, cache_read, cache_creation}
    model: str

    def __post_init__(self):
        """Validate response structure at construction time."""
        # Validate content blocks have 'type' key
        for i, block in enumerate(self.content):
            if not isinstance(block, dict):
                raise ValueError(f"Content block {i} must be a dict, got {type(block)}")
            if "type" not in block:
                raise ValueError(f"Content block {i} missing 'type' key: {block}")

        # Validate required usage keys
        required_usage_keys = ["input_tokens", "output_tokens"]
        for key in required_usage_keys:
            if key not in self.usage:
                raise ValueError(f"Usage dict missing required key '{key}': {self.usage}")

        # Validate token counts are non-negative integers
        for key, value in self.usage.items():
            if not isinstance(value, int):
                raise ValueError(f"Usage {key} must be an integer, got {type(value)}: {value}")
            if value < 0:
                raise ValueError(f"Usage {key} must be non-negative, got {value}")

    @property
    def text(self) -> str:
        """
        Extract text content from response.

        Returns:
            Concatenated text from all text blocks
        """
        return "".join(block["text"] for block in self.content if block.get("type") == "text")

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """
        Extract tool calls from response.

        Returns:
            List of tool_use blocks
        """
        return [block for block in self.content if block.get("type") == "tool_use"]


class BaseProvider(ABC):
    """
    Abstract base for LLM providers.

    PRAGMATIC APPROACH:
    - Keep it simple: Only methods we actually use
    - No premature abstraction: Add methods as needed
    - Provider-specific features via supports_feature()

    Design Philosophy:
    - AgentCore is provider-agnostic (uses this interface)
    - Providers handle format translation internally
    - Graceful degradation for missing features
    """

    @abstractmethod
    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ProviderResponse:
        """
        Create a message synchronously (non-streaming).

        Args:
            messages: Conversation history in Anthropic format
            tools: Tool definitions in Anthropic format (provider translates if needed)
            system: System prompt (structured list or string)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters

        Returns:
            ProviderResponse with content, usage, etc.

        Raises:
            Provider-specific API errors (anthropic.APIError, openai.OpenAIError, etc.)
        """
        pass

    @abstractmethod
    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Iterator[Any]:
        """
        Create a message with streaming (yields events).

        PRAGMATIC NOTE:
        Returns provider-specific stream object (not unified). AgentCore
        type-checks provider and handles streaming appropriately.

        This is pragmatic because each provider has very different streaming
        formats and unifying them would add complexity without benefit.

        Args:
            messages: Conversation history in Anthropic format
            tools: Tool definitions in Anthropic format
            system: System prompt (structured list or string)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Provider-specific parameters

        Returns:
            Iterator yielding provider-specific events

        Raises:
            Provider-specific API errors
        """
        pass

    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """
        Check if provider supports a feature.

        Supported features:
        - "prompt_caching": Anthropic-style prompt caching (90% cost savings)
        - "streaming": Streaming responses
        - "tool_use": Function/tool calling
        - "structured_system": System prompt as structured blocks (vs simple string)

        Args:
            feature: Feature name to check

        Returns:
            True if feature is supported

        Example:
            >>> if provider.supports_feature("prompt_caching"):
            >>>     # Add cache control
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get current model name.

        Returns:
            Model identifier (e.g., "claude-haiku-4-5", "gpt-4o-mini")
        """
        pass

    @abstractmethod
    def set_model(self, model: str) -> None:
        """
        Change model (for /model CLI command).

        Args:
            model: New model name

        Raises:
            ValueError: If model is invalid for this provider
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        pass

    def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
    ) -> Optional[int]:
        """
        Count input tokens before sending a request (pre-call estimation).

        Override in subclasses for provider-specific counting.
        Default returns None (no counting available).

        Args:
            messages: Conversation history in Anthropic format
            tools: Tool definitions in Anthropic format
            system: System prompt (structured list or string)

        Returns:
            Estimated input token count, or None if not available
        """
        return None

    @staticmethod
    def _tiktoken_estimate(
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
    ) -> Optional[int]:
        """
        Estimate input tokens using tiktoken (cl100k_base).

        Shared helper for OpenAI-compatible providers (SSOT).
        Approximation only — does not account for provider-specific overhead.

        Args:
            messages: Conversation history
            tools: Tool definitions
            system: System prompt

        Returns:
            Approximate token count, or None if tiktoken unavailable
        """
        try:
            import json

            import tiktoken
        except ImportError:
            return None

        encoding = tiktoken.get_encoding("cl100k_base")
        total = 0

        # System prompt
        if isinstance(system, str):
            total += len(encoding.encode(system))
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += len(encoding.encode(block.get("text", "")))

        # Messages
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "") or block.get("content", "")
                        if isinstance(text, str):
                            total += len(encoding.encode(text))

        # Tools
        if tools:
            total += len(encoding.encode(json.dumps(tools)))

        return total
