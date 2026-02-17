"""
OpenAI GPT provider implementation.

Translates Anthropic format <-> OpenAI format.
Uses shared helpers from openai_compat for common conversions.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

import openai
from langsmith.wrappers import wrap_openai

from .base import BaseProvider, ProviderResponse
from .openai_compat import (
    STOP_REASON_MAP,
    convert_response_to_anthropic,
    convert_system_to_string,
    convert_assistant_blocks_to_openai,
    convert_tools_to_openai,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI GPT provider with format translation.

    Supports OpenAI models:
    - gpt-4o (most capable)
    - gpt-4o-mini (balanced, cost-effective)
    - o-series (reasoning models: o1, o3, o4)
    """

    def __init__(self, api_key: str, model: str):
        if not api_key or not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
            raise ValueError("Invalid OpenAI API key format (should start with sk- or sk-proj-)")

        self._client = wrap_openai(openai.OpenAI(api_key=api_key))
        self.model = model
        logger.info(f"OpenAIProvider initialized: model={model}")

    def _build_api_params(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build OpenAI API parameters with model-specific handling."""
        openai_messages = self._convert_messages_to_openai(messages, system)
        openai_tools = convert_tools_to_openai(tools) if tools else None

        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            **kwargs,
        }

        if stream:
            api_params["stream"] = True

        if openai_tools:
            api_params["tools"] = openai_tools

        # O-series reasoning models only support default temperature (1.0)
        if not self._is_o_series():
            api_params["temperature"] = temperature

        # Use correct parameter name based on model
        if self._is_o_series():
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens

        return api_params

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> ProviderResponse:
        api_params = self._build_api_params(
            messages, tools, system, max_tokens, temperature, **kwargs
        )
        response = self._client.chat.completions.create(**api_params)
        content = convert_response_to_anthropic(response)
        raw_reason = response.choices[0].finish_reason
        stop_reason = STOP_REASON_MAP.get(raw_reason, "end_turn")

        return ProviderResponse(
            content=content,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
            },
            model=response.model,
        )

    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Iterator[Any]:
        api_params = self._build_api_params(
            messages, tools, system, max_tokens, temperature, stream=True, **kwargs
        )
        return self._client.chat.completions.create(**api_params)

    def _convert_messages_to_openai(
        self, messages: List[Dict[str, Any]], system: List[Dict[str, Any]] | str
    ) -> List[Dict[str, Any]]:
        """Convert Anthropic messages to OpenAI format."""
        openai_msgs: List[Dict[str, Any]] = []

        # System message
        openai_msgs.append({"role": "system", "content": convert_system_to_string(system)})

        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            # OpenAI tool result — content must be a string
                            tool_content = block["content"]
                            if isinstance(tool_content, list):
                                tool_content = "\n".join(
                                    sub.get("text", "")
                                    for sub in tool_content
                                    if isinstance(sub, dict) and sub.get("type") == "text"
                                ) or "See attached content."
                            openai_msgs.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": str(tool_content),
                                }
                            )
                        elif block.get("type") == "text":
                            openai_msgs.append({"role": "user", "content": block["text"]})
                else:
                    openai_msgs.append({"role": "user", "content": content})

            elif msg["role"] == "assistant":
                content = msg["content"]
                if isinstance(content, list):
                    openai_msgs.append(convert_assistant_blocks_to_openai(content))
                else:
                    openai_msgs.append({"role": "assistant", "content": content})

        return openai_msgs

    def supports_feature(self, feature: str) -> bool:
        return feature in {"tool_use", "streaming"}

    def _is_o_series(self) -> bool:
        """Check if current model is an O-series reasoning model."""
        model_lower = self.model.lower()
        return any(pattern in model_lower for pattern in ["o1", "o3", "o4"])

    def get_model_name(self) -> str:
        return self.model

    def set_model(self, model: str) -> None:
        valid_patterns = {"gpt-", "o1", "o3", "o4"}
        if not any(p in model.lower() for p in valid_patterns):
            raise ValueError(f"Invalid OpenAI model: {model}")
        old_model = self.model
        self.model = model
        logger.info(f"Model changed: {old_model} -> {model}")

    def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
    ) -> Optional[int]:
        return self._tiktoken_estimate(messages, tools, system)

    def get_provider_name(self) -> str:
        return "openai"
