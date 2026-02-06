"""
DeepInfra LLM Provider using OpenAI-compatible API.

Supports Llama, Qwen, MiniMax and other models via DeepInfra's OpenAI-compatible endpoint.
Model configuration is in config.json (SSOT).
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Iterator

from openai import OpenAI
from langsmith.wrappers import wrap_openai

from .base import BaseProvider, ProviderResponse
from ...exceptions import APIKeyError

logger = logging.getLogger(__name__)


class DeepInfraProvider(BaseProvider):
    """
    DeepInfra provider for open-source LLMs.

    Uses OpenAI-compatible API format (https://api.deepinfra.com/v1/openai).

    Supported models are configured in config.json model_registry section.
    Common models include:
    - MiniMaxAI/MiniMax-M2 (high context, recommended for complex tasks)
    - meta-llama/Llama-4-Scout-17B-16E-Instruct (recommended for agents)
    - meta-llama/Meta-Llama-3.1-70B-Instruct
    - meta-llama/Meta-Llama-3.1-8B-Instruct (lighter model)
    - Qwen/Qwen2.5-72B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ):
        """
        Initialize DeepInfra provider.

        Args:
            api_key: DeepInfra API key (defaults to DEEPINFRA_API_KEY env var)
            model: Model identifier (default: meta-llama/Llama-4-Scout-17B-16E-Instruct)

        Raises:
            ValueError: If API key is missing
        """
        self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self.api_key:
            raise APIKeyError(
                "DEEPINFRA_API_KEY required. Set in .env file or pass to constructor.",
                details={"provider": "deepinfra"},
            )

        # Model validation is handled by config.json (SSOT)
        # Log warning if model not in config, but don't reject (API will validate)
        try:
            from ...utils.model_registry import ModelRegistry

            config = ModelRegistry.get_model_config(model, "llm")
            if config.provider != "deepinfra":
                logger.warning(
                    f"Model {model} is configured with provider '{config.provider}', not 'deepinfra'"
                )
        except (ImportError, KeyError):
            logger.debug(f"Model {model} not found in config.json model_registry, using as-is")

        self.model = model

        # Initialize OpenAI client with DeepInfra base URL (wrapped for LangSmith tracing)
        self.client = wrap_openai(
            OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepinfra.com/v1/openai",
                timeout=60.0,
                max_retries=3,
            )
        )

        logger.info(f"DeepInfraProvider initialized: {model}")

    @staticmethod
    def _convert_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict]]:
        """Convert Anthropic tool definitions to OpenAI function calling format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in tools
        ]

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs,
    ) -> ProviderResponse:
        """
        Create chat completion (non-streaming).

        Args:
            messages: Conversation messages in Anthropic format
            tools: Tool definitions in Anthropic format
            system: System prompt (string or list of dicts)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            ProviderResponse with content, usage, etc.
        """
        formatted_messages = self._format_messages(messages, system)
        openai_tools = self._convert_tools_to_openai(tools)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=openai_tools,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"DeepInfra API error: {e}")
            raise

        return self._convert_response(response)

    def stream_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs,
    ) -> Iterator:
        """
        Stream chat completion.

        Args:
            messages: Conversation messages in Anthropic format
            tools: Tool definitions in Anthropic format
            system: System prompt (string or list of dicts)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Iterator of streaming response chunks
        """
        formatted_messages = self._format_messages(messages, system)
        openai_tools = self._convert_tools_to_openai(tools)

        return self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            tools=openai_tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

    def _format_messages(self, messages: List[Dict], system: Any) -> List[Dict]:
        """
        Convert Anthropic message format to OpenAI format.

        Handles:
        - Text-only and multimodal (vision) content
        - Tool use blocks (assistant → OpenAI tool_calls)
        - Tool result blocks (user → OpenAI tool messages)
        - Image blocks (Anthropic base64 → OpenAI image_url)

        Args:
            messages: Anthropic-format messages
            system: System prompt (string or list of dicts)

        Returns:
            OpenAI-format messages
        """
        formatted = []

        # Add system message
        if system:
            if isinstance(system, str):
                formatted.append({"role": "system", "content": system})
            elif isinstance(system, list):
                # Extract text from Anthropic structured format
                system_text = "".join(
                    block.get("text", "")
                    for block in system
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if system_text:
                    formatted.append({"role": "system", "content": system_text})

        # Convert messages
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                formatted.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    self._format_assistant_message(content, formatted)
                elif role == "user":
                    self._format_user_message(content, formatted)

        return formatted

    def _format_assistant_message(self, content: List[Dict], formatted: List[Dict]) -> None:
        """Convert Anthropic assistant content blocks to OpenAI format."""
        text_parts = []
        tool_calls = []
        image_blocks = []

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif block.get("type") == "image":
                image_blocks.append(block)

        if image_blocks:
            # Multimodal assistant message (rare but handle it)
            openai_content = self._convert_multimodal_content(content)
            if openai_content:
                formatted.append({"role": "assistant", "content": openai_content})
        elif tool_calls:
            msg = {"role": "assistant", "tool_calls": tool_calls}
            msg["content"] = " ".join(text_parts) if text_parts else None
            formatted.append(msg)
        elif text_parts:
            formatted.append({"role": "assistant", "content": " ".join(text_parts)})

    def _format_user_message(self, content: List[Dict], formatted: List[Dict]) -> None:
        """Convert Anthropic user content blocks to OpenAI format."""
        has_images = any(
            isinstance(block, dict) and block.get("type") == "image" for block in content
        )
        has_tool_results = any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        )

        if has_tool_results:
            # Convert tool_result blocks to OpenAI tool messages
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        # Multimodal tool result — convert images, extract text
                        openai_parts = self._convert_multimodal_content(tool_content)
                        formatted.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": openai_parts if openai_parts else "",
                            }
                        )
                    else:
                        formatted.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": str(tool_content),
                            }
                        )
                elif block.get("type") == "text":
                    formatted.append({"role": "user", "content": block.get("text", "")})
        elif has_images:
            # Multimodal user message
            openai_content = self._convert_multimodal_content(content)
            if openai_content:
                formatted.append({"role": "user", "content": openai_content})
        else:
            # Text-only user message
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text:
                formatted.append({"role": "user", "content": text})

    def _convert_multimodal_content(self, content_blocks: List[Dict]) -> List[Dict]:
        """
        Convert Anthropic multimodal content blocks to OpenAI vision format.

        Anthropic format:
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
            {"type": "text", "text": "caption"}

        OpenAI format:
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            {"type": "text", "text": "caption"}
        """
        openai_blocks = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    openai_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        }
                    )
            elif block.get("type") == "text":
                openai_blocks.append(
                    {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                )
            elif block.get("type") == "tool_result":
                # Tool results with multimodal content — extract text
                tool_content = block.get("content", "")
                if isinstance(tool_content, str):
                    openai_blocks.append({"type": "text", "text": tool_content})
                elif isinstance(tool_content, list):
                    openai_blocks.extend(self._convert_multimodal_content(tool_content))

        return openai_blocks

    def _convert_response(self, response) -> ProviderResponse:
        """
        Convert OpenAI response to Anthropic ProviderResponse format.

        Args:
            response: OpenAI API response

        Returns:
            ProviderResponse in Anthropic format

        Raises:
            ValueError: If response has no choices
        """
        if not response.choices:
            raise ValueError("Empty response from DeepInfra API - no choices returned")

        choice = response.choices[0]
        message = choice.message

        # Build content blocks
        content_blocks = []

        if message.content:
            content_blocks.append({"type": "text", "text": message.content})

        # Convert tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    parsed_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(
                        "Failed to parse tool arguments from DeepInfra: %s. " "Tool: %s, Args: %s",
                        e,
                        tool_call.function.name,
                        (
                            tool_call.function.arguments[:200]
                            if tool_call.function.arguments
                            else "None"
                        ),
                    )
                    continue
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": parsed_args,
                    }
                )

        # Map finish reason
        stop_reason_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}

        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "output_tokens": (
                getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            ),
            "cache_read_tokens": 0,  # DeepInfra doesn't support caching
            "cache_creation_tokens": 0,
        }

        return ProviderResponse(
            content=content_blocks,
            stop_reason=stop_reason_map.get(choice.finish_reason, "end_turn"),
            usage=usage,
            model=self.model,
        )

    def supports_feature(self, feature: str) -> bool:
        """
        Check feature support.

        Args:
            feature: Feature name ("streaming", "tool_use", "prompt_caching", etc.)

        Returns:
            True if feature is supported
        """
        supported_features = {
            "streaming": True,
            "tool_use": True,
            "prompt_caching": False,  # DeepInfra doesn't support Anthropic-style caching
            "structured_system": False,
            "vision": "VL" in self.model or "vl" in self.model.lower(),
        }
        return supported_features.get(feature, False)

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model

    def set_model(self, model: str) -> None:
        """
        Set new model.

        Args:
            model: Model identifier
        """
        old_model = self.model
        self.model = model
        logger.info(f"Model changed: {old_model} → {model}")

    def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: List[Dict[str, Any]] | str,
    ) -> Optional[int]:
        """Count input tokens using tiktoken approximation."""
        return self._tiktoken_estimate(messages, tools, system)

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "deepinfra"
