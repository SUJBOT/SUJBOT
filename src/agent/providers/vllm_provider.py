"""
vLLM Provider using OpenAI-compatible API.

Connects to local vLLM servers (Qwen3-VL models on GB10 nodes).
Model configuration is in config.json (SSOT).
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Iterator

from openai import OpenAI
from langsmith.wrappers import wrap_openai

from .base import BaseProvider, ProviderResponse
from .openai_compat import STOP_REASON_MAP, convert_tools_to_openai

logger = logging.getLogger(__name__)


class VLLMProvider(BaseProvider):
    """
    vLLM provider for local LLMs via OpenAI-compatible API.

    Connects to vLLM servers running Qwen3-VL models on GB10 nodes.
    Supports streaming, tool use, and multimodal (vision) content.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        provider_name: str = "local_llm",
    ):
        """
        Initialize vLLM provider.

        Args:
            base_url: vLLM server URL (e.g. http://host.docker.internal:18080/v1)
            model: Model identifier (e.g. Qwen/Qwen3-VL-30B-A3B-Thinking)
            provider_name: Provider name for cost tracking (local_llm or local_llm_8b)
        """
        self.model = model
        self._provider_name = provider_name

        # Initialize OpenAI client (wrapped for LangSmith tracing)
        self.client = wrap_openai(
            OpenAI(
                api_key="local-no-key",
                base_url=base_url,
                timeout=120.0,
                max_retries=3,
            )
        )

        logger.info("VLLMProvider initialized: %s (%s)", model, base_url)

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
        openai_tools = convert_tools_to_openai(tools) if tools else None

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
            logger.error(f"vLLM API error: {e}")
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
        openai_tools = convert_tools_to_openai(tools) if tools else None

        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=openai_tools,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )
        except Exception as e:
            logger.error(f"vLLM streaming API error: {e}")
            raise

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
            # Convert tool_result blocks to OpenAI tool messages.
            # OpenAI API requires tool message content to be a string, so
            # multimodal results (images) must be split: text → tool message,
            # images → interleaved user message with labels.
            deferred_image_blocks = []  # (label_text, image_url_block) pairs
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        # Multimodal tool result — split text from images,
                        # preserving label↔image association.
                        # Runner emits: image, text, image, text, ...
                        # We pair each image with the NEXT text block as its label.
                        text_parts = []
                        images = []
                        labels = []
                        for sub in tool_content:
                            if not isinstance(sub, dict):
                                continue
                            if sub.get("type") == "text":
                                label = sub.get("text", "")
                                text_parts.append(label)
                                labels.append(label)
                            elif sub.get("type") == "image":
                                source = sub.get("source", {})
                                if source.get("type") == "base64":
                                    media_type = source.get("media_type", "image/png")
                                    data = source.get("data", "")
                                    images.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{media_type};base64,{data}"
                                            },
                                        }
                                    )
                        # Pair images with labels (zip shortest)
                        for idx, img_block in enumerate(images):
                            label = labels[idx] if idx < len(labels) else None
                            deferred_image_blocks.append((label, img_block))
                        formatted.append(
                            {
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": "\n".join(text_parts) if text_parts else "See images below.",
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

            # Append images as a user message with interleaved labels so the
            # model can associate each image with its page metadata.
            if deferred_image_blocks:
                user_content = [
                    {
                        "type": "text",
                        "text": (
                            "Below are the page images returned by the search tool. "
                            "READ each image carefully to find the answer. "
                            "Each image is labeled with its page_id for citation."
                        ),
                    }
                ]
                for label, img_block in deferred_image_blocks:
                    if label:
                        user_content.append({"type": "text", "text": label})
                    user_content.append(img_block)
                formatted.append({"role": "user", "content": user_content})
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

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> reasoning blocks from Qwen3 model output."""
        # Strip complete think blocks
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        # Strip truncated think block (no closing tag, e.g. max_tokens hit)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

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
            raise ValueError("Empty response from vLLM API - no choices returned")

        choice = response.choices[0]
        message = choice.message

        # Build content blocks
        content_blocks = []

        if message.content:
            clean_text = self._strip_think_tags(message.content)
            if clean_text:
                content_blocks.append({"type": "text", "text": clean_text})

        # Convert tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    parsed_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(
                        "Failed to parse tool arguments from vLLM: %s. Tool: %s, Args: %s",
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

        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            "output_tokens": (
                getattr(response.usage, "completion_tokens", 0) if response.usage else 0
            ),
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }

        if not content_blocks:
            logger.warning(
                "vLLM response had no usable content after processing "
                "(text stripped or all tool calls failed parsing). Raw: %s",
                (message.content or "")[:200],
            )
            content_blocks = [{"type": "text", "text": ""}]

        return ProviderResponse(
            content=content_blocks,
            stop_reason=STOP_REASON_MAP.get(choice.finish_reason, "end_turn"),
            usage=usage,
            model=self.model,
        )

    def supports_feature(self, feature: str) -> bool:
        if feature == "vision":
            return "vl" in self.model.lower()
        return feature in {"streaming", "tool_use"}

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model

    def set_model(self, model: str) -> None:
        """
        Set new model.

        Args:
            model: Model identifier

        Raises:
            ValueError: If model string is empty
        """
        if not model or not model.strip():
            raise ValueError("Model identifier cannot be empty")
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
        """Get provider name for cost tracking."""
        return self._provider_name
