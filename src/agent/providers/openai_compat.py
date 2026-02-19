"""
Shared OpenAI-compatible format conversion helpers.

Used by both OpenAIProvider and VLLMProvider to avoid duplicating
Anthropic <-> OpenAI format translation logic.
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Maps OpenAI finish_reason to Anthropic stop_reason
STOP_REASON_MAP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}


def convert_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function calling format.

    Args:
        tools: Anthropic-format tool definitions

    Returns:
        OpenAI-format function tool definitions
    """
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


def convert_response_to_anthropic(response: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI ChatCompletion response to Anthropic content blocks.

    Args:
        response: OpenAI ChatCompletion response

    Returns:
        List of Anthropic content block dicts

    Raises:
        ValueError: If response has no choices
    """
    if not response.choices:
        raise ValueError("Invalid OpenAI response: no choices returned")

    message = response.choices[0].message
    content: List[Dict[str, Any]] = []

    if message.content:
        content.append({"type": "text", "text": message.content})

    if message.tool_calls:
        for tool_call in message.tool_calls:
            if not tool_call.function:
                logger.warning(f"Tool call missing function field: {tool_call}")
                continue
            try:
                parsed_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Malformed tool arguments from LLM: {e}. "
                    f"Tool: {tool_call.function.name}, "
                    f"Args: {tool_call.function.arguments[:200]}"
                )
                parsed_args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": parsed_args,
                }
            )

    return content


def convert_system_to_string(system: List[Dict[str, Any]] | str) -> str:
    """Extract system prompt text from Anthropic structured or string format.

    Multiple text blocks are joined with a single space (paragraph structure
    is not preserved).

    Args:
        system: System prompt (structured list or plain string)

    Returns:
        System prompt as a plain string
    """
    if isinstance(system, str):
        return system
    return " ".join(
        block["text"] for block in system if isinstance(block, dict) and block.get("type") == "text"
    )


def convert_assistant_blocks_to_openai(content: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert Anthropic assistant content blocks to an OpenAI assistant message.

    Args:
        content: Anthropic content blocks

    Returns:
        OpenAI-format assistant message dict
    """
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                }
            )

    msg: Dict[str, Any] = {"role": "assistant"}
    msg["content"] = " ".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls

    return msg
