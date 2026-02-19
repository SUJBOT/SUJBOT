"""
Tests for VLLMProvider._format_messages — Anthropic → OpenAI format translation.

Tests tool_use, tool_result, multimodal image, text-only message handling,
_convert_response, and _strip_think_tags.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agent.providers.vllm_provider import VLLMProvider
from src.exceptions import ProviderError


@pytest.fixture
def provider():
    """Create VLLMProvider with mock vLLM server URL."""
    with patch("src.agent.providers.vllm_provider.wrap_openai", return_value=None):
        return VLLMProvider(
            base_url="http://localhost:18080/v1",
            model="Qwen/Qwen3-VL-30B-A3B-Thinking",
        )


class TestFormatMessagesTextOnly:
    """Text-only message conversion."""

    def test_string_system_prompt(self, provider):
        result = provider._format_messages([], system="You are helpful.")
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_structured_system_prompt(self, provider):
        system = [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": " Be concise.", "cache_control": {"type": "ephemeral"}},
        ]
        result = provider._format_messages([], system=system)
        assert result == [{"role": "system", "content": "You are helpful. Be concise."}]

    def test_simple_text_messages(self, provider):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = provider._format_messages(messages, system=None)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there"}

    def test_user_message_with_text_blocks(self, provider):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        assert len(result) == 1
        assert result[0]["content"] == "Hello world"


class TestFormatMessagesToolUse:
    """Tool use (assistant) message conversion."""

    def test_assistant_tool_use_converted_to_tool_calls(self, provider):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "search",
                        "input": {"query": "safety"},
                    },
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me search."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "safety"}

    def test_assistant_tool_use_without_text(self, provider):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_456",
                        "name": "expand_context",
                        "input": {"chunk_ids": ["c1"]},
                    },
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        msg = result[0]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1


class TestFormatMessagesToolResult:
    """Tool result (user) message conversion."""

    def test_tool_result_converted_to_tool_message(self, provider):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Found 5 results about safety.",
                    },
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "Found 5 results about safety."

    def test_tool_result_with_text_block_alongside(self, provider):
        """User message with both tool_result and text blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_123",
                        "content": "Result data",
                    },
                    {"type": "text", "text": "Also, can you explain?"},
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Also, can you explain?"


class TestFormatMessagesMultimodal:
    """Multimodal (image) message conversion."""

    def test_user_image_message(self, provider):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                    },
                    {"type": "text", "text": "What is on this page?"},
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert "data:image/png;base64,iVBORw0KGgo=" in content[0]["image_url"]["url"]
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "What is on this page?"

    def test_tool_result_with_multimodal_content(self, provider):
        """Tool result containing image blocks (VL mode).

        OpenAI API requires tool message content to be a string.
        Images are split into a user message with interleaved labels.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_789",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            },
                            {"type": "text", "text": "Page 1 from BZ_VR1"},
                        ],
                    },
                ],
            }
        ]
        result = provider._format_messages(messages, system=None)
        # Tool message (text only) + user message (images with labels)
        assert len(result) == 2

        # Tool message gets text as string
        tool_msg = result[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_789"
        assert isinstance(tool_msg["content"], str)
        assert "Page 1 from BZ_VR1" in tool_msg["content"]

        # Images in user message with interleaved labels
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        image_blocks = [b for b in user_msg["content"] if b["type"] == "image_url"]
        assert len(image_blocks) == 1
        assert "data:image/png;base64,AAAA" in image_blocks[0]["image_url"]["url"]
        # Label should appear before the image
        text_blocks = [b for b in user_msg["content"] if b["type"] == "text"]
        assert any("Page 1 from BZ_VR1" in b["text"] for b in text_blocks)
        # Instruction text should be present
        assert any("READ" in b["text"] for b in text_blocks)


class TestStripThinkTags:
    """Test _strip_think_tags edge cases."""

    def test_complete_think_block_removed(self, provider):
        text = "<think>reasoning here</think>Final answer."
        assert provider._strip_think_tags(text) == "Final answer."

    def test_multiple_think_blocks(self, provider):
        text = "<think>first</think>A<think>second</think>B"
        assert provider._strip_think_tags(text) == "AB"

    def test_truncated_think_block(self, provider):
        text = "<think>reasoning without closing tag"
        assert provider._strip_think_tags(text) == ""

    def test_no_think_tags(self, provider):
        text = "Just plain text."
        assert provider._strip_think_tags(text) == "Just plain text."

    def test_empty_string(self, provider):
        assert provider._strip_think_tags("") == ""

    def test_multiline_think_block(self, provider):
        text = "<think>\nline1\nline2\n</think>\nAnswer here."
        assert provider._strip_think_tags(text) == "Answer here."

    def test_think_only_response(self, provider):
        text = "<think>all reasoning, no answer</think>"
        assert provider._strip_think_tags(text) == ""

    def test_nested_angle_brackets(self, provider):
        text = "<think>has <b>html</b> inside</think>Result"
        assert provider._strip_think_tags(text) == "Result"


class TestConvertResponse:
    """Test _convert_response — OpenAI response → Anthropic ProviderResponse."""

    def _mock_response(self, content=None, tool_calls=None, finish_reason="stop",
                       prompt_tokens=10, completion_tokens=5):
        """Build a mock OpenAI response object."""
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = finish_reason

        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    def test_text_response(self, provider):
        resp = self._mock_response(content="Hello world")
        result = provider._convert_response(resp)
        assert result.content == [{"type": "text", "text": "Hello world"}]
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    def test_text_with_think_tags_stripped(self, provider):
        resp = self._mock_response(content="<think>reasoning</think>Answer")
        result = provider._convert_response(resp)
        assert result.content == [{"type": "text", "text": "Answer"}]

    def test_tool_call_response(self, provider):
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function.name = "search"
        tc.function.arguments = '{"query": "safety"}'
        resp = self._mock_response(content=None, tool_calls=[tc])
        result = provider._convert_response(resp)
        assert len(result.content) == 1
        assert result.content[0]["type"] == "tool_use"
        assert result.content[0]["name"] == "search"
        assert result.content[0]["input"] == {"query": "safety"}

    def test_text_plus_tool_call(self, provider):
        tc = MagicMock()
        tc.id = "call_xyz"
        tc.function.name = "expand_context"
        tc.function.arguments = '{"chunk_ids": ["c1"]}'
        resp = self._mock_response(content="Let me search.", tool_calls=[tc])
        result = provider._convert_response(resp)
        assert len(result.content) == 2
        assert result.content[0]["type"] == "text"
        assert result.content[1]["type"] == "tool_use"

    def test_empty_choices_raises(self, provider):
        resp = MagicMock()
        resp.choices = []
        with pytest.raises(ValueError, match="no choices"):
            provider._convert_response(resp)

    def test_all_tool_calls_unparseable_raises_provider_error(self, provider):
        tc = MagicMock()
        tc.id = "call_bad"
        tc.function.name = "search"
        tc.function.arguments = "not-json{"
        resp = self._mock_response(content=None, tool_calls=[tc])
        with pytest.raises(ProviderError, match="unparseable"):
            provider._convert_response(resp)

    def test_partial_tool_call_failure_keeps_good_ones(self, provider):
        tc_good = MagicMock()
        tc_good.id = "call_1"
        tc_good.function.name = "search"
        tc_good.function.arguments = '{"query": "ok"}'

        tc_bad = MagicMock()
        tc_bad.id = "call_2"
        tc_bad.function.name = "expand"
        tc_bad.function.arguments = "broken{"

        resp = self._mock_response(content=None, tool_calls=[tc_good, tc_bad])
        result = provider._convert_response(resp)
        assert len(result.content) == 1
        assert result.content[0]["name"] == "search"

    def test_think_only_response_yields_empty_text(self, provider):
        resp = self._mock_response(content="<think>only thinking</think>")
        result = provider._convert_response(resp)
        assert result.content == [{"type": "text", "text": ""}]

    def test_no_usage_defaults_to_zero(self, provider):
        resp = self._mock_response(content="Answer")
        resp.usage = None
        result = provider._convert_response(resp)
        assert result.usage["input_tokens"] == 0
        assert result.usage["output_tokens"] == 0


class TestFormatMessagesFullToolLoop:
    """End-to-end tool loop conversation."""

    def test_full_tool_loop_conversation(self, provider):
        """Simulate a complete tool loop: user → assistant(tool_use) → user(tool_result) → assistant(text)."""
        messages = [
            {"role": "user", "content": "What is the safety margin?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search for that."},
                    {
                        "type": "tool_use",
                        "id": "call_001",
                        "name": "search",
                        "input": {"query": "safety margin"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_001",
                        "content": "Safety margin is 1.5x as per section 4.2.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Based on the search results, the safety margin is 1.5x (section 4.2).",
            },
        ]
        result = provider._format_messages(messages, system="You are a RAG agent.")
        assert len(result) == 5  # system + 4 messages

        # System
        assert result[0]["role"] == "system"
        # User query
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What is the safety margin?"
        # Assistant with tool call
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[2]["content"] == "I'll search for that."
        # Tool result
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_001"
        # Final assistant answer
        assert result[4]["role"] == "assistant"
        assert "1.5x" in result[4]["content"]
