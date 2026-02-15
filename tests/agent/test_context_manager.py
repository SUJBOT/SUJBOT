"""
Tests for context compaction manager (src/agent/context_manager.py).

Covers:
- ContextBudget: frozen dataclass, validation, ratio
- ContextBudgetMonitor: threshold triggering, recommended_action, update, is_initialized,
  non-dict usage objects, edge cases
- CompactionLayer: enum dispatch
- prune_tool_outputs: preserves first/last messages, replaces content, handles images,
  preserves tool_use_ids, text truncation, protect_last_n=0
- compact_with_summary: mock provider, preserves recent messages, handles provider failure,
  multimodal content serialization, message role alternation, protect_last_n_pairs=0
- emergency_truncate: keeps EMERGENCY_KEEP_MESSAGES, no-op for small history
- get_context_window: model registry lookup, fallback
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.agent.context_manager import (
    COMPACT_THRESHOLD,
    CompactionLayer,
    DEFAULT_CONTEXT_WINDOW,
    EMERGENCY_KEEP_MESSAGES,
    EMERGENCY_THRESHOLD,
    PRUNE_THRESHOLD,
    ContextBudget,
    ContextBudgetMonitor,
    compact_with_summary,
    emergency_truncate,
    get_context_window,
    prune_tool_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(input_tokens: int = 0, output_tokens: int = 0):
    """Create a mock provider response with usage dict."""
    resp = MagicMock()
    resp.usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    return resp


def _make_tool_result_message(tool_use_id: str, content: Any) -> Dict:
    """Build a user message with a single tool_result block."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        ],
    }


def _make_tool_use_message(tool_use_id: str, name: str) -> Dict:
    """Build an assistant message with a tool_use block."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
                "input": {"query": "test"},
            }
        ],
    }


def _make_image_content() -> List[Dict]:
    """Build content blocks with an image (simulating VL page result)."""
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "A" * 5000,
            },
        },
        {
            "type": "text",
            "text": "[BZ_VR1_p003 | Page 3 from doc1 | score: 0.821]",
        },
    ]


# ===========================================================================
# TestContextBudgetMonitor
# ===========================================================================


class TestContextBudgetMonitor:
    def test_initial_state_no_action(self):
        """Before any update, recommended_action returns NONE."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        assert monitor.recommended_action() == CompactionLayer.NONE
        assert monitor.check().ratio == 0.0
        assert not monitor.is_initialized

    def test_update_from_response(self):
        """update_from_response extracts input_tokens and enables thresholds."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = _make_response(input_tokens=75_000)
        monitor.update_from_response(resp)

        budget = monitor.check()
        assert budget.used_tokens == 75_000
        assert budget.ratio == 0.75
        assert monitor.is_initialized
        assert monitor.recommended_action() == CompactionLayer.PRUNE

    def test_threshold_boundaries(self):
        """Each threshold triggers at exact boundary via recommended_action."""
        cw = 100_000
        monitor = ContextBudgetMonitor(context_window=cw)

        # Below prune
        monitor.update_from_response(_make_response(input_tokens=69_999))
        assert monitor.recommended_action() == CompactionLayer.NONE

        # At prune boundary
        monitor.update_from_response(_make_response(input_tokens=70_000))
        assert monitor.recommended_action() == CompactionLayer.PRUNE

        # At compact boundary
        monitor.update_from_response(_make_response(input_tokens=85_000))
        assert monitor.recommended_action() == CompactionLayer.COMPACT

        # At emergency boundary
        monitor.update_from_response(_make_response(input_tokens=95_000))
        assert monitor.recommended_action() == CompactionLayer.EMERGENCY

    def test_update_with_no_usage(self):
        """Response without usage attribute doesn't crash or change state."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = MagicMock(spec=[])  # No usage attribute
        monitor.update_from_response(resp)
        assert not monitor.is_initialized
        assert monitor.check().used_tokens == 0

    def test_update_with_empty_usage(self):
        """Response with empty usage dict doesn't crash."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = MagicMock()
        resp.usage = {}
        monitor.update_from_response(resp)
        assert not monitor.is_initialized

    def test_update_with_zero_tokens(self):
        """Response with 0 input_tokens doesn't enable the monitor."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = _make_response(input_tokens=0)
        monitor.update_from_response(resp)
        assert not monitor.is_initialized

    def test_update_with_non_dict_usage_object(self):
        """Non-dict usage with input_tokens attribute is handled correctly."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = MagicMock()
        # Simulate a Pydantic model / dataclass with input_tokens attribute
        usage_obj = MagicMock()
        usage_obj.input_tokens = 80_000
        resp.usage = usage_obj
        monitor.update_from_response(resp)
        assert monitor.is_initialized
        assert monitor.check().used_tokens == 80_000

    def test_update_with_unknown_usage_type(self):
        """Unknown usage type without input_tokens logs warning and doesn't initialize."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = MagicMock()
        resp.usage = "invalid-string-usage"
        monitor.update_from_response(resp)
        assert not monitor.is_initialized

    def test_zero_context_window_raises(self):
        """Zero context window raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ContextBudgetMonitor(context_window=0)

    def test_negative_context_window_raises(self):
        """Negative context window raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ContextBudgetMonitor(context_window=-100)

    def test_context_window_is_readonly(self):
        """context_window property is read-only."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        assert monitor.context_window == 100_000
        with pytest.raises(AttributeError):
            monitor.context_window = 50_000

    def test_recommended_action_none_before_init(self):
        """Before any update, recommended_action returns NONE."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        assert monitor.recommended_action() == CompactionLayer.NONE

    def test_recommended_action_prune(self):
        """At 75%, recommended_action returns PRUNE."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        monitor.update_from_response(_make_response(input_tokens=75_000))
        assert monitor.recommended_action() == CompactionLayer.PRUNE

    def test_recommended_action_compact(self):
        """At 90%, recommended_action returns COMPACT (not PRUNE)."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        monitor.update_from_response(_make_response(input_tokens=90_000))
        assert monitor.recommended_action() == CompactionLayer.COMPACT

    def test_recommended_action_emergency(self):
        """At 96%, recommended_action returns EMERGENCY (highest priority)."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        monitor.update_from_response(_make_response(input_tokens=96_000))
        assert monitor.recommended_action() == CompactionLayer.EMERGENCY

    def test_recommended_action_below_threshold(self):
        """At 50%, recommended_action returns NONE."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        monitor.update_from_response(_make_response(input_tokens=50_000))
        assert monitor.recommended_action() == CompactionLayer.NONE


# ===========================================================================
# TestContextBudget
# ===========================================================================


class TestContextBudget:
    def test_ratio_calculation(self):
        b = ContextBudget(used_tokens=50_000, max_tokens=100_000)
        assert b.ratio == 0.5

    def test_zero_max_tokens_raises(self):
        """max_tokens=0 raises ValueError (aligned with ContextBudgetMonitor)."""
        with pytest.raises(ValueError, match="must be positive"):
            ContextBudget(used_tokens=0, max_tokens=0)

    def test_frozen(self):
        """ContextBudget is immutable."""
        b = ContextBudget(used_tokens=100, max_tokens=1000)
        with pytest.raises(AttributeError):
            b.used_tokens = 200

    def test_negative_used_tokens_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContextBudget(used_tokens=-1, max_tokens=100)

    def test_negative_max_tokens_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            ContextBudget(used_tokens=0, max_tokens=-1)

    def test_ratio_above_one(self):
        """Overflow scenario: ratio can exceed 1.0."""
        b = ContextBudget(used_tokens=150_000, max_tokens=100_000)
        assert b.ratio == 1.5


# ===========================================================================
# TestPruneToolOutputs
# ===========================================================================


class TestPruneToolOutputs:
    def test_preserves_first_and_last_messages(self):
        """First user message and last 2 tool-result messages stay intact."""
        messages = [
            {"role": "user", "content": "original query"},
            _make_tool_use_message("t1", "search"),
            _make_tool_result_message("t1", "result 1 " * 200),
            _make_tool_use_message("t2", "search"),
            _make_tool_result_message("t2", "result 2 " * 200),
            _make_tool_use_message("t3", "search"),
            _make_tool_result_message("t3", "result 3"),
            _make_tool_use_message("t4", "expand"),
            _make_tool_result_message("t4", "result 4"),
        ]

        pruned = prune_tool_outputs(messages, protect_last_n=2)

        # First message unchanged
        assert pruned[0]["content"] == "original query"

        # Last 2 tool-result messages (indices 6, 8) should be unchanged
        last_tool_results = [
            m for m in pruned if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        assert last_tool_results[-1]["content"][0]["content"] == "result 4"
        assert last_tool_results[-2]["content"][0]["content"] == "result 3"

    def test_preserves_tool_use_id(self):
        """Pruned tool_result blocks MUST retain tool_use_id."""
        messages = [
            {"role": "user", "content": "query"},
            _make_tool_use_message("id-abc", "search"),
            _make_tool_result_message("id-abc", "long " * 300),
            _make_tool_use_message("id-def", "search"),
            _make_tool_result_message("id-def", "short"),
        ]

        pruned = prune_tool_outputs(messages, protect_last_n=1)

        # First tool_result was pruned but must keep tool_use_id
        tool_result_msgs = [
            m for m in pruned if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        first_tr = tool_result_msgs[0]["content"][0]
        assert first_tr["tool_use_id"] == "id-abc"
        assert first_tr["type"] == "tool_result"

    def test_replaces_image_blocks(self):
        """Image blocks in tool results are replaced with text placeholders."""
        image_content = _make_image_content()
        messages = [
            {"role": "user", "content": "query"},
            _make_tool_use_message("t1", "search"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": image_content,
                    }
                ],
            },
            _make_tool_use_message("t2", "search"),
            _make_tool_result_message("t2", "recent result"),
        ]

        pruned = prune_tool_outputs(messages, protect_last_n=1)

        # Find the pruned tool result
        tr_msgs = [
            m for m in pruned if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        pruned_tr = tr_msgs[0]["content"][0]
        inner = pruned_tr["content"]
        # Image should be replaced with text placeholder
        assert isinstance(inner, list)
        assert inner[0]["type"] == "text"
        assert "pruned" in inner[0]["text"]
        # No image blocks should remain
        assert all(b.get("type") != "image" for b in inner)

    def test_no_op_when_few_messages(self):
        """With <= protect_last_n tool results, nothing is pruned."""
        messages = [
            {"role": "user", "content": "query"},
            _make_tool_use_message("t1", "search"),
            _make_tool_result_message("t1", "result 1"),
        ]

        pruned = prune_tool_outputs(messages, protect_last_n=2)
        # Should be unchanged
        assert len(pruned) == len(messages)

    def test_does_not_mutate_original(self):
        """prune_tool_outputs returns a new list; original is unchanged."""
        messages = [
            {"role": "user", "content": "query"},
            _make_tool_use_message("t1", "search"),
            _make_tool_result_message("t1", "x" * 500),
            _make_tool_use_message("t2", "search"),
            _make_tool_result_message("t2", "recent"),
        ]
        original_content = messages[2]["content"][0]["content"]

        prune_tool_outputs(messages, protect_last_n=1)

        # Original should be untouched
        assert messages[2]["content"][0]["content"] == original_content

    def test_text_truncation_boundary(self):
        """Text blocks > 300 chars are truncated to first 150 + marker + last 100."""
        long_text = "A" * 400
        messages = [
            {"role": "user", "content": "query"},
            _make_tool_use_message("t1", "search"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": long_text}],
                    }
                ],
            },
            _make_tool_use_message("t2", "search"),
            _make_tool_result_message("t2", "recent"),
        ]

        pruned = prune_tool_outputs(messages, protect_last_n=1)

        tr = pruned[2]["content"][0]["content"][0]
        assert tr["type"] == "text"
        assert tr["text"].startswith("A" * 150)
        assert "[...pruned...]" in tr["text"]
        assert tr["text"].endswith("A" * 100)
        assert len(tr["text"]) < 400  # Shorter than original

    def test_protect_last_n_zero(self):
        """protect_last_n=0 prunes all tool results."""
        messages = [
            {"role": "user", "content": "query"},
            _make_tool_use_message("t1", "search"),
            _make_tool_result_message("t1", "x" * 500),
            _make_tool_use_message("t2", "search"),
            _make_tool_result_message("t2", "y" * 500),
        ]

        pruned = prune_tool_outputs(messages, protect_last_n=0)

        # Both tool results should be pruned
        for msg in pruned:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                tr = msg["content"][0]
                if tr.get("type") == "tool_result":
                    content = tr["content"]
                    if isinstance(content, str):
                        assert "[...pruned...]" in content


# ===========================================================================
# TestCompactWithSummary
# ===========================================================================


class TestCompactWithSummary:
    def _make_provider(self, summary_text: str = "Summary of conversation."):
        """Create a mock provider that returns a summary."""
        provider = MagicMock()
        resp = MagicMock()
        resp.text = summary_text
        provider.create_message.return_value = resp
        return provider

    def test_compacts_middle_preserves_ends(self):
        """First message and last N pairs are preserved, middle is summarized."""
        messages = [
            {"role": "user", "content": "original query"},          # 0
            {"role": "assistant", "content": "thought 1"},          # 1
            {"role": "user", "content": "tool result 1"},           # 2
            {"role": "assistant", "content": "thought 2"},          # 3
            {"role": "user", "content": "tool result 2"},           # 4
            {"role": "assistant", "content": "thought 3"},          # 5
            {"role": "user", "content": "tool result 3"},           # 6
            {"role": "assistant", "content": "recent thought"},     # 7
            {"role": "user", "content": "recent tool result"},      # 8
        ]

        provider = self._make_provider("Compacted summary here.")
        result = compact_with_summary(messages, provider, protect_last_n_pairs=2)

        # Should be: first_msg + summary + bridge + last 4 messages
        assert result[0]["content"] == "original query"
        assert result[0]["role"] == "user"
        assert "[Context summary" in result[1]["content"]
        assert result[1]["role"] == "assistant"
        # Bridging user message
        assert result[2]["role"] == "user"
        assert "Continuing" in result[2]["content"]
        # Last 4 messages preserved
        assert result[-1]["content"] == "recent tool result"
        assert result[-2]["content"] == "recent thought"

    def test_message_roles_alternate(self):
        """Compacted messages always alternate user/assistant (Anthropic API requirement)."""
        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "r2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "r3"},
            {"role": "assistant", "content": "a4"},
            {"role": "user", "content": "r4"},
        ]

        provider = self._make_provider("Summary.")
        result = compact_with_summary(messages, provider, protect_last_n_pairs=2)

        # Verify strict alternation
        for i in range(1, len(result)):
            assert result[i]["role"] != result[i - 1]["role"], (
                f"Consecutive same roles at positions {i-1},{i}: "
                f"{result[i-1]['role']}, {result[i]['role']}"
            )

    def test_returns_unchanged_on_provider_failure(self):
        """If LLM call fails with ProviderError, returns messages unchanged."""
        from src.exceptions import ProviderError

        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "r2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "r3"},
        ]

        provider = MagicMock()
        provider.create_message.side_effect = ProviderError("API down")

        result = compact_with_summary(messages, provider, protect_last_n_pairs=2)
        assert result == messages

    def test_returns_unchanged_on_connection_error(self):
        """If LLM call fails with ConnectionError, returns messages unchanged."""
        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "r2"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "r3"},
        ]

        provider = MagicMock()
        provider.create_message.side_effect = ConnectionError("Network down")

        result = compact_with_summary(messages, provider, protect_last_n_pairs=2)
        assert result == messages

    def test_protect_last_n_pairs_zero(self):
        """protect_last_n_pairs=0 summarizes everything except the first message."""
        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "r1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "r2"},
        ]

        provider = self._make_provider("Full summary.")
        result = compact_with_summary(messages, provider, protect_last_n_pairs=0)

        # Should be: first_msg + summary + bridge (no tail)
        assert result[0]["content"] == "query"
        assert "[Context summary" in result[1]["content"]
        assert result[2]["role"] == "user"
        assert len(result) == 3

    def test_not_enough_messages(self):
        """With too few messages, returns unchanged."""
        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "answer"},
        ]

        provider = self._make_provider()
        result = compact_with_summary(messages, provider, protect_last_n_pairs=2)
        assert result == messages
        provider.create_message.assert_not_called()

    def test_multimodal_content_serialization(self):
        """Middle messages with tool_use, tool_result, and image blocks serialize correctly."""
        messages = [
            {"role": "user", "content": "original query"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "data": "AAA"}},
                            {"type": "text", "text": "[page_id | Page 1 from doc]"},
                        ],
                    }
                ],
            },
            {"role": "assistant", "content": "thinking about results"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "recent thought"},
            {"role": "user", "content": "recent result"},
            {"role": "assistant", "content": "last thought"},
            {"role": "user", "content": "last result"},
        ]

        provider = self._make_provider("Summary with citations.")
        result = compact_with_summary(messages, provider, protect_last_n_pairs=2)

        # Provider was called — verify the conversation_text includes tool info
        call_args = provider.create_message.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "[tool_use: search]" in user_content
        assert "[image]" in user_content
        assert "[page_id | Page 1 from doc]" in user_content


# ===========================================================================
# TestEmergencyTruncate
# ===========================================================================


class TestEmergencyTruncate:
    def test_keeps_emergency_count_messages(self):
        """With many messages, keeps first + last (EMERGENCY_KEEP_MESSAGES - 1)."""
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]

        result = emergency_truncate(messages)

        assert len(result) == EMERGENCY_KEEP_MESSAGES
        assert result[0]["content"] == "msg-0"
        assert result[-1]["content"] == "msg-19"
        # Verify structure: first msg + (EMERGENCY_KEEP_MESSAGES - 1) recent
        assert result == [messages[0]] + messages[-(EMERGENCY_KEEP_MESSAGES - 1):]

    def test_no_op_for_small_history(self):
        """With <= EMERGENCY_KEEP_MESSAGES messages, returns unchanged."""
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(5)]

        result = emergency_truncate(messages)
        assert result == messages

    def test_exact_boundary(self):
        """With exactly EMERGENCY_KEEP_MESSAGES messages, no truncation."""
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(EMERGENCY_KEEP_MESSAGES)]
        result = emergency_truncate(messages)
        assert len(result) == EMERGENCY_KEEP_MESSAGES


# ===========================================================================
# TestGetContextWindow
# ===========================================================================


class TestGetContextWindow:
    def test_known_model(self):
        """Registered model returns its context window."""
        with patch("src.utils.model_registry.ModelRegistry.get_model_config") as mock_get:
            mock_config = MagicMock()
            mock_config.context_window = 200_000
            mock_get.return_value = mock_config

            result = get_context_window("claude-sonnet-4-5-20250929")
            assert result == 200_000

    def test_unknown_model_returns_default(self):
        """Unknown model returns DEFAULT_CONTEXT_WINDOW."""
        with patch("src.utils.model_registry.ModelRegistry.get_model_config") as mock_get:
            mock_get.side_effect = KeyError("not found")
            result = get_context_window("unknown-model-xyz")
            assert result == DEFAULT_CONTEXT_WINDOW
