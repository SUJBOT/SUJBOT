"""
Tests for context compaction manager (src/agent/context_manager.py).

Covers:
- ContextBudgetMonitor: threshold triggering, update from response, edge cases
- prune_tool_outputs: preserves first/last messages, replaces content, handles images,
  preserves tool_use_ids
- compact_with_summary: mock provider, preserves recent messages, handles provider failure
- emergency_truncate: keeps 7 messages, no-op for small history
- get_context_window: model registry lookup, fallback
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.agent.context_manager import (
    COMPACT_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW,
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
        """Before any update, all thresholds return False."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        assert not monitor.needs_pruning()
        assert not monitor.needs_compaction()
        assert not monitor.needs_emergency_truncation()
        assert monitor.check().ratio == 0.0

    def test_update_from_response(self):
        """update_from_response extracts input_tokens and enables thresholds."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = _make_response(input_tokens=75_000)
        monitor.update_from_response(resp)

        budget = monitor.check()
        assert budget.used_tokens == 75_000
        assert budget.ratio == 0.75
        assert monitor.needs_pruning()
        assert not monitor.needs_compaction()

    def test_threshold_boundaries(self):
        """Each threshold triggers at exact boundary."""
        cw = 100_000
        monitor = ContextBudgetMonitor(context_window=cw)

        # Below prune
        monitor.update_from_response(_make_response(input_tokens=69_999))
        assert not monitor.needs_pruning()

        # At prune boundary
        monitor.update_from_response(_make_response(input_tokens=70_000))
        assert monitor.needs_pruning()
        assert not monitor.needs_compaction()

        # At compact boundary
        monitor.update_from_response(_make_response(input_tokens=85_000))
        assert monitor.needs_compaction()
        assert not monitor.needs_emergency_truncation()

        # At emergency boundary
        monitor.update_from_response(_make_response(input_tokens=95_000))
        assert monitor.needs_emergency_truncation()

    def test_update_with_no_usage(self):
        """Response without usage attribute doesn't crash or change state."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = MagicMock(spec=[])  # No usage attribute
        monitor.update_from_response(resp)
        assert not monitor._initialized
        assert monitor.check().used_tokens == 0

    def test_update_with_empty_usage(self):
        """Response with empty usage dict doesn't crash."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = MagicMock()
        resp.usage = {}
        monitor.update_from_response(resp)
        assert not monitor._initialized

    def test_update_with_zero_tokens(self):
        """Response with 0 input_tokens doesn't enable the monitor."""
        monitor = ContextBudgetMonitor(context_window=100_000)
        resp = _make_response(input_tokens=0)
        monitor.update_from_response(resp)
        assert not monitor._initialized

    def test_zero_context_window(self):
        """Zero context window doesn't cause division by zero."""
        monitor = ContextBudgetMonitor(context_window=0)
        assert monitor.check().ratio == 0.0


# ===========================================================================
# TestContextBudget
# ===========================================================================


class TestContextBudget:
    def test_ratio_calculation(self):
        b = ContextBudget(used_tokens=50_000, max_tokens=100_000)
        assert b.ratio == 0.5

    def test_ratio_zero_max(self):
        b = ContextBudget(used_tokens=100, max_tokens=0)
        assert b.ratio == 0.0


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
        result = compact_with_summary(messages, provider, "system prompt", protect_last_n=2)

        # Should be: first_msg + summary + last 4 messages
        assert result[0]["content"] == "original query"
        assert "[Context summary" in result[1]["content"]
        assert result[1]["role"] == "assistant"
        # Last 4 messages preserved
        assert result[-1]["content"] == "recent tool result"
        assert result[-2]["content"] == "recent thought"

    def test_returns_unchanged_on_provider_failure(self):
        """If LLM call fails, returns messages unchanged."""
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
        provider.create_message.side_effect = RuntimeError("API down")

        result = compact_with_summary(messages, provider, "system", protect_last_n=2)
        assert result == messages

    def test_not_enough_messages(self):
        """With too few messages, returns unchanged."""
        messages = [
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "answer"},
        ]

        provider = self._make_provider()
        result = compact_with_summary(messages, provider, "system", protect_last_n=2)
        assert result == messages
        provider.create_message.assert_not_called()


# ===========================================================================
# TestEmergencyTruncate
# ===========================================================================


class TestEmergencyTruncate:
    def test_keeps_7_messages(self):
        """With many messages, keeps first + last 6."""
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]

        result = emergency_truncate(messages)

        assert len(result) == 7
        assert result[0]["content"] == "msg-0"
        assert result[-1]["content"] == "msg-19"

    def test_no_op_for_small_history(self):
        """With <= 7 messages, returns unchanged."""
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(5)]

        result = emergency_truncate(messages)
        assert result == messages

    def test_exact_boundary(self):
        """With exactly 7 messages, no truncation."""
        messages = [{"role": "user", "content": f"msg-{i}"} for i in range(7)]
        result = emergency_truncate(messages)
        assert len(result) == 7


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
