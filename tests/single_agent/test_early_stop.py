"""
Tests for SingleAgentRunner._should_stop_early logic.
"""

import pytest

from src.single_agent.runner import SingleAgentRunner


@pytest.fixture
def runner():
    """Create a minimal SingleAgentRunner (no initialization needed for _should_stop_early)."""
    return SingleAgentRunner(config={})


class TestShouldStopEarly:
    """Tests for the early stop heuristic in tool loops."""

    def test_empty_history_returns_false(self, runner):
        assert runner._should_stop_early([]) is False

    def test_single_call_returns_false(self, runner):
        history = [{"tool": "search", "success": False}]
        assert runner._should_stop_early(history) is False

    def test_two_consecutive_failed_searches_returns_true(self, runner):
        history = [
            {"tool": "search", "success": False},
            {"tool": "search", "success": False},
        ]
        assert runner._should_stop_early(history) is True

    def test_failed_search_variants_trigger_early_stop(self, runner):
        """Consecutive failed search calls should trigger early stop."""
        history = [
            {"tool": "search", "success": False},
            {"tool": "search", "success": False},
        ]
        assert runner._should_stop_early(history) is True

    def test_non_search_tool_failure_does_not_trigger(self, runner):
        history = [
            {"tool": "expand_context", "success": False},
            {"tool": "expand_context", "success": False},
        ]
        assert runner._should_stop_early(history) is False

    def test_successful_search_breaks_streak(self, runner):
        history = [
            {"tool": "search", "success": False},
            {"tool": "search", "success": True},
            {"tool": "search", "success": False},
        ]
        assert runner._should_stop_early(history) is False

    def test_non_search_between_failed_searches_breaks_streak(self, runner):
        history = [
            {"tool": "search", "success": False},
            {"tool": "expand_context", "success": True},
            {"tool": "search", "success": False},
        ]
        assert runner._should_stop_early(history) is False

    def test_three_consecutive_failed_searches(self, runner):
        history = [
            {"tool": "search", "success": False},
            {"tool": "search", "success": False},
            {"tool": "search", "success": False},
        ]
        assert runner._should_stop_early(history) is True
