"""
Tests for SingleAgentRunner.run_query() — initialization guard, provider failure, basic flow.
"""

import pytest

from src.single_agent.runner import SingleAgentRunner


@pytest.fixture
def runner():
    """Create a minimal SingleAgentRunner (not initialized)."""
    return SingleAgentRunner(config={})


class TestRunQueryInitializationGuard:
    """Test that run_query refuses to run when not initialized."""

    @pytest.mark.anyio
    async def test_not_initialized_yields_error(self, runner):
        """run_query must fail fast when _initialized is False."""
        events = []
        async for event in runner.run_query("test query"):
            events.append(event)

        assert len(events) == 1
        final = events[0]
        assert final["type"] == "final"
        assert final["success"] is False
        assert "not initialized" in final["final_answer"].lower()
        assert len(final["errors"]) > 0

    @pytest.mark.anyio
    async def test_not_initialized_does_not_raise(self, runner):
        """run_query should yield error, not raise exception."""
        async for event in runner.run_query("test query"):
            pass  # Should not raise


class TestRunQueryProviderFailure:
    """Test that run_query handles provider creation failures gracefully."""

    @pytest.mark.anyio
    async def test_bad_model_yields_error(self, runner):
        """Invalid model name should yield a final error event."""
        runner._initialized = True

        # Mock minimal tool_adapter
        from unittest.mock import MagicMock

        runner.tool_adapter = MagicMock()
        runner.tool_adapter.get_available_tools.return_value = []
        runner.system_prompt = "test prompt"

        events = []
        async for event in runner.run_query("test query", model="nonexistent-model-xyz"):
            events.append(event)

        assert len(events) == 1
        final = events[0]
        assert final["type"] == "final"
        assert final["success"] is False
        assert "nonexistent-model-xyz" in final["final_answer"]


class TestForceFinishAnswer:
    """Test the _force_final_answer fallback mechanism."""

    @pytest.mark.anyio
    async def test_force_final_answer_returns_string(self, runner):
        """_force_final_answer should return a string even on provider error."""
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_provider.create_message.side_effect = RuntimeError("API down")

        result = await runner._force_final_answer(
            provider=mock_provider,
            messages=[{"role": "user", "content": "test"}],
            system="system prompt",
            max_tokens=1024,
            temperature=0.3,
        )

        assert isinstance(result, str)
        assert "internal error" in result.lower()

    @pytest.mark.anyio
    async def test_force_final_answer_does_not_mutate_messages(self, runner):
        """_force_final_answer must NOT mutate the input messages list."""
        from unittest.mock import MagicMock

        mock_provider = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Final answer"
        mock_provider.create_message.return_value = mock_response

        original_messages = [{"role": "user", "content": "test"}]
        original_len = len(original_messages)

        await runner._force_final_answer(
            provider=mock_provider,
            messages=original_messages,
            system="system prompt",
            max_tokens=1024,
            temperature=0.3,
        )

        assert len(original_messages) == original_len


class TestSingleAgentConfig:
    """Test configuration handling."""

    def test_default_config_values(self):
        runner = SingleAgentRunner(config={})
        assert runner._initialized is False
        assert runner.provider is None
        assert runner.system_prompt == ""

    def test_custom_config_stored(self):
        config = {"single_agent": {"model": "test-model", "max_iterations": 5}}
        runner = SingleAgentRunner(config=config)
        assert runner.single_agent_config["model"] == "test-model"
        assert runner.single_agent_config["max_iterations"] == 5
