"""
Tests for SingleAgentRunner.run_query() — initialization guard, provider failure, basic flow.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

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


class TestDisabledToolRejection:
    """Test that run_query rejects LLM calls to disabled tools."""

    @pytest.mark.anyio
    async def test_disabled_tool_yields_failed_event(self, runner):
        """When LLM calls a disabled tool, runner yields failed tool_call and
        returns error tool_result to LLM instead of executing the tool."""
        from src.agent.providers.base import ProviderResponse

        runner._initialized = True
        runner.tool_adapter = MagicMock()
        runner.tool_adapter.get_available_tools.return_value = ["search", "web_search"]
        runner.tool_adapter.get_tool_schema.side_effect = lambda name: {
            "name": name,
            "description": f"{name} tool",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        runner.tool_adapter.registry = MagicMock()
        runner.system_prompt = "test prompt"

        # First call: LLM hallucinates web_search (disabled)
        # Second call: LLM gives final answer
        call_count = 0

        def create_message_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ProviderResponse(
                    content=[
                        {"type": "tool_use", "id": "call_1", "name": "web_search",
                         "input": {"query": "test"}},
                    ],
                    stop_reason="tool_use",
                    usage={"input_tokens": 100, "output_tokens": 50},
                    model="claude-haiku-4-5",
                )
            else:
                return ProviderResponse(
                    content=[{"type": "text", "text": "Final answer without web search."}],
                    stop_reason="end_turn",
                    usage={"input_tokens": 200, "output_tokens": 30},
                    model="claude-haiku-4-5",
                )

        mock_provider = MagicMock()
        mock_provider.get_provider_name.return_value = "anthropic"
        mock_provider.get_model_name.return_value = "claude-haiku-4-5"
        mock_provider.supports_feature.return_value = False
        mock_provider.create_message.side_effect = create_message_side_effect

        runner._create_provider = MagicMock(return_value=mock_provider)

        events = []
        async for event in runner.run_query(
            "test query",
            model="claude-haiku-4-5",
            stream_progress=True,
            disabled_tools={"web_search"},
        ):
            events.append(event)

        # Should have a failed tool_call event for web_search
        tool_events = [e for e in events if e.get("type") == "tool_call"]
        assert any(
            e["tool"] == "web_search" and e["status"] == "failed"
            for e in tool_events
        ), "Expected a failed tool_call event for disabled web_search"

        # Tool adapter.execute should NOT have been called for web_search
        runner.tool_adapter.execute.assert_not_called()

        # Final event should still succeed
        final = next(e for e in events if e["type"] == "final")
        assert final["success"] is True

    @pytest.mark.anyio
    async def test_disabled_tool_not_in_schemas(self, runner):
        """Disabled tools should be excluded from tool schemas sent to LLM."""
        runner._initialized = True
        runner.tool_adapter = MagicMock()
        runner.tool_adapter.get_available_tools.return_value = ["search", "web_search"]
        runner.tool_adapter.get_tool_schema.side_effect = lambda name: {
            "name": name,
            "description": f"{name} tool",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        runner.tool_adapter.registry = MagicMock()
        runner.system_prompt = "test prompt"

        from src.agent.providers.base import ProviderResponse

        mock_provider = MagicMock()
        mock_provider.get_provider_name.return_value = "anthropic"
        mock_provider.get_model_name.return_value = "claude-haiku-4-5"
        mock_provider.supports_feature.return_value = False
        mock_provider.create_message.return_value = ProviderResponse(
            content=[{"type": "text", "text": "Answer"}],
            stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 50},
            model="claude-haiku-4-5",
        )
        runner._create_provider = MagicMock(return_value=mock_provider)

        events = []
        async for event in runner.run_query(
            "test query",
            model="claude-haiku-4-5",
            disabled_tools={"web_search"},
        ):
            events.append(event)

        # Verify only "search" schema was passed (not "web_search")
        call_args = mock_provider.create_message.call_args
        tools_passed = call_args.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools_passed]
        assert "search" in tool_names
        assert "web_search" not in tool_names


class TestLangSmithConfigKey:
    """Test that _setup_langsmith reads from the correct config key."""

    def test_langsmith_reads_from_top_level_key(self):
        """LangSmith config should be read from config['langsmith']."""
        config = {"langsmith": {"enabled": False}}
        runner = SingleAgentRunner(config=config)
        langsmith_config = runner.config.get("langsmith", {})
        assert langsmith_config.get("enabled") is False

    def test_langsmith_missing_key_is_noop(self):
        """Missing langsmith key should not crash."""
        runner = SingleAgentRunner(config={})
        # _setup_langsmith should not raise
        runner._setup_langsmith()


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


class TestDedupWithAttachments:
    """Test that dedup logic preserves multimodal content when attachments are present."""

    @pytest.mark.anyio
    async def test_dedup_skips_when_no_attachments_and_text_matches(self, runner):
        """Without attachments, matching query text should NOT add a duplicate."""
        from src.agent.providers.base import ProviderResponse

        runner._initialized = True
        runner.tool_adapter = MagicMock()
        runner.tool_adapter.get_available_tools.return_value = []
        runner.tool_adapter.registry = MagicMock()
        runner.system_prompt = "test"

        mock_provider = MagicMock()
        mock_provider.get_provider_name.return_value = "anthropic"
        mock_provider.get_model_name.return_value = "claude-haiku-4-5"
        mock_provider.supports_feature.return_value = False
        mock_provider.create_message.return_value = ProviderResponse(
            content=[{"type": "text", "text": "Answer"}],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 5},
            model="claude-haiku-4-5",
        )
        runner._create_provider = MagicMock(return_value=mock_provider)

        events = []
        async for event in runner.run_query(
            "hello",
            model="claude-haiku-4-5",
            conversation_history=[{"role": "user", "content": "hello"}],
            attachment_blocks=None,
        ):
            events.append(event)

        # Check messages passed to provider — should be 1 (from history, no duplicate)
        call_args = mock_provider.create_message.call_args
        messages = call_args.kwargs["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "hello"

    @pytest.mark.anyio
    async def test_dedup_adds_when_attachments_present(self, runner):
        """With attachments, ALWAYS add multimodal content even if text matches history."""
        from src.agent.providers.base import ProviderResponse

        runner._initialized = True
        runner.tool_adapter = MagicMock()
        runner.tool_adapter.get_available_tools.return_value = []
        runner.tool_adapter.registry = MagicMock()
        runner.system_prompt = "test"

        mock_provider = MagicMock()
        mock_provider.get_provider_name.return_value = "anthropic"
        mock_provider.get_model_name.return_value = "claude-haiku-4-5"
        mock_provider.supports_feature.return_value = False
        mock_provider.create_message.return_value = ProviderResponse(
            content=[{"type": "text", "text": "I see the image"}],
            stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 20},
            model="claude-haiku-4-5",
        )
        runner._create_provider = MagicMock(return_value=mock_provider)

        attachment_blocks = [
            {"type": "image", "source": {"type": "base64", "data": "abc123", "media_type": "image/jpeg"}}
        ]

        events = []
        async for event in runner.run_query(
            "hello",
            model="claude-haiku-4-5",
            conversation_history=[{"role": "user", "content": "hello"}],
            attachment_blocks=attachment_blocks,
        ):
            events.append(event)

        # Check messages — should have 2 user messages:
        # 1. From history (text-only "hello")
        # 2. Fresh multimodal content (image + text)
        call_args = mock_provider.create_message.call_args
        messages = call_args.kwargs["messages"]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 2
        # First is text-only from history
        assert user_messages[0]["content"] == "hello"
        # Second is multimodal (list with image block + text block)
        assert isinstance(user_messages[1]["content"], list)
        assert any(
            b.get("type") == "image" for b in user_messages[1]["content"]
        )

        # Ordering: history text MUST come before multimodal content
        all_messages = messages
        user_indices = [i for i, m in enumerate(all_messages) if m["role"] == "user"]
        assert len(user_indices) == 2
        assert user_indices[0] < user_indices[1], "History text must precede multimodal content"
