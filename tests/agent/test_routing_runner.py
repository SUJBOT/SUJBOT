"""Tests for RoutingAgentRunner — 8B router with simple tools + 30B delegation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.single_agent.routing_runner import (
    ANSWER_DIRECTLY_SCHEMA,
    DELEGATE_TOOL_SCHEMA,
    ROUTER_SIMPLE_TOOLS,
    RoutingAgentRunner,
)
from src.agent.providers.base import ProviderResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def routing_config():
    return {
        "routing": {
            "enabled": True,
            "router_model": "qwen3-vl-8b-local",
            "worker_model": "qwen3-vl-30b-local",
            "router_prompt_file": "prompts/agents/router.txt",
            "router_max_tokens": 2048,
            "router_temperature": 0.1,
            "thinking_budgets": {
                "low": 1024,
                "medium": 4096,
                "high": 16384,
                "maximum": 32768,
            },
        },
        "single_agent": {},
    }


@pytest.fixture
def mock_inner_runner():
    """Mock SingleAgentRunner that the RoutingAgentRunner wraps."""
    runner = MagicMock()
    runner.get_tool_health.return_value = {"healthy": True, "total_available": 8}
    # Mock tool_adapter with get_tool_schema
    runner.tool_adapter = MagicMock()
    runner.tool_adapter.get_tool_schema.return_value = {
        "name": "get_document_list",
        "description": "List available documents.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
    return runner


def _make_direct_response(text: str) -> ProviderResponse:
    """Build a ProviderResponse simulating answer_directly tool call."""
    return ProviderResponse(
        content=[
            {"type": "text", "text": ""},
            {
                "type": "tool_use",
                "id": "call_direct",
                "name": "answer_directly",
                "input": {"response": text},
            },
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 50, "output_tokens": 20},
        model="Qwen/Qwen3-VL-8B-Instruct",
    )


def _make_delegate_response(
    task_desc: str = "Search for radiation protection requirements",
    complexity: str = "medium",
    budget: str = "medium",
) -> ProviderResponse:
    return ProviderResponse(
        content=[
            {"type": "text", "text": ""},
            {
                "type": "tool_use",
                "id": "call_001",
                "name": "delegate_to_thinking_agent",
                "input": {
                    "task_description": task_desc,
                    "complexity": complexity,
                    "thinking_budget": budget,
                },
            },
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 100, "output_tokens": 40},
        model="Qwen/Qwen3-VL-8B-Instruct",
    )


def _make_simple_tool_response(tool_name: str, tool_inputs: dict) -> ProviderResponse:
    """Build a ProviderResponse simulating a simple tool call."""
    return ProviderResponse(
        content=[
            {"type": "text", "text": ""},
            {
                "type": "tool_use",
                "id": "call_simple",
                "name": tool_name,
                "input": tool_inputs,
            },
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 80, "output_tokens": 30},
        model="Qwen/Qwen3-VL-8B-Instruct",
    )


# ---------------------------------------------------------------------------
# Tests: tool schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_delegate_schema_has_required_fields(self):
        schema = DELEGATE_TOOL_SCHEMA
        assert schema["name"] == "delegate_to_thinking_agent"
        props = schema["input_schema"]["properties"]
        assert "task_description" in props
        assert "complexity" in props
        assert "thinking_budget" in props

    def test_delegate_schema_required_list(self):
        required = DELEGATE_TOOL_SCHEMA["input_schema"]["required"]
        assert set(required) == {"task_description", "complexity", "thinking_budget"}

    def test_thinking_budget_enum(self):
        enum = DELEGATE_TOOL_SCHEMA["input_schema"]["properties"]["thinking_budget"]["enum"]
        assert enum == ["low", "medium", "high", "maximum"]

    def test_complexity_enum(self):
        enum = DELEGATE_TOOL_SCHEMA["input_schema"]["properties"]["complexity"]["enum"]
        assert enum == ["simple", "medium", "complex", "expert"]

    def test_answer_directly_schema(self):
        schema = ANSWER_DIRECTLY_SCHEMA
        assert schema["name"] == "answer_directly"
        assert "response" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["response"]

    def test_router_simple_tools_default(self):
        assert "get_document_list" in ROUTER_SIMPLE_TOOLS
        assert "get_stats" in ROUTER_SIMPLE_TOOLS
        assert "web_search" in ROUTER_SIMPLE_TOOLS


# ---------------------------------------------------------------------------
# Tests: routing decisions
# ---------------------------------------------------------------------------


class TestRoutingDecisions:
    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_direct_answer(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        """8B calls answer_directly → yields text_delta + final."""
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = _make_direct_response("Ahoj! Jak vám mohu pomoci?")
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        events = []
        async for event in runner.run_query("Ahoj", stream_progress=True):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "routing" in event_types
        assert "text_delta" in event_types
        assert "final" in event_types

        final = next(e for e in events if e["type"] == "final")
        assert final["success"] is True
        assert final["final_answer"] == "Ahoj! Jak vám mohu pomoci?"
        assert final["model"] == "qwen3-vl-8b-local"
        assert final["tool_call_count"] == 0

    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_tool_choice_required_is_passed(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        """Verify tool_choice='required' is passed to router provider."""
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = _make_direct_response("Hi")
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        async for _ in runner.run_query("Hello"):
            pass

        call_kwargs = mock_provider.create_message.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"

    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_delegation(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        """8B delegates → calls inner runner with thinking kwargs."""
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = _make_delegate_response(
            complexity="complex", budget="high"
        )
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        async def mock_run_query(**kwargs):
            yield {"type": "final", "success": True, "final_answer": "30B result"}

        mock_inner_runner.run_query = mock_run_query
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        events = []
        async for event in runner.run_query(
            "Jaké jsou požadavky na radiační ochranu?",
            stream_progress=True,
        ):
            events.append(event)

        routing_events = [e for e in events if e["type"] == "routing"]
        delegate_event = next(e for e in routing_events if e.get("decision") == "delegate")
        assert delegate_event["complexity"] == "complex"
        assert delegate_event["thinking_budget"] == "high"

        final = next(e for e in events if e["type"] == "final")
        assert final["success"] is True

    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_simple_tool_call(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        """8B calls get_document_list → execute tool → 8B streams response."""
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = _make_simple_tool_response("get_document_list", {})
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        # Mock tool execution
        mock_inner_runner.tool_adapter.execute = AsyncMock(return_value={
            "success": True,
            "result": [{"type": "text", "text": "5 documents available"}],
        })

        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        # Mock streaming helper to yield text_delta + usage
        async def mock_stream(*args, **kwargs):
            yield {"type": "text_delta", "content": "Mám k dispozici "}
            yield {"type": "text_delta", "content": "5 dokumentů."}
            yield {"type": "_usage", "data": {"input_tokens": 200, "output_tokens": 30}}

        runner._stream_router_response = mock_stream

        events = []
        async for event in runner.run_query("Kolik máš dokumentů?", stream_progress=True):
            events.append(event)

        # Verify tool was executed
        mock_inner_runner.tool_adapter.execute.assert_called_once_with(
            tool_name="get_document_list",
            inputs={},
            agent_name="router_8b",
        )

        # Verify streaming text_delta events
        text_deltas = [e for e in events if e["type"] == "text_delta"]
        assert len(text_deltas) == 2
        streamed_text = "".join(e["content"] for e in text_deltas)
        assert streamed_text == "Mám k dispozici 5 dokumentů."

        final = next(e for e in events if e["type"] == "final")
        assert final["success"] is True
        assert final["tool_call_count"] == 1
        assert final["final_answer"] == "Mám k dispozici 5 dokumentů."
        assert "get_document_list" in final["tools_used"]
        assert final["model"] == "qwen3-vl-8b-local"

    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_router_failure_fallback(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        """Router 8B failure → falls back to 30B directly."""
        mock_provider = MagicMock()
        mock_provider.create_message.side_effect = RuntimeError("8B server down")
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        async def mock_run_query(**kwargs):
            yield {"type": "final", "success": True, "final_answer": "Fallback 30B answer"}

        mock_inner_runner.run_query = mock_run_query
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        events = []
        async for event in runner.run_query("Test query"):
            events.append(event)

        final = next(e for e in events if e["type"] == "final")
        assert final["success"] is True
        assert final["final_answer"] == "Fallback 30B answer"

    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_no_tool_call_fallback(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        """No tool call in response → fallback to delegation."""
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = ProviderResponse(
            content=[{"type": "text", "text": "unexpected"}],
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 20},
            model="Qwen/Qwen3-VL-8B-Instruct",
        )
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        async def mock_run_query(**kwargs):
            yield {"type": "final", "success": True, "final_answer": "Fallback result"}

        mock_inner_runner.run_query = mock_run_query
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        events = []
        async for event in runner.run_query("Some query", stream_progress=True):
            events.append(event)

        final = next(e for e in events if e["type"] == "final")
        assert final["success"] is True
        assert final["final_answer"] == "Fallback result"


# ---------------------------------------------------------------------------
# Tests: tool building
# ---------------------------------------------------------------------------


class TestToolBuilding:
    def test_build_router_tools_includes_virtual_and_real(self, routing_config, mock_inner_runner):
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        tools = runner._build_router_tools()

        names = {t["name"] for t in tools}
        assert "answer_directly" in names
        assert "delegate_to_thinking_agent" in names
        # Real tool from mock
        assert "get_document_list" in names

    def test_build_router_tools_skips_missing(self, routing_config, mock_inner_runner):
        mock_inner_runner.tool_adapter.get_tool_schema.return_value = None
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        tools = runner._build_router_tools()

        # Only virtual tools (real ones returned None)
        names = {t["name"] for t in tools}
        assert "answer_directly" in names
        assert "delegate_to_thinking_agent" in names

    def test_build_router_tools_respects_disabled(self, routing_config, mock_inner_runner):
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        tools = runner._build_router_tools(disabled_tools={"web_search"})

        names = {t["name"] for t in tools}
        assert "web_search" not in names
        # Virtual tools always present
        assert "answer_directly" in names
        assert "delegate_to_thinking_agent" in names

    def test_dynamic_prompt_excludes_disabled_tools(self, routing_config, mock_inner_runner):
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        # With all tools
        full_prompt = runner._build_router_system_prompt()
        assert "`web_search`" in full_prompt
        assert "get_document_list" in full_prompt

        # With web_search disabled — tool line removed, fallback rule added
        limited_prompt = runner._build_router_system_prompt(disabled_tools={"web_search"})
        assert "- `web_search`" not in limited_prompt
        assert "get_document_list" in limited_prompt
        assert "web search is disabled" in limited_prompt


# ---------------------------------------------------------------------------
# Tests: thinking budget mapping
# ---------------------------------------------------------------------------


class TestThinkingBudgets:
    def test_budget_mapping(self, routing_config, mock_inner_runner):
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        assert runner.thinking_budgets["low"] == 1024
        assert runner.thinking_budgets["medium"] == 4096
        assert runner.thinking_budgets["high"] == 16384
        assert runner.thinking_budgets["maximum"] == 32768

    def test_default_budgets_when_not_configured(self, mock_inner_runner):
        config = {"routing": {}, "single_agent": {}}
        runner = RoutingAgentRunner(config, mock_inner_runner)
        assert runner.thinking_budgets["medium"] == 4096


# ---------------------------------------------------------------------------
# Tests: conversation history + attribute delegation
# ---------------------------------------------------------------------------


class TestConversationHistory:
    @pytest.mark.anyio
    @patch("src.agent.providers.factory.create_provider")
    @patch("src.cost_tracker.get_global_tracker")
    async def test_history_passed_to_router(self, mock_tracker_fn, mock_create, routing_config, mock_inner_runner):
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = _make_direct_response("OK")
        mock_create.return_value = mock_provider
        mock_tracker_fn.return_value = MagicMock()

        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        async for _ in runner.run_query("Thanks", conversation_history=history):
            pass

        call_args = mock_provider.create_message.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there!"
        assert messages[2]["content"] == "Thanks"


# ---------------------------------------------------------------------------
# Tests: streaming
# ---------------------------------------------------------------------------


class TestStreamingResponse:
    @pytest.mark.anyio
    async def test_stream_router_response_yields_text_deltas(self, routing_config, mock_inner_runner):
        """_stream_router_response yields text_delta events from OpenAI stream chunks."""
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        # Build mock OpenAI streaming chunks
        mock_chunks = []
        for text in ["Mám ", "k dispozici ", "5 dokumentů."]:
            chunk = MagicMock()
            chunk.usage = None
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            chunk.choices[0].finish_reason = None
            mock_chunks.append(chunk)

        # Final chunk with usage
        final_chunk = MagicMock()
        final_chunk.usage = MagicMock()
        final_chunk.usage.prompt_tokens = 150
        final_chunk.usage.completion_tokens = 25
        final_chunk.choices = []
        mock_chunks.append(final_chunk)

        mock_provider = MagicMock()
        mock_provider.stream_message.return_value = iter(mock_chunks)

        events = []
        async for ev in runner._stream_router_response(
            mock_provider, [{"role": "user", "content": "test"}], True
        ):
            events.append(ev)

        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 3
        full_text = "".join(e["content"] for e in text_events)
        assert full_text == "Mám k dispozici 5 dokumentů."

        usage_events = [e for e in events if e["type"] == "_usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["data"]["input_tokens"] == 150
        assert usage_events[0]["data"]["output_tokens"] == 25

    @pytest.mark.anyio
    async def test_stream_fallback_to_create_message(self, routing_config, mock_inner_runner):
        """When streaming fails, falls back to non-streaming create_message."""
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)

        mock_provider = MagicMock()
        mock_provider.stream_message.side_effect = ConnectionError("Stream failed")
        mock_provider.create_message.return_value = ProviderResponse(
            content=[{"type": "text", "text": "Fallback response."}],
            stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 15},
            model="Qwen/Qwen3-VL-8B-Instruct",
        )

        events = []
        async for ev in runner._stream_router_response(
            mock_provider, [{"role": "user", "content": "test"}], True
        ):
            events.append(ev)

        text_events = [e for e in events if e["type"] == "text_delta"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "Fallback response."

        usage_events = [e for e in events if e["type"] == "_usage"]
        assert len(usage_events) == 1


class TestAttributeDelegation:
    def test_get_tool_health_delegated(self, routing_config, mock_inner_runner):
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        health = runner.get_tool_health()
        assert health["healthy"] is True
        mock_inner_runner.get_tool_health.assert_called_once()

    def test_unknown_attr_delegated(self, routing_config, mock_inner_runner):
        mock_inner_runner.some_method = MagicMock(return_value=42)
        runner = RoutingAgentRunner(routing_config, mock_inner_runner)
        assert runner.some_method() == 42
