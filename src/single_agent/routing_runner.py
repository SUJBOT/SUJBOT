"""
Routing Agent Runner — 8B router handles simple queries, 30B handles complex ones.

Architecture:
  User Query → RoutingAgentRunner.run_query()
    → 8B Router (non-streaming classification, thinking DISABLED, tool_choice=required)
       ├── answer_directly → yield text_delta + final
       ├── simple tool (get_document_list, get_stats, web_search)
       │    → execute tool → feed result back → 8B streams response (text_delta)
       └── delegate_to_thinking_agent
            → 30B worker with full autonomous tool loop + thinking budget
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from ..exceptions import (
    ProviderError,
    ProviderTimeoutError,
    ToolExecutionError,
)
from ..utils.security import sanitize_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Virtual tool schemas (intercepted by router, not in ToolAdapter)
# ---------------------------------------------------------------------------

ANSWER_DIRECTLY_SCHEMA = {
    "name": "answer_directly",
    "description": (
        "Answer ONLY greetings, thank-you messages, or meta-questions about the system. "
        "NEVER use for factual questions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Your brief greeting or acknowledgment.",
            },
        },
        "required": ["response"],
    },
}

DELEGATE_TOOL_SCHEMA = {
    "name": "delegate_to_thinking_agent",
    "description": (
        "Delegate to the expert thinking agent with deep reasoning and full "
        "document search (search, compliance_check, graph_search, expand_context). "
        "Use for ANY question that needs document search, deep analysis, or "
        "multi-step reasoning."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Reformulated task for the expert agent.",
            },
            "complexity": {
                "type": "string",
                "enum": ["simple", "medium", "complex", "expert"],
                "description": "Query complexity assessment.",
            },
            "thinking_budget": {
                "type": "string",
                "enum": ["low", "medium", "high", "maximum"],
                "description": (
                    "Thinking depth: low=quick lookup, medium=standard search, "
                    "high=compliance analysis, maximum=deep multi-document reasoning."
                ),
            },
        },
        "required": ["task_description", "complexity", "thinking_budget"],
    },
}

# Tools the 8B router can execute directly (simple, fast, no deep reasoning)
ROUTER_SIMPLE_TOOLS = ["get_document_list", "get_stats", "web_search"]

# Shared extra_body to disable thinking on 8B router calls
_THINKING_DISABLED_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# Provider exceptions to catch (typed + OpenAI SDK + standard Python network errors)
try:
    import openai as _openai
    _PROVIDER_ERRORS = (
        ProviderError, ProviderTimeoutError,
        ConnectionError, TimeoutError, OSError,
        _openai.APIError, _openai.APIConnectionError,
    )
except ImportError:
    _PROVIDER_ERRORS = (
        ProviderError, ProviderTimeoutError,
        ConnectionError, TimeoutError, OSError,
    )


class RoutingAgentRunner:
    """
    Router agent: 8B handles simple queries directly, 30B handles complex ones.

    The 8B router has access to:
    - answer_directly: greetings, small talk
    - Simple tools: get_document_list, get_stats, web_search
    - delegate_to_thinking_agent: document search, compliance, deep reasoning

    Uses composition (wraps SingleAgentRunner) to reuse tool initialization.
    """

    def __init__(self, config: Dict[str, Any], inner_runner):
        from ..agent.providers.factory import create_provider

        self.config = config
        self.inner_runner = inner_runner

        routing = config.get("routing", {})
        self.router_model = routing.get("router_model", "qwen3-vl-8b-local")
        self.worker_model = routing.get("worker_model", "qwen3-vl-30b-local")
        self.router_max_tokens = routing.get("router_max_tokens", 2048)
        self.router_temperature = routing.get("router_temperature", 0.1)
        self.thinking_budgets = routing.get("thinking_budgets", {
            "low": 1024, "medium": 4096, "high": 16384, "maximum": 32768,
        })
        self.simple_tool_names = set(
            routing.get("router_simple_tools", ROUTER_SIMPLE_TOOLS)
        )

        # Load router prompt
        router_prompt_file = routing.get(
            "router_prompt_file", "prompts/agents/router.txt"
        )
        self.router_prompt = self._load_router_prompt(router_prompt_file)

        # Create provider once (reused across queries)
        self.router_provider = create_provider(self.router_model)

    async def _stream_router_response(
        self, provider, messages: List[Dict[str, Any]],
        system: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream an 8B response, yielding text_delta events and a final _usage event.

        Thinking is disabled but Qwen3 may still emit <think> tags — parsed as safety net.
        Falls back to non-streaming create_message() on errors.
        Raises on total failure (both streaming and fallback fail).
        """
        from ..agent.providers.think_parser import ChunkType, ThinkTagStreamParser

        loop = asyncio.get_running_loop()

        try:
            stream = await loop.run_in_executor(
                None,
                lambda: provider.stream_message(
                    messages=messages,
                    system=system,
                    max_tokens=self.router_max_tokens,
                    temperature=self.router_temperature,
                    extra_body=_THINKING_DISABLED_BODY,
                ),
            )

            parser = ThinkTagStreamParser(start_thinking=False)
            usage_data = None

            def _next_chunk(it):
                try:
                    return next(it)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, _next_chunk, stream)
                if chunk is None:
                    break

                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_data = {
                        "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    for pc in parser.feed(delta.content):
                        if pc.type == ChunkType.TEXT:
                            yield {"type": "text_delta", "content": pc.content}

            # Flush parser
            for pc in parser.flush():
                if pc.type == ChunkType.TEXT:
                    yield {"type": "text_delta", "content": pc.content}

            if usage_data:
                yield {"type": "_usage", "data": usage_data}

        except _PROVIDER_ERRORS as e:
            logger.warning(
                "Router streaming failed, falling back to non-streaming: %s", e,
                exc_info=True,
            )
            try:
                response = provider.create_message(
                    messages=messages,
                    system=system,
                    max_tokens=self.router_max_tokens,
                    temperature=self.router_temperature,
                    extra_body=_THINKING_DISABLED_BODY,
                )
                text = response.text or ""
                if text:
                    yield {"type": "text_delta", "content": text}
                if hasattr(response, "usage") and response.usage:
                    yield {
                        "type": "_usage",
                        "data": {
                            "input_tokens": response.usage.get("input_tokens", 0),
                            "output_tokens": response.usage.get("output_tokens", 0),
                        },
                    }
            except _PROVIDER_ERRORS as fallback_err:
                logger.error(
                    "Router non-streaming fallback also failed: %s",
                    fallback_err, exc_info=True,
                )
                raise ProviderError(
                    f"Router 8B completely unreachable: {fallback_err}",
                    details={"router_model": self.router_model},
                    cause=fallback_err,
                ) from fallback_err

    @staticmethod
    def _load_router_prompt(prompt_file: str) -> str:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            prompt_path = Path(__file__).parent.parent.parent / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Router prompt not found: {prompt_file}")
        return prompt_path.read_text(encoding="utf-8")

    def __getattr__(self, name):
        return getattr(self.inner_runner, name)

    def _build_router_tools(
        self, disabled_tools: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Build tool list for the 8B router: virtual tools + real simple tool schemas."""
        tools = [ANSWER_DIRECTLY_SCHEMA, DELEGATE_TOOL_SCHEMA]
        _disabled = disabled_tools or set()

        # Add real tool schemas from inner runner's tool adapter
        tool_adapter = self.inner_runner.tool_adapter
        for tool_name in self.simple_tool_names:
            if tool_name in _disabled:
                continue
            schema = tool_adapter.get_tool_schema(tool_name)
            if schema:
                tools.append(schema)
            else:
                logger.warning("Simple tool '%s' not found in tool adapter", tool_name)

        return tools

    def _build_router_system_prompt(
        self, disabled_tools: Optional[Set[str]] = None
    ) -> str:
        """Build system prompt with dynamic tool list based on what's actually available."""
        _disabled = disabled_tools or set()
        active_tools = [t for t in self.simple_tool_names if t not in _disabled]

        tool_descriptions = {
            "get_document_list": '- `get_document_list` — list available documents ("Kolik máš dokumentů?", "Jaké dokumenty máš?")',
            "get_stats": '- `get_stats` — system statistics ("Kolik je stránek?", "Jaké máš statistiky?")',
            "web_search": "- `web_search` — ONLY for real-time/current info: weather, news, live events, prices",
        }

        lines = [tool_descriptions[t] for t in active_tools if t in tool_descriptions]
        available_section = "\n".join(lines) if lines else "- (no additional tools available)"

        extra_rules = []
        if "web_search" in active_tools:
            extra_rules.append(
                "4. NEVER use web_search for factual/knowledge questions — delegate instead.\n"
                "5. web_search is ONLY for time-sensitive current information (weather, news, events)."
            )
        else:
            extra_rules.append(
                "4. You do NOT have web_search. For questions about current/real-time info "
                "(weather, news, prices, events), use answer_directly to tell the user that "
                "web search is disabled and you cannot provide this information. "
                "Do NOT delegate these — the expert cannot search the web either."
            )

        prompt = self.router_prompt.replace("{available_tools}", available_section)
        prompt = prompt.replace("{extra_rules}", "\n".join(extra_rules))
        return prompt

    @staticmethod
    def _extract_tool_result_text(tool_result: Dict[str, Any]) -> str:
        """Extract human-readable text from a tool result dict."""
        content = tool_result.get("result", str(tool_result))
        if isinstance(content, list):
            return "\n".join(
                b.get("text", str(b)) for b in content if isinstance(b, dict)
            )
        return str(content)

    async def _delegate_to_worker(
        self,
        query: str,
        model: str,
        stream_progress: bool,
        conversation_history: Optional[List[Dict[str, str]]],
        attachment_blocks: Optional[List[Dict[str, Any]]],
        disabled_tools: Optional[Set[str]],
        extra_llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield all events from inner_runner.run_query() — shared delegation helper."""
        async for event in self.inner_runner.run_query(
            query=query,
            model=model,
            stream_progress=stream_progress,
            conversation_history=conversation_history,
            attachment_blocks=attachment_blocks,
            disabled_tools=disabled_tools,
            extra_llm_kwargs=extra_llm_kwargs,
        ):
            yield event

    async def run_query(
        self,
        query: str,
        model: str = "",
        stream_progress: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        attachment_blocks: Optional[List[Dict[str, Any]]] = None,
        disabled_tools: Optional[Set[str]] = None,
        extra_llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """8B routing: direct answer, simple tool call, or 30B delegation."""
        from ..cost_tracker import get_global_tracker

        router_tools = self._build_router_tools(disabled_tools)
        router_system = self._build_router_system_prompt(disabled_tools)

        # Build messages with conversation history
        router_messages = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                router_messages.append({"role": msg["role"], "content": msg["content"]})
        router_messages.append({"role": "user", "content": query})

        if stream_progress:
            yield {"type": "routing", "decision": "classifying"}

        logger.info("Router phase: classifying query with %s", self.router_model)
        router_start = time.time()

        try:
            router_response = self.router_provider.create_message(
                messages=router_messages,
                tools=router_tools,
                system=router_system,
                max_tokens=self.router_max_tokens,
                temperature=self.router_temperature,
                tool_choice="required",
                extra_body=_THINKING_DISABLED_BODY,
            )
        except _PROVIDER_ERRORS as e:
            logger.error("Router 8B call failed: %s, falling back to 30B", e, exc_info=True)
            if stream_progress:
                yield {"type": "routing", "decision": "fallback"}
            async for event in self._delegate_to_worker(
                query=query,
                model=model or self.worker_model,
                stream_progress=stream_progress,
                conversation_history=conversation_history,
                attachment_blocks=attachment_blocks,
                disabled_tools=disabled_tools,
                extra_llm_kwargs=extra_llm_kwargs,
            ):
                yield event
            return

        router_time_ms = (time.time() - router_start) * 1000
        logger.info("Router classification took %.0fms", router_time_ms)

        # Track router cost
        tracker = get_global_tracker()
        if hasattr(router_response, "usage") and router_response.usage:
            tracker.track_llm(
                provider="local_llm_8b",
                model=self.router_model,
                input_tokens=router_response.usage.get("input_tokens", 0),
                output_tokens=router_response.usage.get("output_tokens", 0),
                operation="router",
                response_time_ms=router_time_ms,
            )

        # Extract tool call with type guard
        content_blocks = (
            router_response.content
            if isinstance(router_response.content, list)
            else []
        )
        if not content_blocks:
            logger.warning(
                "Router returned empty/non-list content: %s",
                type(router_response.content),
            )

        tool_call = next(
            (b for b in content_blocks
             if isinstance(b, dict) and b.get("type") == "tool_use"),
            None,
        )

        if not tool_call:
            logger.warning("Router returned no tool call, falling back to delegation")
            tool_name = "delegate_to_thinking_agent"
            tool_inputs = {"task_description": query, "complexity": "medium", "thinking_budget": "medium"}
        else:
            tool_name = tool_call.get("name", "")
            tool_inputs = tool_call.get("input", {})
            logger.info("Router chose tool=%s, inputs=%s", tool_name, str(tool_inputs)[:200])

        # --- Branch 1: Direct answer (answer_directly tool) ---
        if tool_name == "answer_directly":
            # If user sent attachments, delegate to 30B (8B can't handle multimodal)
            if attachment_blocks:
                logger.info("Router chose answer_directly but attachments present, delegating to 30B")
                if stream_progress:
                    yield {"type": "routing", "decision": "delegate", "complexity": "simple", "thinking_budget": "low"}
                async for event in self._delegate_to_worker(
                    query=query,
                    model=self.worker_model,
                    stream_progress=stream_progress,
                    conversation_history=conversation_history,
                    attachment_blocks=attachment_blocks,
                    disabled_tools=disabled_tools,
                    extra_llm_kwargs=extra_llm_kwargs,
                ):
                    yield event
                return

            final_text = tool_inputs.get("response", "")
            logger.info("Router answered directly (%d chars)", len(final_text))

            if stream_progress:
                yield {"type": "routing", "decision": "direct"}
                if final_text:
                    yield {"type": "text_delta", "content": final_text}

            yield {
                "type": "final",
                "success": True,
                "final_answer": final_text,
                "tools_used": [],
                "tool_call_count": 0,
                "iterations": 0,
                "model": self.router_model,
                "errors": [],
                "text_already_streamed": stream_progress,
            }
            return

        # --- Branch 2: Delegate to 30B thinking agent ---
        if tool_name == "delegate_to_thinking_agent":
            task_desc = tool_inputs.get("task_description", query)
            budget_key = tool_inputs.get("thinking_budget", "medium")
            complexity = tool_inputs.get("complexity", "medium")
            budget_tokens = self.thinking_budgets.get(budget_key, 4096)

            logger.info(
                "Router delegating: complexity=%s, thinking_budget=%s (%d tokens)",
                complexity, budget_key, budget_tokens,
            )

            if stream_progress:
                yield {
                    "type": "routing",
                    "decision": "delegate",
                    "complexity": complexity,
                    "thinking_budget": budget_key,
                }

            thinking_kwargs = {
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "thinking_token_budget": budget_tokens,
                    }
                }
            }

            async for event in self._delegate_to_worker(
                query=query,
                model=self.worker_model,
                stream_progress=stream_progress,
                conversation_history=conversation_history,
                attachment_blocks=attachment_blocks,
                disabled_tools=disabled_tools,
                extra_llm_kwargs=thinking_kwargs,
            ):
                yield event
            return

        # --- Branch 3: Simple tool call (8B executes directly) ---
        if tool_name in self.simple_tool_names:
            # If user sent attachments, delegate to 30B (8B can't handle multimodal)
            if attachment_blocks:
                logger.info("Router chose simple tool '%s' but attachments present, delegating to 30B", tool_name)
                if stream_progress:
                    yield {"type": "routing", "decision": "delegate", "complexity": "simple", "thinking_budget": "low"}
                async for event in self._delegate_to_worker(
                    query=query,
                    model=self.worker_model,
                    stream_progress=stream_progress,
                    conversation_history=conversation_history,
                    attachment_blocks=attachment_blocks,
                    disabled_tools=disabled_tools,
                    extra_llm_kwargs=extra_llm_kwargs,
                ):
                    yield event
                return

            logger.info("Router executing simple tool: %s", tool_name)

            if stream_progress:
                yield {"type": "routing", "decision": "simple_tool", "tool": tool_name}
                yield {"type": "tool_call", "tool": tool_name, "status": "running"}

            # Execute tool via inner runner's tool adapter
            try:
                tool_result = await self.inner_runner.tool_adapter.execute(
                    tool_name=tool_name,
                    inputs=tool_inputs,
                    agent_name="router_8b",
                )
            except ToolExecutionError as e:
                safe_err = sanitize_error(e)
                logger.error("Router simple tool '%s' failed: %s", tool_name, e, exc_info=True)
                if stream_progress:
                    yield {"type": "tool_call", "tool": tool_name, "status": "failed"}
                yield {
                    "type": "final",
                    "success": False,
                    "final_answer": f"Tool '{tool_name}' failed: {safe_err}",
                    "tools_used": [tool_name],
                    "tool_call_count": 1,
                    "iterations": 1,
                    "model": self.router_model,
                    "errors": [safe_err],
                }
                return
            except Exception as e:
                safe_err = sanitize_error(e)
                logger.error(
                    "Router simple tool '%s' unexpected error: %s", tool_name, e,
                    exc_info=True,
                )
                if stream_progress:
                    yield {"type": "tool_call", "tool": tool_name, "status": "failed"}
                yield {
                    "type": "final",
                    "success": False,
                    "final_answer": f"Tool '{tool_name}' failed: {safe_err}",
                    "tools_used": [tool_name],
                    "tool_call_count": 1,
                    "iterations": 1,
                    "model": self.router_model,
                    "errors": [safe_err],
                }
                return

            if stream_progress:
                yield {"type": "tool_call", "tool": tool_name, "status": "completed"}

            # Feed tool result back to 8B for final response
            tool_result_text = self._extract_tool_result_text(tool_result)

            # Build follow-up messages: original + tool call + tool result
            followup_messages = list(router_messages)
            followup_messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tool_call.get("id", "call_simple"),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_inputs),
                    },
                }],
            })
            followup_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", "call_simple"),
                "content": tool_result_text[:4000],  # Truncate to avoid token overflow
            })

            if stream_progress:
                yield {"type": "routing", "decision": "direct"}

            try:
                final_text = ""
                usage_data = None

                # Stream 8B follow-up for real-time text_delta events
                async for ev in self._stream_router_response(
                    self.router_provider, followup_messages, system=router_system,
                ):
                    if ev["type"] == "text_delta":
                        final_text += ev["content"]
                        if stream_progress:
                            yield ev
                    elif ev["type"] == "_usage":
                        usage_data = ev["data"]

                # Track follow-up cost
                if usage_data:
                    tracker.track_llm(
                        provider="local_llm_8b",
                        model=self.router_model,
                        input_tokens=usage_data.get("input_tokens", 0),
                        output_tokens=usage_data.get("output_tokens", 0),
                        operation="router_followup",
                    )
            except _PROVIDER_ERRORS as e:
                logger.error("Router follow-up failed: %s", e, exc_info=True)
                final_text = tool_result_text  # Fallback: raw tool result
                if stream_progress:
                    yield {"type": "text_delta", "content": final_text}

            logger.info("Router simple tool response (%d chars)", len(final_text))

            yield {
                "type": "final",
                "success": True,
                "final_answer": final_text,
                "tools_used": [tool_name],
                "tool_call_count": 1,
                "iterations": 1,
                "model": self.router_model,
                "errors": [],
                "text_already_streamed": stream_progress,
            }
            return

        # --- Branch 4: Unknown tool — fallback to 30B delegation ---
        logger.warning(
            "Router returned unknown tool '%s', falling back to delegation", tool_name
        )
        if stream_progress:
            yield {"type": "routing", "decision": "delegate", "complexity": "medium", "thinking_budget": "medium"}
        async for event in self._delegate_to_worker(
            query=query,
            model=self.worker_model,
            stream_progress=stream_progress,
            conversation_history=conversation_history,
            attachment_blocks=attachment_blocks,
            disabled_tools=disabled_tools,
            extra_llm_kwargs=extra_llm_kwargs,
        ):
            yield event
