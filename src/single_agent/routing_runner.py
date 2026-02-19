"""
Routing Agent Runner — 8B router handles simple queries, 30B handles complex ones.

Architecture:
  User Query → RoutingAgentRunner.run_query()
    → 8B Router (streaming classification, thinking DISABLED, tool_choice=auto)
       ├── text-only response → buffer then flush text_delta (greetings, meta)
       ├── simple tool (get_document_list, get_stats)
       │    → execute tool → feed result back → 8B streams response (text_delta)
       └── delegate_to_thinking_agent (SILENT — pre-delegation text discarded)
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

DELEGATE_TOOL_SCHEMA = {
    "name": "delegate_to_thinking_agent",
    "description": (
        "Delegate to the expert thinking agent with deep reasoning and full "
        "document search (search, compliance_check, graph_search, expand_context, web_search). "
        "Use for ANY question that needs document search, deep analysis, research, "
        "multi-step reasoning, or gap analysis. Also use when you are unsure — "
        "the expert is always the safe choice."
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
ROUTER_SIMPLE_TOOLS = ["get_document_list", "get_stats"]

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

    The 8B router uses tool_choice="auto":
    - Text-only response: greetings, small talk (streamed token-by-token)
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

    async def _stream_classify(
        self, provider, messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]], system: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the 8B classification call with tool_choice="auto".

        Text content is yielded immediately as text_delta events.
        Tool call deltas are accumulated and yielded as _tool_call events at stream end.
        Falls back to non-streaming create_message() on streaming errors.
        Raises ProviderError if both streaming and fallback fail.

        Yields:
          - {"type": "text_delta", "content": ...}  — streamed text (real-time)
          - {"type": "_tool_call", "name": ..., "input": ..., "id": ...}  — accumulated tool call
          - {"type": "_usage", "data": ...}  — token usage
        """
        from ..agent.providers.think_parser import ChunkType, ThinkTagStreamParser

        loop = asyncio.get_running_loop()

        try:
            stream = await loop.run_in_executor(
                None,
                lambda: provider.stream_message(
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=self.router_max_tokens,
                    temperature=self.router_temperature,
                    tool_choice="auto",
                    extra_body=_THINKING_DISABLED_BODY,
                ),
            )

            parser = ThinkTagStreamParser(start_thinking=False)
            usage_data = None
            tool_call_deltas: Dict[int, Dict[str, Any]] = {}

            def _next_chunk(it):
                try:
                    return next(it)
                except StopIteration:
                    return None

            text_yielded = False

            while True:
                chunk = await loop.run_in_executor(None, _next_chunk, stream)
                if chunk is None:
                    break

                try:
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        usage_data = {
                            "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                            "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                        }

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Text content — stream immediately
                    if hasattr(delta, "content") and delta.content:
                        for pc in parser.feed(delta.content):
                            if pc.type == ChunkType.TEXT:
                                text_yielded = True
                                yield {"type": "text_delta", "content": pc.content}

                    # Tool call deltas — accumulate
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_deltas:
                                tool_call_deltas[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc_delta.id:
                                tool_call_deltas[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_call_deltas[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_call_deltas[idx]["arguments"] += tc_delta.function.arguments
                except (TypeError, AttributeError, KeyError) as chunk_err:
                    logger.warning("Skipping malformed streaming chunk: %s", chunk_err)
                    continue

            # Flush parser
            for pc in parser.flush():
                if pc.type == ChunkType.TEXT:
                    yield {"type": "text_delta", "content": pc.content}

            # Yield accumulated tool calls
            for idx in sorted(tool_call_deltas):
                tc = tool_call_deltas[idx]
                try:
                    parsed_input = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse tool call arguments for tool '%s': %s",
                        tc["name"], tc["arguments"][:200],
                    )
                    parsed_input = {}
                yield {
                    "type": "_tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": parsed_input,
                }

            if usage_data:
                yield {"type": "_usage", "data": usage_data}

        except _PROVIDER_ERRORS as e:
            logger.warning(
                "Router streaming classification failed, falling back to non-streaming: %s", e,
                exc_info=True,
            )
            try:
                response = provider.create_message(
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=self.router_max_tokens,
                    temperature=self.router_temperature,
                    tool_choice="auto",
                    extra_body=_THINKING_DISABLED_BODY,
                )

                text = response.text or ""
                if text and not text_yielded:
                    yield {"type": "text_delta", "content": text}

                content_blocks = response.content if isinstance(response.content, list) else []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        yield {
                            "type": "_tool_call",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input", {}),
                        }

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
        """Build tool list for the 8B router: delegation + real simple tool schemas."""
        tools = [DELEGATE_TOOL_SCHEMA]
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
                "6. web_search (your tool) is ONLY for simple real-time lookups (weather, news, events).\n"
                "7. For research tasks that COMBINE web search with document analysis, DELEGATE — "
                "the expert has web_search too and can reason across both web and corpus results."
            )
        else:
            extra_rules.append(
                "6. You do NOT have web_search. The expert agent does NOT have it either.\n"
                "7. For questions needing current/real-time info, delegate anyway — "
                "the expert can use document search and reasoning to provide the best available answer."
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
        """8B routing: streamed text response, simple tool call, or 30B delegation."""
        from ..cost_tracker import get_global_tracker

        router_tools = self._build_router_tools(disabled_tools)
        router_system = self._build_router_system_prompt(disabled_tools)

        # Build messages with conversation history
        router_messages = []
        if conversation_history:
            for msg in conversation_history[-6:]:
                router_messages.append({"role": msg["role"], "content": msg["content"]})

        # Hint to router about image attachments (8B can't see images, must delegate)
        user_query = query
        if attachment_blocks:
            image_count = sum(
                1 for b in attachment_blocks
                if isinstance(b, dict) and b.get("type") == "image"
            )
            if image_count > 0:
                user_query = (
                    f"[User has attached {image_count} image(s) to this message. "
                    f"You CANNOT see them — delegate to the expert agent for visual analysis.] "
                    f"{query}"
                )
        router_messages.append({"role": "user", "content": user_query})

        if stream_progress:
            yield {"type": "routing", "decision": "classifying"}

        logger.info("Router phase: streaming classification with %s", self.router_model)
        router_start = time.time()

        try:
            # Stream classification — buffer text until we know if delegation follows.
            # If router delegates, discard any pre-delegation text (invisible handoff).
            # If router responds directly, flush buffered text as text_delta events.
            final_text = ""
            buffered_text_events: List[Dict[str, Any]] = []
            tool_call = None
            usage_data = None

            async for event in self._stream_classify(
                self.router_provider, router_messages, router_tools, router_system,
            ):
                if event["type"] == "text_delta":
                    final_text += event["content"]
                    buffered_text_events.append(event)
                elif event["type"] == "_tool_call":
                    tool_call = event  # Take the first tool call
                elif event["type"] == "_usage":
                    usage_data = event["data"]

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
        if usage_data:
            tracker.track_llm(
                provider="local_llm_8b",
                model=self.router_model,
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                operation="router",
                response_time_ms=router_time_ms,
            )

        # --- No tool call: flush buffered text to user ---
        if tool_call is None:
            # Attachment guard: if user sent images, 8B can't process them meaningfully
            if attachment_blocks:
                logger.info("Router responded with text but attachments present, delegating to 30B")
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

            logger.info("Router answered directly (%d chars)", len(final_text))

            if stream_progress:
                yield {"type": "routing", "decision": "direct"}
                # Flush buffered text as text_delta events (was buffered during classification)
                for text_ev in buffered_text_events:
                    yield text_ev

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

        # --- Has tool call: discard any pre-tool-call text (silent handoff) ---
        if final_text.strip():
            logger.info(
                "Discarding %d chars of router pre-delegation text: %s",
                len(final_text), final_text[:100],
            )

        tool_name = tool_call.get("name", "")
        tool_inputs = tool_call.get("input", {})
        tool_id = tool_call.get("id", "")
        logger.info("Router chose tool=%s, inputs=%s", tool_name, str(tool_inputs)[:200])

        # --- Branch 1: Delegate to 30B thinking agent ---
        if tool_name == "delegate_to_thinking_agent":
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

        # --- Branch 2: Simple tool call (8B executes directly) ---
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
                    "id": tool_id or "call_simple",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_inputs),
                    },
                }],
            })
            followup_messages.append({
                "role": "tool",
                "tool_call_id": tool_id or "call_simple",
                "content": tool_result_text[:4000],  # Truncate to avoid token overflow
            })

            if stream_progress:
                yield {"type": "routing", "decision": "direct"}

            try:
                followup_text = ""
                followup_usage = None

                # Stream 8B follow-up for real-time text_delta events
                async for ev in self._stream_router_response(
                    self.router_provider, followup_messages, system=router_system,
                ):
                    if ev["type"] == "text_delta":
                        followup_text += ev["content"]
                        if stream_progress:
                            yield ev
                    elif ev["type"] == "_usage":
                        followup_usage = ev["data"]

                # Track follow-up cost
                if followup_usage:
                    tracker.track_llm(
                        provider="local_llm_8b",
                        model=self.router_model,
                        input_tokens=followup_usage.get("input_tokens", 0),
                        output_tokens=followup_usage.get("output_tokens", 0),
                        operation="router_followup",
                    )
            except _PROVIDER_ERRORS as e:
                logger.error("Router follow-up failed: %s", e, exc_info=True)
                followup_text = tool_result_text  # Fallback: raw tool result
                if stream_progress:
                    yield {"type": "text_delta", "content": followup_text}

            logger.info("Router simple tool response (%d chars)", len(followup_text))

            yield {
                "type": "final",
                "success": True,
                "final_answer": followup_text,
                "tools_used": [tool_name],
                "tool_call_count": 1,
                "iterations": 1,
                "model": self.router_model,
                "errors": [],
                "text_already_streamed": stream_progress,
            }
            return

        # --- Branch 3: Unknown tool — fallback to 30B delegation ---
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
