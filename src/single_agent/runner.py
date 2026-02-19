"""
Single Agent Runner — autonomous agent with unified prompt and all RAG tools.

Replaces the multi-agent orchestrator + 8 specialist agents with a single
LLM that decides which tools to call and when to stop.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from dotenv import load_dotenv

from ..agent.context_manager import (
    CompactionLayer,
    ContextBudgetMonitor,
    compact_with_summary,
    emergency_truncate,
    get_context_window,
    prune_tool_outputs,
)
from ..exceptions import AgentInitializationError

logger = logging.getLogger(__name__)


class SingleAgentRunner:
    """
    Single autonomous agent with access to all RAG tools.

    Replaces MultiAgentRunner by:
    - Using one LLM call loop instead of orchestrator + specialist routing
    - Exposing ALL tools (search, expand_context, etc.) to one agent
    - Loading system prompt from configurable path (default: prompts/agents/unified.txt)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.single_agent_config = config.get("single_agent", {})
        self.provider = None
        self.tool_adapter = None
        self.system_prompt = ""
        self.vector_store = None
        self._initialized = False
        logger.info("SingleAgentRunner initialized")

    async def initialize(self) -> bool:
        """Initialize tools, provider, and system prompt."""
        logger.info("Initializing single-agent system...")

        try:
            # 1. Setup LangSmith observability (reuse existing)
            self._setup_langsmith()

            # 2. Initialize tool registry with RAG components
            await self._initialize_tools()

            # 3. Initialize tool adapter (bridge to tool schemas + execution)
            from ..agent.tools.adapter import ToolAdapter

            self.tool_adapter = ToolAdapter()

            # 4. Load system prompt
            self._load_system_prompt()

            # 5. Create LLM provider (deferred until query — model comes from variant)
            self._initialized = True
            logger.info("Single-agent system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize single-agent system: {e}", exc_info=True)
            raise AgentInitializationError(
                f"Single-agent initialization failed: {e}", cause=e
            ) from e

    def _setup_langsmith(self) -> None:
        """Setup LangSmith tracing if configured."""
        langsmith_config = self.config.get("multi_agent", {}).get("langsmith", {})
        if not langsmith_config.get("enabled", False):
            return
        try:
            from ..agent.observability import setup_langsmith

            self.langsmith = setup_langsmith(langsmith_config)
            logger.info("LangSmith tracing enabled")
        except Exception as e:
            logger.warning(f"LangSmith setup failed (non-fatal): {e}")
            self.langsmith = None

    async def _initialize_tools(self) -> None:
        """Initialize tool registry with all RAG components (vector store, VL components)."""
        from ..agent.tools import get_registry
        from ..storage import load_vector_store_adapter
        from ..agent.config import ToolConfig
        from ..agent.providers.factory import create_provider

        storage_config = self.config.get("storage", {})
        backend = storage_config.get("backend", "postgresql")

        # LLM provider for tools — use cheap model
        tool_model = self.config.get("models", {}).get("llm_model", "gpt-4o-mini")
        try:
            self.llm_provider = create_provider(model=tool_model)
            logger.info(f"Tool LLM provider: {tool_model}")
        except Exception as e:
            logger.error(
                "Tool LLM provider failed for %s: %s.",
                tool_model,
                e,
                exc_info=True,
            )
            self.llm_provider = None

        # Vector store (PostgreSQL only)
        if backend != "postgresql":
            raise ValueError(
                f"Storage backend '{backend}' is no longer supported. "
                f"Only 'postgresql' is available."
            )

        connection_string = os.getenv(
            storage_config.get("postgresql", {}).get("connection_string_env", "DATABASE_URL")
        )
        if not connection_string:
            raise ValueError("DATABASE_URL not set. Cannot initialize vector store.")

        vector_store = await load_vector_store_adapter(
            backend="postgresql",
            connection_string=connection_string,
            pool_size=storage_config.get("postgresql", {}).get("pool_size", 20),
            dimensions=storage_config.get("postgresql", {}).get("dimensions", 2048),
        )

        self.vector_store = vector_store

        # Graph storage (optional — initialized before ToolConfig so it can be passed in)
        graph_storage = None
        if hasattr(vector_store, "pool") and vector_store.pool:
            try:
                from ..graph import GraphEmbedder, GraphStorageAdapter
            except ImportError as e:
                logger.warning(f"Graph module not importable: {e}")
                GraphStorageAdapter = None
                GraphEmbedder = None

            if GraphStorageAdapter is not None:
                try:
                    graph_embedder = GraphEmbedder()
                    graph_storage = GraphStorageAdapter(
                        pool=vector_store.pool, embedder=graph_embedder
                    )
                    logger.info("Graph storage initialized with embedder (shares vector_store pool)")
                except Exception as e:
                    logger.error(f"Graph storage initialization failed: {e}", exc_info=True)

        # Adaptive retrieval config
        from ..retrieval.adaptive_k import AdaptiveKConfig

        agent_tools_config = self.config.get("agent_tools", {})
        adaptive_raw = agent_tools_config.get("adaptive_retrieval", {})
        adaptive_config = AdaptiveKConfig(
            enabled=adaptive_raw.get("enabled", True),
            method=adaptive_raw.get("method", "otsu"),
            fetch_k=adaptive_raw.get("fetch_k", 20),
            min_k=adaptive_raw.get("min_k", 1),
            max_k=adaptive_raw.get("max_k", 10),
            score_gap_threshold=adaptive_raw.get("score_gap_threshold", 0.05),
            min_samples_for_adaptive=adaptive_raw.get("min_samples_for_adaptive", 3),
        )

        # Web search config
        web_search_config = agent_tools_config.get("web_search", {})

        # Tool config
        tool_config = ToolConfig(
            graph_storage=graph_storage,
            default_k=agent_tools_config.get("default_k", 6),
            adaptive_retrieval=adaptive_config,
            max_document_compare=agent_tools_config.get("max_document_compare", 3),
            compliance_threshold=agent_tools_config.get("compliance_threshold", 0.7),
            web_search_enabled=web_search_config.get("enabled", True),
            web_search_model=web_search_config.get("model", "gemini-2.0-flash"),
        )

        # VL components
        vl_retriever = None
        page_store = None

        try:
            from ..vl import create_vl_components

            vl_config = self.config.get("vl", {})
            vl_retriever, page_store = create_vl_components(
                vl_config, vector_store, adaptive_config=adaptive_config
            )
        except Exception as e:
            logger.error(f"Failed to init VL components: {e}", exc_info=True)
            raise AgentInitializationError(
                f"VL initialization failed: {e}.",
                details={"phase": "vl_initialization"},
                cause=e,
            ) from e

        # Initialize registry
        registry = get_registry()
        registry.initialize_tools(
            vector_store=vector_store,
            llm_provider=self.llm_provider,
            config=tool_config,
            vl_retriever=vl_retriever,
            page_store=page_store,
        )

        total_tools = len(registry)
        logger.info(f"Tool registry initialized: {total_tools} tools")

    def _load_system_prompt(self) -> None:
        """Load system prompt."""
        prompt_file = self.single_agent_config.get("prompt_file", "prompts/agents/unified.txt")
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            # Try relative to project root
            prompt_path = Path(__file__).parent.parent.parent / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {prompt_file}")

        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        logger.info(f"Loaded system prompt from {prompt_path} ({len(self.system_prompt)} chars)")

    def _create_provider(self, model: str):
        """Create LLM provider for the given model."""
        from ..agent.providers.factory import create_provider

        return create_provider(model=model)

    def get_tool_health(self) -> Dict[str, Any]:
        """Get health status of all registered tools."""
        try:
            from ..agent.tools import get_registry

            registry = get_registry()
            available = list(registry._tools.keys())
            unavailable = registry.get_unavailable_tools()

            critical_tools = ["search", "expand_context", "get_document_info"]
            missing_critical = [t for t in critical_tools if t not in available]

            if missing_critical:
                summary = f"Critical tools unavailable: {', '.join(missing_critical)}"
                healthy = False
            else:
                summary = f"All {len(available)} tools healthy"
                healthy = True

            # Track degraded components
            degraded = []
            if not self.llm_provider:
                degraded.append("llm_provider (compliance_check will return UNCLEAR)")

            # Check graph storage via tool config
            from ..agent.tools import get_registry
            reg = get_registry()
            sample_tool = next(iter(reg._tools.values()), None)
            if sample_tool and not getattr(getattr(sample_tool, "config", None), "graph_storage", None):
                degraded.append("graph_storage (graph_search/graph_context/compliance_check unavailable)")

            return {
                "healthy": healthy and not missing_critical,
                "available_tools": available,
                "unavailable_tools": unavailable,
                "degraded_tools": degraded,
                "critical_missing": missing_critical,
                "summary": summary,
                "total_available": len(available),
                "total_unavailable": len(unavailable),
            }
        except Exception as e:
            logger.error(f"Tool health check failed: {e}", exc_info=True)
            return {
                "healthy": False,
                "available_tools": [],
                "unavailable_tools": {"unknown": str(e)},
                "degraded_tools": [],
                "critical_missing": ["unknown"],
                "summary": f"Tool health check failed: {e}",
                "total_available": 0,
                "total_unavailable": 1,
            }

    async def run_query(
        self,
        query: str,
        model: str = "",
        stream_progress: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        attachment_blocks: Optional[List[Dict[str, Any]]] = None,
        disabled_tools: Optional[Set[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run query through single agent with autonomous tool loop.

        Args:
            query: User query
            model: LLM model to use (overrides config default)
            stream_progress: If True, yields intermediate progress events
            conversation_history: Previous messages for context
            attachment_blocks: Multimodal content blocks from user attachments

        Yields:
            Dict events. Final event has type='final'.
        """
        if not self._initialized:
            yield {
                "type": "final",
                "success": False,
                "final_answer": "SingleAgentRunner not initialized. Call initialize() first.",
                "errors": ["Runner not initialized"],
            }
            return

        from ..cost_tracker import get_global_tracker, reset_global_tracker

        reset_global_tracker()
        cost_tracker = get_global_tracker()

        # Resolve model
        if not model:
            model = self.single_agent_config.get("model", "claude-haiku-4-5-20251001")

        max_tokens = self.single_agent_config.get("max_tokens", 4096)
        temperature = self.single_agent_config.get("temperature", 0.3)
        max_iterations = self.single_agent_config.get("max_iterations", 10)
        enable_prompt_caching = self.single_agent_config.get("enable_prompt_caching", True)

        try:
            provider = self._create_provider(model)
            provider_name = provider.get_provider_name()
            model_name = provider.get_model_name()
            # local_llm: ensure at least 32768 tokens (thinking models emit large <think> blocks).
            # Anthropic: if configured max_tokens exceeds 16384, clamp to 4096
            # (Anthropic SDK rejects high non-streaming values).
            if provider_name == "local_llm":
                max_tokens = max(max_tokens, 32768)
            elif provider_name == "anthropic" and max_tokens > 16384:
                max_tokens = 4096
        except Exception as e:
            yield {
                "type": "final",
                "success": False,
                "final_answer": f"Failed to create LLM provider for {model}: {e}",
                "errors": [str(e)],
            }
            return

        # Context budget monitor (uses response.usage.input_tokens for thresholds)
        context_window = get_context_window(model_name)
        monitor = ContextBudgetMonitor(context_window)

        # Get tool schemas (exclude any per-request disabled tools)
        _disabled = disabled_tools or set()
        tool_schemas = [
            schema
            for tool_name in self.tool_adapter.get_available_tools()
            if tool_name not in _disabled
            and (schema := self.tool_adapter.get_tool_schema(tool_name))
        ]

        logger.info(f"Running query with model={model_name}, tools={len(tool_schemas)}")

        # Build user message content (text or multimodal with attachments)
        if attachment_blocks:
            user_content = attachment_blocks + [{"type": "text", "text": query}]
        else:
            user_content = query

        # Build messages
        messages = []
        if conversation_history:
            # Add last 6 messages (3 Q&A pairs) for context
            for msg in conversation_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Avoid duplicate user message: if history already ends with
            # the current query (frontend may include it), don't add again.
            if (
                messages
                and messages[-1]["role"] == "user"
                and isinstance(messages[-1]["content"], str)
                and messages[-1]["content"].strip() == query.strip()
            ):
                pass  # query already in history
            else:
                messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_content})

        # Prepare system prompt (with caching only for providers that support it)
        system = self.system_prompt
        if enable_prompt_caching and provider.supports_feature("prompt_caching"):
            system = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Extract attachment images for tool access (image search)
        attachment_images = self._extract_attachment_images(attachment_blocks)
        self.tool_adapter.registry.set_request_context(
            attachment_images=attachment_images,
        )

        tool_call_history = []
        total_tool_cost = 0.0
        final_text = ""
        iteration = 0
        text_was_streamed = False  # Track whether final text was actually streamed live
        seen_page_ids: set = set()  # Track seen page_ids to avoid duplicate base64 images (~1600 tokens/page)

        try:
            for iteration in range(max_iterations):
                logger.info(f"Iteration {iteration + 1}/{max_iterations}")

                # Force tool use on first iteration for models that don't
                # voluntarily call tools (e.g., Qwen3-VL on DeepInfra)
                extra_kwargs = {}
                if iteration == 0 and tool_schemas:
                    if provider_name in ("deepinfra", "local_llm"):
                        extra_kwargs["tool_choice"] = "required"
                    elif provider_name == "anthropic":
                        extra_kwargs["tool_choice"] = {"type": "any"}

                # LLM call — streaming for local_llm, non-streaming otherwise
                llm_start = time.time()
                try:
                    if provider_name == "local_llm":
                        # Streaming path: yields thinking/text events live
                        response = None
                        used_fallback = False
                        async for event in self._stream_llm_iteration(
                            provider, messages, tool_schemas, system,
                            max_tokens, temperature, **extra_kwargs,
                        ):
                            if event["type"] == "llm_response":
                                response = event["response"]
                            elif event["type"] == "streaming_fallback":
                                used_fallback = True
                            elif stream_progress:
                                yield event  # Forward thinking/text events to caller
                        if response is None:
                            raise RuntimeError("Streaming produced no response")
                        if not used_fallback:
                            text_was_streamed = True
                    else:
                        response = provider.create_message(
                            messages=messages,
                            tools=tool_schemas,
                            system=system,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **extra_kwargs,
                        )
                    llm_time_ms = (time.time() - llm_start) * 1000
                except Exception as e:
                    logger.error(f"LLM call failed: {e}", exc_info=True)
                    yield {
                        "type": "final",
                        "success": False,
                        "final_answer": f"LLM call failed: {e}",
                        "errors": [str(e)],
                    }
                    return

                # Update context budget from actual token usage
                monitor.update_from_response(response)

                # Track cost
                if hasattr(response, "usage") and response.usage:
                    cost_tracker.track_llm(
                        provider=provider_name,
                        model=model_name,
                        input_tokens=response.usage.get("input_tokens", 0),
                        output_tokens=response.usage.get("output_tokens", 0),
                        operation="single_agent",
                        cache_creation_tokens=response.usage.get("cache_creation_tokens", 0),
                        cache_read_tokens=response.usage.get("cache_read_tokens", 0),
                        response_time_ms=llm_time_ms,
                    )

                # Check if LLM wants to use tools
                if hasattr(response, "stop_reason") and response.stop_reason == "tool_use":
                    tool_uses = [
                        b
                        for b in response.content
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    ]

                    tool_results = []
                    for tool_use in tool_uses:
                        tool_name = tool_use.get("name")
                        tool_input = tool_use.get("input", {})
                        tool_use_id = tool_use.get("id")

                        logger.info(f"Tool call: {tool_name}")

                        if stream_progress:
                            yield {
                                "type": "tool_call",
                                "tool": tool_name,
                                "status": "running",
                            }

                        # Execute tool
                        tool_start = time.time()
                        result = await self.tool_adapter.execute(
                            tool_name=tool_name,
                            inputs=tool_input,
                            agent_name="single_agent",
                        )
                        tool_duration_ms = (time.time() - tool_start) * 1000

                        tool_cost = (
                            result.get("metadata", {}).get("api_cost_usd", 0.0)
                            if isinstance(result, dict)
                            else 0.0
                        )
                        total_tool_cost += tool_cost

                        if stream_progress:
                            yield {
                                "type": "tool_call",
                                "tool": tool_name,
                                "status": "completed" if result.get("success") else "failed",
                            }

                        # Handle VL multimodal page images
                        page_images = (
                            result.get("metadata", {}).get("page_images")
                            if result.get("success")
                            else None
                        )

                        if page_images:
                            content_blocks = []
                            dedup_skipped = 0
                            for page in page_images:
                                pid = page.get("page_id", "unknown")
                                if pid in seen_page_ids:
                                    # Already sent this page image — send text reference only
                                    dedup_skipped += 1
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": (
                                                f"[Already shown: {pid} | "
                                                f"Page {page.get('page_number', '?')} from "
                                                f"{page.get('document_id', 'unknown')} | "
                                                f"score: {page.get('score', 0):.3f}]"
                                            ),
                                        }
                                    )
                                    continue
                                seen_page_ids.add(pid)
                                content_blocks.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": page.get("media_type", "image/png"),
                                            "data": page["base64_data"],
                                        },
                                    }
                                )
                                content_blocks.append(
                                    {
                                        "type": "text",
                                        "text": (
                                            f"[{pid} | "
                                            f"Page {page.get('page_number', '?')} from "
                                            f"{page.get('document_id', 'unknown')} | "
                                            f"score: {page.get('score', 0):.3f}]"
                                        ),
                                    }
                                )
                            if dedup_skipped:
                                logger.info(
                                    "Image dedup: skipped %d duplicate pages for tool %s",
                                    dedup_skipped,
                                    tool_name,
                                )
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": content_blocks,
                                }
                            )
                        else:
                            raw_content = (
                                str(result.get("data", ""))
                                if result.get("success")
                                else f"Error: {result.get('error', 'Unknown error')}"
                            )
                            # Truncate long results
                            if len(raw_content) > 1500:
                                head = raw_content[:900]
                                tail = raw_content[-500:]
                                raw_content = f"{head}\n\n[...truncated...]\n\n{tail}"

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": raw_content,
                                    "is_error": not result.get("success", False),
                                }
                            )

                        tool_call_history.append(
                            {
                                "tool": tool_name,
                                "input": tool_input,
                                "success": result.get("success", False),
                                "duration_ms": tool_duration_ms,
                                "api_cost_usd": tool_cost,
                            }
                        )

                    # Add to conversation
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                    # 3-layer progressive context compaction (mutually exclusive)
                    action = monitor.recommended_action()
                    if action == CompactionLayer.EMERGENCY:
                        messages = emergency_truncate(messages)
                    elif action == CompactionLayer.COMPACT:
                        pre_len = len(messages)
                        messages = compact_with_summary(messages, provider)
                        if len(messages) == pre_len:
                            # Compaction failed — escalate to Layer 3 to prevent retry loop
                            logger.warning("Layer 2 compaction unchanged, escalating to Layer 3")
                            messages = emergency_truncate(messages)
                    elif action == CompactionLayer.PRUNE:
                        messages = prune_tool_outputs(messages)
                    elif action != CompactionLayer.NONE:
                        logger.debug("Unrecognized compaction action: %s", action)

                    # Early stop: 2+ consecutive empty searches
                    if self._should_stop_early(tool_call_history):
                        logger.info("Early stop: consecutive empty searches")
                        final_text = await self._force_final_answer(
                            provider, messages, system, max_tokens, temperature
                        )
                        break

                else:
                    # LLM provided final answer
                    final_text = (
                        response.text if hasattr(response, "text") else str(response.content)
                    )
                    break
            else:
                # Max iterations reached
                logger.warning(f"Max iterations ({max_iterations}) reached")
                final_text = await self._force_final_answer(
                    provider, messages, system, max_tokens, temperature
                )

            # Build final result
            total_cost_usd = cost_tracker.get_total_cost()

            tools_used = [tc["tool"] for tc in tool_call_history]

            yield {
                "type": "final",
                "success": True,
                "final_answer": final_text,
                "total_cost_cents": total_cost_usd * 100.0,
                "tools_used": tools_used,
                "tool_call_count": len(tool_call_history),
                "iterations": iteration + 1,
                "model": model_name,
                "errors": [],
                "text_already_streamed": text_was_streamed,
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            yield {
                "type": "final",
                "success": False,
                "final_answer": f"Query processing failed: {type(e).__name__}: {e}",
                "errors": [str(e)],
            }
        finally:
            # Clear per-request context (attachment images)
            self.tool_adapter.registry.clear_request_context()

    @staticmethod
    def _extract_attachment_images(
        attachment_blocks: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Extract image entries from attachment blocks for tool access.

        Builds a flat list of images with metadata (filename, page number)
        so the search tool can reference them by index.

        Returns:
            List of dicts with keys: index, base64_data, filename, page, media_type
        """
        if not attachment_blocks:
            return []

        images: List[Dict[str, Any]] = []
        pending_image: Optional[Dict[str, Any]] = None

        for block in attachment_blocks:
            if block.get("type") == "image":
                source = block.get("source", {})
                pending_image = {
                    "base64_data": source.get("data", ""),
                    "media_type": source.get("media_type", "image/png"),
                    "filename": None,
                    "page": None,
                }
            elif block.get("type") == "text" and pending_image is not None:
                # Parse label: "[Attached PDF: doc.pdf, page 1]" or "[Attached image: photo.jpg]"
                text = block.get("text", "")
                pdf_match = re.match(
                    r"\[Attached PDF: (.+?), page (\d+)\]", text
                )
                img_match = re.match(r"\[Attached image: (.+?)\]", text)

                if pdf_match:
                    pending_image["filename"] = pdf_match.group(1)
                    pending_image["page"] = int(pdf_match.group(2))
                elif img_match:
                    pending_image["filename"] = img_match.group(1)

                pending_image["index"] = len(images)
                images.append(pending_image)
                pending_image = None

        # Handle trailing image without a text label
        if pending_image is not None:
            pending_image["index"] = len(images)
            images.append(pending_image)

        if images:
            logger.info(
                "Extracted %d attachment images for tool context", len(images)
            )

        return images

    async def _stream_llm_iteration(
        self,
        provider,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        system: Any,
        max_tokens: int,
        temperature: float,
        **extra_kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a single LLM iteration for local_llm provider.

        Parses <think> tags live and yields events:
          - thinking_delta: Thinking content (streamed immediately)
          - thinking_done: Thinking phase finished
          - text_delta: Text content chunks (whitespace-only buffered until substantive text arrives)
          - streaming_fallback: Streaming failed, fell back to non-streaming
          - llm_response: Final ProviderResponse

        Falls back to non-streaming create_message() on network/timeout errors.
        """
        from ..agent.providers.think_parser import (
            ChunkType,
            ThinkTagStreamParser,
        )
        from ..agent.providers.base import ProviderResponse
        from ..agent.providers.openai_compat import STOP_REASON_MAP

        # vLLM Qwen3 chat template strips opening <think> — model starts in thinking mode
        parser = ThinkTagStreamParser(start_thinking=True)
        text_buffer = ""
        text_delta_pending = ""  # Buffer leading whitespace after </think> (avoid premature text_delta)
        tool_call_deltas: Dict[int, Dict[str, Any]] = {}  # index → {id, name, args_str}
        finish_reason = None
        usage_data = None
        had_thinking = False
        thinking_done_sent = False

        try:
            # Call sync stream_message via executor to avoid blocking async loop
            loop = asyncio.get_running_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: provider.stream_message(
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra_kwargs,
                ),
            )

            # Consume sync iterator in executor, chunk by chunk
            def _next_chunk(it):
                try:
                    return next(it)
                except StopIteration:
                    return None

            while True:
                chunk = await loop.run_in_executor(None, _next_chunk, stream)
                if chunk is None:
                    break

                # Extract usage from final chunk (stream_options.include_usage)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_data = {
                        "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                        "cache_read_tokens": 0,
                        "cache_creation_tokens": 0,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                chunk_finish = chunk.choices[0].finish_reason

                if chunk_finish:
                    finish_reason = chunk_finish

                # Accumulate tool call deltas
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_deltas:
                            tool_call_deltas[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "args_str": "",
                            }
                        if tc_delta.id:
                            tool_call_deltas[idx]["id"] = tc_delta.id
                        if hasattr(tc_delta, "function") and tc_delta.function:
                            if tc_delta.function.name:
                                tool_call_deltas[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_call_deltas[idx]["args_str"] += tc_delta.function.arguments

                # Parse text content through think-tag parser
                if hasattr(delta, "content") and delta.content:
                    parsed_chunks = parser.feed(delta.content)
                    for pc in parsed_chunks:
                        if pc.type == ChunkType.THINKING:
                            had_thinking = True
                            yield {"type": "thinking_delta", "content": pc.content}
                        else:
                            # Accumulate for ProviderResponse always
                            text_buffer += pc.content
                            if not thinking_done_sent and had_thinking:
                                thinking_done_sent = True
                                yield {"type": "thinking_done"}
                            # Buffer leading whitespace after </think> to avoid
                            # premature text_delta that clears frontend progress indicator
                            # before tool calls arrive.
                            text_delta_pending += pc.content
                            if text_delta_pending.strip():
                                # Substantive content — flush entire buffer
                                yield {"type": "text_delta", "content": text_delta_pending}
                                text_delta_pending = ""

            # Flush parser (handles truncated tags)
            for pc in parser.flush():
                if pc.type == ChunkType.THINKING:
                    had_thinking = True
                    yield {"type": "thinking_delta", "content": pc.content}
                else:
                    text_buffer += pc.content
                    if not thinking_done_sent and had_thinking:
                        thinking_done_sent = True
                        yield {"type": "thinking_done"}
                    text_delta_pending += pc.content
                    if text_delta_pending.strip():
                        yield {"type": "text_delta", "content": text_delta_pending}
                        text_delta_pending = ""

            # Flush any remaining pending text (whitespace-only at end of iteration
            # is discarded — it's just formatting between </think> and tool calls)
            # Only emit if there's substantive content we haven't flushed yet.
            # Note: text_buffer already has this content for ProviderResponse.

            # Signal thinking done if no text followed (tool-call-only iteration)
            if had_thinking and not thinking_done_sent:
                yield {"type": "thinking_done"}

            # Build ProviderResponse from accumulated data
            content_blocks = []
            clean_text = text_buffer.strip()
            if clean_text:
                content_blocks.append({"type": "text", "text": clean_text})

            # Convert tool call deltas to content blocks
            for idx in sorted(tool_call_deltas.keys()):
                tc = tool_call_deltas[idx]
                try:
                    parsed_args = json.loads(tc["args_str"]) if tc["args_str"] else {}
                except json.JSONDecodeError:
                    logger.error(
                        "Failed to parse tool args for %s (skipping tool call): %s",
                        tc["name"], tc["args_str"][:200],
                    )
                    content_blocks.append({
                        "type": "text",
                        "text": f"[Tool call '{tc['name']}' had unparseable arguments and was skipped]",
                    })
                    continue
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": parsed_args,
                })

            stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

            if not usage_data:
                usage_data = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                }

            response = ProviderResponse(
                content=content_blocks if content_blocks else [{"type": "text", "text": ""}],
                stop_reason=stop_reason,
                usage=usage_data,
                model=provider.get_model_name(),
            )

            yield {"type": "llm_response", "response": response}

        except asyncio.CancelledError:
            logger.info("Streaming cancelled by user")
            raise
        except Exception as e:
            logger.error("Streaming failed, falling back to non-streaming: %s", e, exc_info=True)
            # Signal that we fell back so text_already_streamed is set correctly
            yield {"type": "streaming_fallback"}
            response = provider.create_message(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_kwargs,
            )
            yield {"type": "llm_response", "response": response}

    def _should_stop_early(self, tool_call_history: List[Dict]) -> bool:
        """Check if we should stop (2+ consecutive failed searches)."""
        if len(tool_call_history) < 2:
            return False

        search_tools = {"search"}
        failed = 0
        for call in reversed(tool_call_history[-3:]):
            if call.get("tool") in search_tools and not call.get("success", True):
                failed += 1
            else:
                break
        return failed >= 2

    async def _force_final_answer(self, provider, messages, system, max_tokens, temperature) -> str:
        """Force LLM to produce final text answer (no tools)."""
        final_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Provide your FINAL answer now based on the information gathered. "
                    "Do NOT call any more tools. Use \\cite{id} citations for all facts, "
                    "where id is the exact page_id from search results "
                    "(e.g., \\cite{BZ_VR1_p003})."
                ),
            }
        ]
        try:
            final_response = provider.create_message(
                messages=final_messages,
                tools=[],
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (
                final_response.text
                if hasattr(final_response, "text")
                else str(final_response.content)
            )
        except Exception as e:
            logger.error(f"Final answer synthesis failed: {e}", exc_info=True)
            return (
                "System could not generate a final answer due to an internal error. "
                "Search results were retrieved but could not be synthesized. "
                "Please try again. / "
                "Systém nemohl vygenerovat odpověď kvůli interní chybě. "
                "Výsledky vyhledávání byly nalezeny, ale nebyly zpracovány. "
                "Zkuste to prosím znovu."
            )

    async def shutdown_async(self) -> None:
        """Async shutdown — close connection pools."""
        logger.info("Shutting down single-agent system...")
        if self.vector_store and hasattr(self.vector_store, "close"):
            try:
                await self.vector_store.close()
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")
        logger.info("Single-agent system shut down")
