"""
Single Agent Runner — autonomous agent with unified prompt and all RAG tools.

Replaces the multi-agent orchestrator + 8 specialist agents with a single
LLM that decides which tools to call and when to stop.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

from ..exceptions import AgentInitializationError

logger = logging.getLogger(__name__)


class SingleAgentRunner:
    """
    Single autonomous agent with access to all RAG tools.

    Replaces MultiAgentRunner by:
    - Using one LLM call loop instead of orchestrator + specialist routing
    - Exposing ALL tools (search, expand_context, etc.) to one agent
    - Loading system prompt: unified.txt (VL) or unified_ocr.txt (OCR)
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
            from ..multi_agent.tools.adapter import ToolAdapter

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
            from ..multi_agent.observability import setup_langsmith

            self.langsmith = setup_langsmith(langsmith_config)
            logger.info("LangSmith tracing enabled")
        except Exception as e:
            logger.warning(f"LangSmith setup failed (non-fatal): {e}")
            self.langsmith = None

    async def _initialize_tools(self) -> None:
        """Initialize tool registry with all RAG components (vector store, embedder, VL components)."""
        from ..agent.tools import get_registry
        from ..storage import load_vector_store_adapter
        from ..embedding_generator import EmbeddingGenerator
        from ..agent.config import ToolConfig
        from ..agent.providers.factory import create_provider
        from ..config import EmbeddingConfig

        storage_config = self.config.get("storage", {})
        backend = storage_config.get("backend", "postgresql")

        # LLM provider for tools (OCR mode: HyDE/synthesis) — use cheap model
        tool_model = self.config.get("models", {}).get("llm_model", "gpt-4o-mini")
        try:
            self.llm_provider = create_provider(model=tool_model)
            logger.info(f"Tool LLM provider: {tool_model}")
        except Exception as e:
            logger.error(
                "Tool LLM provider failed for %s: %s. HyDE search disabled — "
                "expect degraded retrieval quality.",
                tool_model,
                e,
                exc_info=True,
            )
            self.llm_provider = None

        architecture = self.config.get("architecture", "vl")

        # Vector store
        if backend == "postgresql":
            connection_string = os.getenv(
                storage_config.get("postgresql", {}).get("connection_string_env", "DATABASE_URL")
            )
            if not connection_string:
                raise ValueError("DATABASE_URL not set. Cannot initialize vector store.")

            vector_store = await load_vector_store_adapter(
                backend="postgresql",
                connection_string=connection_string,
                pool_size=storage_config.get("postgresql", {}).get("pool_size", 20),
                dimensions=storage_config.get("postgresql", {}).get("dimensions", 4096),
                architecture=architecture,
            )
        else:
            vector_store_path = Path(self.config.get("vector_store_path", "vector_db"))
            if not vector_store_path.exists():
                raise ValueError(f"Vector store not found at {vector_store_path}")
            vector_store = await load_vector_store_adapter(
                backend="faiss", path=str(vector_store_path)
            )

        self.vector_store = vector_store

        # Embedder
        model_name = self.config.get("models", {}).get("embedding_model", "bge-m3")
        model_provider = self.config.get("models", {}).get("embedding_provider", "huggingface")
        embedding_config = EmbeddingConfig(
            provider=model_provider,
            model=model_name,
            batch_size=64,
            normalize=True,
            enable_multi_layer=True,
            cache_enabled=True,
            cache_max_size=1000,
        )
        embedder = EmbeddingGenerator(embedding_config)

        # Graph storage (optional — initialized before ToolConfig so it can be passed in)
        graph_storage = None
        if architecture == "vl" and hasattr(vector_store, "pool") and vector_store.pool:
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

        # Tool config
        agent_tools_config = self.config.get("agent_tools", {})
        tool_config = ToolConfig(
            graph_storage=graph_storage,
            default_k=agent_tools_config.get("default_k", 6),
            enable_reranking=agent_tools_config.get("enable_reranking", False),
            reranker_candidates=agent_tools_config.get("reranker_candidates", 50),
            reranker_model=agent_tools_config.get("reranker_model", "bge-reranker-large"),
            max_document_compare=agent_tools_config.get("max_document_compare", 3),
            compliance_threshold=agent_tools_config.get("compliance_threshold", 0.7),
            context_window=agent_tools_config.get("context_window", 2),
            lazy_load_reranker=agent_tools_config.get("lazy_load_reranker", False),
            cache_embeddings=agent_tools_config.get("cache_embeddings", True),
            hyde_num_hypotheses=agent_tools_config.get("hyde_num_hypotheses", 3),
            query_expansion_provider=agent_tools_config.get("query_expansion_provider", "openai"),
            query_expansion_model=agent_tools_config.get("query_expansion_model", "gpt-4o-mini"),
        )

        # VL components (optional)
        vl_retriever = None
        page_store = None

        if architecture == "vl":
            try:
                from ..vl import create_vl_components

                vl_config = self.config.get("vl", {})
                vl_retriever, page_store = create_vl_components(vl_config, vector_store)
            except Exception as e:
                logger.error(f"Failed to init VL components: {e}", exc_info=True)
                raise AgentInitializationError(
                    f"VL architecture was explicitly configured but initialization failed: {e}. "
                    f"Fix VL config or switch to architecture='ocr' in config.json.",
                    details={"phase": "vl_initialization", "architecture": "vl"},
                    cause=e,
                ) from e

        # Initialize registry
        registry = get_registry()
        registry.initialize_tools(
            vector_store=vector_store,
            embedder=embedder,
            reranker=None,
            context_assembler=None,
            llm_provider=self.llm_provider,
            config=tool_config,
            vl_retriever=vl_retriever,
            page_store=page_store,
        )

        total_tools = len(registry)
        logger.info(f"Tool registry initialized: {total_tools} tools")

    def _load_system_prompt(self) -> None:
        """Load system prompt based on architecture mode (VL or OCR)."""
        architecture = self.config.get("architecture", "vl")
        default_prompt = (
            "prompts/agents/unified.txt"
            if architecture == "vl"
            else "prompts/agents/unified_ocr.txt"
        )
        prompt_file = self.single_agent_config.get("prompt_file", default_prompt)
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            # Try relative to project root
            prompt_path = Path(__file__).parent.parent.parent / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt not found: {prompt_file} (architecture={architecture})"
            )

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

            return {
                "healthy": healthy,
                "available_tools": available,
                "unavailable_tools": unavailable,
                "degraded_tools": [],
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run query through single agent with autonomous tool loop.

        Args:
            query: User query
            model: LLM model to use (overrides config default)
            stream_progress: If True, yields intermediate progress events
            conversation_history: Previous messages for context

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
        except Exception as e:
            yield {
                "type": "final",
                "success": False,
                "final_answer": f"Failed to create LLM provider for {model}: {e}",
                "errors": [str(e)],
            }
            return

        # Get all tool schemas
        tool_schemas = [
            schema
            for tool_name in self.tool_adapter.get_available_tools()
            if (schema := self.tool_adapter.get_tool_schema(tool_name))
        ]

        logger.info(f"Running query with model={model_name}, tools={len(tool_schemas)}")

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
                and messages[-1]["content"].strip() == query.strip()
            ):
                pass  # query already in history
            else:
                messages.append({"role": "user", "content": query})
        else:
            messages.append({"role": "user", "content": query})

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

        tool_call_history = []
        total_tool_cost = 0.0
        final_text = ""
        iteration = 0

        try:
            for iteration in range(max_iterations):
                logger.info(f"Iteration {iteration + 1}/{max_iterations}")

                # Force tool use on first iteration for models that don't
                # voluntarily call tools (e.g., Qwen3-VL on DeepInfra)
                extra_kwargs = {}
                if iteration == 0 and tool_schemas:
                    if provider_name == "deepinfra":
                        extra_kwargs["tool_choice"] = "required"
                    elif provider_name == "anthropic":
                        extra_kwargs["tool_choice"] = {"type": "any"}

                # LLM call
                llm_start = time.time()
                try:
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
                            for page in page_images:
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
                                            f"[USE THIS EXACT page_id IN \\cite{{}}: "
                                            f"{page.get('page_id', 'unknown')} | "
                                            f"Page {page.get('page_number', '?')} from "
                                            f"{page.get('document_id', 'unknown')} | "
                                            f"score: {page.get('score', 0):.3f}]"
                                        ),
                                    }
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

                    # Truncate history to prevent overflow
                    max_history = 7  # 1 original + 6 recent
                    if len(messages) > max_history:
                        messages = [messages[0]] + messages[-6:]

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
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            yield {
                "type": "final",
                "success": False,
                "final_answer": f"Query processing failed: {type(e).__name__}: {e}",
                "errors": [str(e)],
            }

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
                    "where id is the exact chunk_id or page_id from search results "
                    "(e.g., \\cite{BZ_VR1_p003} for VL page results, \\cite{BZ_VR1_L3_5} for text chunks)."
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
