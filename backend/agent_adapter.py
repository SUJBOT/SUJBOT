"""
Agent Adapter - Wraps SingleAgentRunner for web frontend.

This adapter:
1. Initializes SingleAgentRunner
2. Handles SSE event formatting (tool_call, thinking_delta, text_delta, etc.)
3. Tracks cost per message
4. Provides clean interface for FastAPI
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional

from src.single_agent.runner import SingleAgentRunner
from src.single_agent.routing_runner import RoutingAgentRunner
from src.agent.config import AgentConfig
from src.cost_tracker import get_global_tracker, reset_global_tracker
from src.utils.security import sanitize_error
from backend.constants import (
    get_default_variant,
    get_variant_model,
    get_variant_display_name,
    is_valid_variant,
)

logger = logging.getLogger(__name__)


class AgentAdapter:
    """
    Adapter wrapping SingleAgentRunner for web frontend.

    Responsibilities:
    - Initialize SingleAgentRunner
    - Convert runner events to SSE format
    - Track costs per request
    - Resolve variant → model for each request
    """

    def __init__(self, vector_store_path: Optional[Path] = None, model: Optional[str] = None):
        """
        Initialize agent adapter with single-agent system.

        Args:
            vector_store_path: Path to vector store (default: ../vector_db from backend/)
            model: Model to use (default: from config)
        """
        # Load config from environment
        config_overrides = {}

        # Set default vector_store_path relative to project root (parent of backend/)
        project_root = Path(__file__).parent.parent
        if vector_store_path:
            config_overrides["vector_store_path"] = vector_store_path
        else:
            config_overrides["vector_store_path"] = project_root / "vector_db"

        if model:
            config_overrides["model"] = model

        # Load full configuration
        self.config = AgentConfig.from_env(**config_overrides)

        # Validate config
        try:
            self.config.validate()
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise

        # Load full config.json for runner
        config_path = project_root / "config.json"
        full_config = {}

        if config_path.exists():
            try:
                with open(config_path) as f:
                    full_config = json.load(f)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"config.json at {config_path} contains invalid JSON: {e}. "
                    f"Fix the file before starting."
                ) from e
            except Exception as e:
                logger.error(f"Failed to read config.json: {e}", exc_info=True)
                raise RuntimeError(f"Cannot read config.json at {config_path}: {e}") from e
        else:
            logger.warning(f"config.json not found at {config_path}, using defaults")

        # Build runner configuration
        runner_config = {
            "api_keys": {
                "anthropic_api_key": self.config.anthropic_api_key,
                "openai_api_key": self.config.openai_api_key,
                "google_api_key": self.config.google_api_key,
                "deepinfra_api_key": os.getenv("DEEPINFRA_API_KEY"),
            },
            "vector_store_path": str(self.config.vector_store_path),
            "models": full_config.get("models", {}),
            "storage": full_config.get("storage", {}),
            "agent_tools": full_config.get("agent_tools", {}),
            "single_agent": full_config.get("single_agent", {}),
            "langsmith": full_config.get("langsmith", {}),
            "vl": full_config.get("vl", {}),
            "routing": full_config.get("routing", {}),
        }

        # Initialize single-agent runner
        logger.info("Initializing single-agent system...")
        self.runner = SingleAgentRunner(runner_config)

        # Initialize routing runner (8B router → 30B worker) if enabled
        routing_config = full_config.get("routing", {})
        self.routing_runner = None
        if routing_config.get("enabled", False):
            try:
                self.routing_runner = RoutingAgentRunner(runner_config, self.runner)
                logger.info(
                    "Routing runner initialized: %s → %s",
                    routing_config.get("router_model", "qwen3-vl-8b-local"),
                    routing_config.get("worker_model", "qwen3-vl-30b-local"),
                )
            except FileNotFoundError as e:
                logger.error("Routing runner init failed (prompt missing?): %s", e)
            except Exception as e:
                logger.error("Routing runner init failed: %s", e, exc_info=True)

        # Track degraded components for health endpoint
        self.degraded_components = []

        # Store current model
        self.current_model = model or full_config.get("single_agent", {}).get(
            "model", "claude-haiku-4-5-20251001"
        )

        logger.info(
            f"AgentAdapter initialized: "
            f"model={self.current_model}, vector_store={self.config.vector_store_path}"
        )

    async def initialize(self) -> bool:
        """
        Initialize the single-agent runner.

        Returns:
            True if successful
        """
        try:
            success = await self.runner.initialize()
            if not success:
                logger.error("Single-agent system initialization failed")
                self.degraded_components.append(
                    {
                        "component": "single_agent_system",
                        "error": "Initialization failed",
                        "severity": "critical",
                        "user_message": "Agent system failed to start",
                    }
                )
                return False

            logger.info("Single-agent system initialized successfully")
            return True

        except Exception as e:
            safe_error = sanitize_error(e)
            logger.error(f"Failed to initialize single-agent system: {safe_error}")
            self.degraded_components.append(
                {
                    "component": "single_agent_system",
                    "error": safe_error,
                    "severity": "critical",
                    "user_message": f"System initialization error: {safe_error}",
                }
            )
            return False

    def get_tool_health(self) -> Dict[str, Any]:
        """Get health status of all RAG tools."""
        return self.runner.get_tool_health()

    async def stream_response(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        attachment_blocks: Optional[List[Dict]] = None,
        web_search_enabled: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream agent response as SSE-compatible events.

        Yields SSE events:
        - tool_health: Tool status before query
        - progress: Workflow stage updates
        - tool_call: Tool execution events
        - thinking_delta: Thinking content from local LLM
        - thinking_done: Thinking phase finished
        - text_delta: Answer text chunks (live-streamed or chunked from final)
        - tool_calls_summary: Summary of tools used
        - cost_summary: Cost breakdown
        - done: Stream completed
        - error: Error occurred

        Args:
            query: User query
            conversation_id: Optional conversation ID
            user_id: User ID for loading agent variant preference
            messages: Conversation history
            attachment_blocks: Multimodal content blocks from user attachments
            web_search_enabled: Whether to enable web_search tool for this request

        Yields:
            Dict containing event type and data
        """
        # Reset cost tracker for this request
        reset_global_tracker()
        tracker = get_global_tracker()

        # Resolve variant → model
        variant = get_default_variant()
        model = get_variant_model(variant)

        if user_id:
            try:
                from backend.deps import get_auth_queries

                queries = get_auth_queries()
                variant = await queries.get_agent_variant(user_id)
                model = get_variant_model(variant)
                logger.info(f"User {user_id} variant: {variant} → model: {model}")
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                logger.warning(f"Config error loading variant for user {user_id}: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error loading variant for user {user_id}: {sanitize_error(e)}"
                )

        try:
            # Run tool health check
            tool_health = self.runner.get_tool_health()

            yield {
                "event": "tool_health",
                "data": {
                    "healthy": tool_health["healthy"],
                    "available_count": tool_health["total_available"],
                    "unavailable_count": tool_health["total_unavailable"],
                    "unavailable_tools": tool_health["unavailable_tools"],
                    "degraded_tools": tool_health["degraded_tools"],
                    "summary": tool_health["summary"],
                },
            }
            await asyncio.sleep(0)

            if not tool_health["healthy"]:
                logger.warning(f"Tool health check: {tool_health['summary']}")
            else:
                logger.info(f"Tool health check: {tool_health['summary']}")

            # Emit start event
            display_name = get_variant_display_name(variant)
            yield {
                "event": "progress",
                "data": {"message": f"Processing query ({display_name})...", "stage": "init"},
            }
            await asyncio.sleep(0)

            # Choose runner: routing runner for local variant, standard for remote
            use_routing = self.routing_runner is not None and variant == "local"
            runner = self.routing_runner if use_routing else self.runner

            logger.info(
                "Starting query execution: variant=%s, model=%s, routing=%s",
                variant, model, use_routing,
            )
            if messages:
                logger.info(f"Including {len(messages)} messages of conversation history")

            result = None
            async for event in runner.run_query(
                query,
                model=model,
                stream_progress=True,
                conversation_history=messages or [],
                attachment_blocks=attachment_blocks,
                disabled_tools={"web_search"} if not web_search_enabled else None,
            ):
                event_type = event.get("type")

                if event_type == "routing":
                    yield {
                        "event": "routing",
                        "data": {
                            "decision": event.get("decision"),
                            "complexity": event.get("complexity"),
                            "thinking_budget": event.get("thinking_budget"),
                            "tool": event.get("tool"),
                        },
                    }
                    await asyncio.sleep(0)

                elif event_type == "tool_call":
                    yield {
                        "event": "tool_call",
                        "data": {
                            "tool": event.get("tool"),
                            "status": event.get("status"),
                        },
                    }
                    await asyncio.sleep(0)

                elif event_type == "thinking_delta":
                    yield {
                        "event": "thinking_delta",
                        "data": {"content": event.get("content", "")},
                    }
                    await asyncio.sleep(0)

                elif event_type == "thinking_done":
                    yield {
                        "event": "thinking_done",
                        "data": {},
                    }
                    await asyncio.sleep(0)

                elif event_type == "text_delta":
                    # Live-streamed text from local_llm
                    yield {
                        "event": "text_delta",
                        "data": {"content": event.get("content", "")},
                    }
                    await asyncio.sleep(0)

                elif event_type == "final":
                    result = event
                    break

            if not result:
                raise RuntimeError("Agent produced no result")

            # Check success
            if not result.get("success", False):
                error_msg = result.get("final_answer", "Unknown error")
                yield {
                    "event": "error",
                    "data": {
                        "error": error_msg,
                        "type": "ExecutionError",
                        "errors": result.get("errors", []),
                    },
                }
                return

            # Stream final answer as text chunks (skip if already streamed live)
            if not result.get("text_already_streamed"):
                final_answer = result.get("final_answer") or "No answer generated"
                paragraphs = final_answer.split("\n\n")

                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        chunk = paragraph + ("\n\n" if i < len(paragraphs) - 1 else "")
                        yield {"event": "text_delta", "data": {"content": chunk}}
                        await asyncio.sleep(0.05)

            # Cost summary
            tracker = get_global_tracker()
            total_cost_usd = tracker.get_total_cost()
            agent_breakdown = tracker.get_agent_breakdown()

            agent_costs = [
                {
                    "agent": agent_name,
                    "cost": stats.get("cost", 0.0),
                    "input_tokens": stats.get("input_tokens", 0),
                    "output_tokens": stats.get("output_tokens", 0),
                    "cache_read_tokens": stats.get("cache_read_tokens", 0),
                    "cache_creation_tokens": stats.get("cache_creation_tokens", 0),
                    "call_count": stats.get("call_count", 0),
                    "response_time_ms": stats.get("response_time_ms", 0.0),
                }
                for agent_name, stats in agent_breakdown.items()
            ]
            agent_costs.sort(key=lambda x: x["cost"], reverse=True)

            yield {
                "event": "cost_summary",
                "data": {
                    "total_cost": total_cost_usd,
                    "agent_breakdown": agent_costs,
                    "total_input_tokens": tracker.total_input_tokens,
                    "total_output_tokens": tracker.total_output_tokens,
                    "cache_stats": tracker.get_cache_stats(),
                },
            }
            await asyncio.sleep(0)

            # Tool calls summary (if any tools were used)
            tools_used = result.get("tools_used", [])
            if tools_used:
                yield {
                    "event": "tool_calls_summary",
                    "data": {
                        "tool_calls": [
                            {"name": tool_name, "success": True}
                            for tool_name in tools_used
                        ],
                        "count": len(tools_used),
                    },
                }
                await asyncio.sleep(0)

            # Signal completion
            yield {
                "event": "done",
                "data": {
                    "model": result.get("model", model),
                    "variant": variant,
                    "tools_used": tools_used,
                    "tool_call_count": result.get("tool_call_count", 0),
                    "iterations": result.get("iterations", 0),
                },
            }

        except Exception as e:
            logger.error(
                f"Error during query execution: {type(e).__name__}: {e}",
                exc_info=True,
                extra={
                    "query": query[:200] if query else "N/A",
                    "conversation_id": conversation_id,
                    "variant": variant,
                    "model": model,
                    "error_phase": "single_agent_execution",
                },
            )

            yield {
                "event": "error",
                "data": {"error": sanitize_error(e)},
            }

    def switch_model(self, model: str) -> None:
        """Switch default model."""
        old_model = self.current_model
        self.current_model = model
        logger.info(f"Model switched: {old_model} → {model}")

    async def shutdown_variant_runners(self) -> None:
        """Shutdown runner (backward-compatible method name)."""
        await self.runner.shutdown_async()

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        try:
            if not self.runner:
                return {
                    "status": "error",
                    "message": "Agent runner not initialized",
                    "details": {},
                    "degraded_components": [],
                }

            # Check API keys
            has_anthropic_key = bool(self.config.anthropic_api_key)
            has_openai_key = bool(self.config.openai_api_key)
            has_google_key = bool(self.config.google_api_key)

            if not has_anthropic_key and not has_openai_key and not has_google_key:
                return {
                    "status": "error",
                    "message": "No API keys configured",
                    "details": {},
                    "degraded_components": [],
                }

            status = "degraded" if self.degraded_components else "healthy"
            message = (
                "Agent system ready"
                if status == "healthy"
                else "Agent system running in degraded mode"
            )

            return {
                "status": status,
                "message": message,
                "details": {
                    "model": self.current_model,
                    "vector_store": str(self.config.vector_store_path),
                    "has_anthropic_key": has_anthropic_key,
                    "has_openai_key": has_openai_key,
                    "has_google_key": has_google_key,
                },
                "degraded_components": self.degraded_components,
            }

        except Exception as e:
            safe_error = sanitize_error(e)
            logger.error(f"Health check failed: {safe_error}")
            return {
                "status": "error",
                "message": safe_error,
                "details": {},
                "degraded_components": [],
            }
