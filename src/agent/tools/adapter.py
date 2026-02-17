"""
Tool Adapter — Bridge between LLM agents and the tool registry.

Handles tool lookup, input validation, hallucination detection,
execution tracking, and result formatting.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError as PydanticValidationError

from . import get_registry, ToolResult
from .models import ToolExecution, ToolUsageMetrics

logger = logging.getLogger(__name__)


class ToolErrorType(str, Enum):
    HALLUCINATION = "hallucination"
    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    SUCCESS = "success"


class ToolAdapter:
    """
    Adapts existing tool infrastructure for LLM agents.

    Handles tool lookup, input validation, execution, error handling,
    result formatting, and metric tracking.
    """

    def __init__(self, tool_registry=None):
        if tool_registry is None:
            tool_registry = get_registry()

        self.registry = tool_registry
        self.execution_history: List[ToolExecution] = []
        self.usage_metrics = ToolUsageMetrics()
        self._available_tools_cache: Optional[Set[str]] = None

        logger.info(f"ToolAdapter initialized with {len(self.registry)} tools")

    async def execute(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        agent_name: str,
    ) -> Dict[str, Any]:
        """Execute a tool with hallucination detection and input validation."""
        start_time = datetime.now()

        # Check if tool exists (hallucination detection)
        tool = self.registry.get_tool(tool_name)

        if tool is None:
            logger.warning(
                f"HALLUCINATION DETECTED: Agent '{agent_name}' called "
                f"non-existent tool '{tool_name}'"
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=f"Tool '{tool_name}' does not exist. Available tools: {self._get_available_tools_str()}",
                start_time=start_time,
                error_type=ToolErrorType.HALLUCINATION,
                was_hallucinated=True,
            )

        # Validate inputs against Pydantic schema
        validation_error = self._validate_inputs(tool, inputs)
        if validation_error:
            logger.warning(
                f"INPUT VALIDATION FAILED: Tool '{tool_name}' by agent '{agent_name}': {validation_error}"
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=f"Invalid inputs for tool '{tool_name}': {validation_error}",
                start_time=start_time,
                error_type=ToolErrorType.VALIDATION,
                validation_error=validation_error,
            )

        # Execute tool
        try:
            result: ToolResult = tool.execute(**inputs)

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            execution = ToolExecution(
                tool_name=tool_name,
                agent_name=agent_name,
                timestamp=start_time,
                duration_ms=duration_ms,
                input_tokens=self._estimate_tokens(inputs),
                output_tokens=result.estimated_tokens,
                success=result.success,
                error=result.error,
                result_summary=self._summarize_result(result),
                was_hallucinated=False,
                validation_error=None,
            )
            self.execution_history.append(execution)
            self.usage_metrics.record_execution(execution)

            return {
                "success": result.success,
                "data": result.data,
                "citations": result.citations,
                "metadata": {
                    **result.metadata,
                    "agent_name": agent_name,
                    "duration_ms": duration_ms,
                    "estimated_tokens": result.estimated_tokens,
                },
                "error": result.error,
                "error_type": ToolErrorType.SUCCESS if result.success else ToolErrorType.EXECUTION,
            }

        except TimeoutError as e:
            logger.error(f"Tool timeout: {tool_name} by {agent_name}")
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=f"Tool '{tool_name}' timed out: {e}",
                start_time=start_time,
                error_type=ToolErrorType.TIMEOUT,
            )

        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name} by {agent_name}: {e}",
                exc_info=True,
            )
            return self._format_error_result(
                tool_name=tool_name,
                agent_name=agent_name,
                error=str(e),
                start_time=start_time,
                error_type=ToolErrorType.EXECUTION,
            )

    def _validate_inputs(self, tool, inputs: Dict[str, Any]) -> Optional[str]:
        if not hasattr(tool, "input_schema") or tool.input_schema is None:
            return None

        try:
            tool.input_schema(**inputs)
            return None
        except PydanticValidationError as e:
            errors = e.errors()
            if errors:
                first_error = errors[0]
                field = ".".join(str(loc) for loc in first_error.get("loc", []))
                msg = first_error.get("msg", "validation error")
                return f"{field}: {msg}"
            return str(e)
        except TypeError as e:
            return str(e)

    def _get_available_tools_str(self) -> str:
        if self._available_tools_cache is None:
            self._available_tools_cache = set(self.registry._tools.keys())
        return ", ".join(sorted(self._available_tools_cache)[:10]) + "..."

    def _format_error_result(
        self,
        tool_name: str,
        agent_name: str,
        error: str,
        start_time: datetime,
        error_type: ToolErrorType = ToolErrorType.EXECUTION,
        was_hallucinated: bool = False,
        validation_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        execution = ToolExecution(
            tool_name=tool_name,
            agent_name=agent_name,
            timestamp=start_time,
            duration_ms=duration_ms,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=error,
            result_summary=f"Error ({error_type.value}): {error[:80]}",
            was_hallucinated=was_hallucinated,
            validation_error=validation_error,
        )
        self.execution_history.append(execution)
        self.usage_metrics.record_execution(execution)

        return {
            "success": False,
            "data": None,
            "citations": [],
            "metadata": {
                "agent_name": agent_name,
                "duration_ms": duration_ms,
                "error_type": error_type.value,
                "was_hallucinated": was_hallucinated,
                "validation_error": validation_error,
            },
            "error": error,
            "error_type": error_type,
        }

    def _estimate_tokens(self, data: Any) -> int:
        try:
            json_str = json.dumps(data, ensure_ascii=False, default=str)
            return len(json_str) // 4
        except (TypeError, ValueError, OverflowError):
            return 0

    def _summarize_result(self, result: ToolResult) -> str:
        try:
            data_str = json.dumps(result.data, ensure_ascii=False, default=str)
            return data_str[:200] + ("..." if len(data_str) > 200 else "")
        except (TypeError, ValueError, OverflowError):
            return str(result.data)[:200]

    def get_available_tools(self) -> List[str]:
        return list(self.registry._tools.keys())

    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema in LLM-compatible format (Anthropic/OpenAI)."""
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            return None

        try:
            if hasattr(tool, "input_schema") and tool.input_schema:
                if hasattr(tool.input_schema, "model_json_schema"):
                    pydantic_schema = tool.input_schema.model_json_schema()
                else:
                    pydantic_schema = tool.input_schema.schema()

                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": pydantic_schema.get("properties", {}),
                        "required": pydantic_schema.get("required", []),
                    },
                }
            else:
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"Failed to get schema for tool {tool_name}: {e}", exc_info=True)
            return None

    def get_execution_stats(self) -> Dict[str, Any]:
        if not self.execution_history:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "avg_duration_ms": 0,
                "total_tokens": 0,
                "hallucination_rate": 0.0,
                "validation_error_rate": 0.0,
            }

        successful = [e for e in self.execution_history if e.success]
        failed = [e for e in self.execution_history if not e.success]
        hallucinated = [e for e in self.execution_history if e.was_hallucinated]
        validation_errors = [e for e in self.execution_history if e.validation_error]

        total_duration = sum(e.duration_ms for e in self.execution_history)
        total_tokens = sum(e.input_tokens + e.output_tokens for e in self.execution_history)
        total_count = len(self.execution_history)

        return {
            "total_executions": total_count,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / total_count * 100,
            "avg_duration_ms": total_duration / total_count,
            "total_tokens": total_tokens,
            "hallucinated": len(hallucinated),
            "validation_errors": len(validation_errors),
            "hallucination_rate": len(hallucinated) / total_count,
            "validation_error_rate": len(validation_errors) / total_count,
        }

    def get_usage_metrics(self) -> ToolUsageMetrics:
        return self.usage_metrics

    def reset_metrics(self) -> None:
        self.usage_metrics = ToolUsageMetrics()

    def clear_history(self) -> None:
        self.execution_history.clear()
        self.reset_metrics()
        self._available_tools_cache = None
