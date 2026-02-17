"""
Tool execution models — Pydantic models for tracking tool calls and metrics.

Extracted from multi_agent/core/state.py (only the models used by ToolAdapter).
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolExecution(BaseModel):
    """Record of a single tool execution."""

    tool_name: str
    agent_name: str
    timestamp: datetime
    duration_ms: float
    input_tokens: int
    output_tokens: int
    success: bool
    error: Optional[str] = None
    result_summary: str  # First 200 chars of result

    # Hallucination detection fields
    was_hallucinated: bool = False  # True if tool doesn't exist
    validation_error: Optional[str] = None  # Input validation error


class ToolStats(BaseModel):
    """Statistics for a single tool."""

    tool_name: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    hallucination_count: int = 0
    validation_error_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0

    def success_rate(self) -> float:
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count


class ToolUsageMetrics(BaseModel):
    """Aggregated tool usage metrics for evaluation."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    hallucinated_calls: int = 0
    validation_errors: int = 0

    tool_stats: Dict[str, ToolStats] = Field(default_factory=dict)

    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0

    def hallucination_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.hallucinated_calls / self.total_calls

    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls

    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return (self.hallucinated_calls + self.validation_errors + self.failed_calls) / self.total_calls

    def record_execution(self, execution: ToolExecution) -> None:
        self.total_calls += 1
        self.total_duration_ms += execution.duration_ms

        if execution.was_hallucinated:
            self.hallucinated_calls += 1
        elif execution.validation_error:
            self.validation_errors += 1
        elif execution.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        tool_name = execution.tool_name
        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolStats(tool_name=tool_name)

        stats = self.tool_stats[tool_name]
        stats.call_count += 1
        stats.total_duration_ms += execution.duration_ms

        if execution.was_hallucinated:
            stats.hallucination_count += 1
        elif execution.validation_error:
            stats.validation_error_count += 1
        elif execution.success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

        stats.avg_duration_ms = stats.total_duration_ms / stats.call_count
        self.avg_duration_ms = self.total_duration_ms / self.total_calls

    def to_feedback_dict(self) -> Dict[str, float]:
        return {
            "tool_success_rate": self.success_rate(),
            "tool_hallucination_rate": self.hallucination_rate(),
            "tool_error_rate": self.error_rate(),
        }
