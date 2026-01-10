"""Routing system for multi-agent framework.

This package handles:
- Agent sequence determination
- LangGraph workflow construction

Note: ComplexityAnalyzer was removed (deprecated 2024-11-28).
Complexity scoring is now handled by orchestrator's unified LLM analysis.
"""

from .workflow_builder import WorkflowBuilder

__all__ = ["WorkflowBuilder"]
