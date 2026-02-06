"""
Single-agent runner for SUJBOT.

Replaces the multi-agent system with a single autonomous agent
that has access to all RAG tools and a unified system prompt.
"""

from .runner import SingleAgentRunner

__all__ = ["SingleAgentRunner"]
