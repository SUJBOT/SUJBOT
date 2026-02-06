"""
Agent shared utilities and configuration.

This module provides shared configuration, tools, and providers
used by the single-agent runner and multi-agent system.

Modules:
- config: Shared configuration (AgentConfig, ToolConfig)
- tools: 5 RAG tools (search, expand_context, get_document_info, get_document_list, get_stats)
- providers: LLM provider abstractions (Anthropic, OpenAI, Google, DeepInfra)
"""

from .config import AgentConfig

__version__ = "2.0.0"  # Multi-agent system
__all__ = ["AgentConfig"]
