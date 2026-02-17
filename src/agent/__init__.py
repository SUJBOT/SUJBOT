"""
Agent shared utilities and configuration.

Modules:
- config: Shared configuration (AgentConfig, ToolConfig)
- tools: RAG tools (search, expand_context, get_document_info, get_document_list, get_stats, graph_*, web_search, compliance_check)
- providers: LLM provider abstractions (Anthropic, OpenAI, Google, DeepInfra)
"""

from .config import AgentConfig

__all__ = ["AgentConfig"]
