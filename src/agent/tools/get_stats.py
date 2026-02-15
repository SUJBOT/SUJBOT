"""
Get Stats Tool

Corpus/index statistics for the RAG system.
"""

import logging
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetStatsInput(ToolInput):
    """Input for get_stats tool."""

    stat_scope: str = Field(
        ...,
        description="Statistics scope: 'corpus' (overall stats), 'index' (VL index info), 'document' (per-document stats)",
    )


@register_tool
class GetStatsTool(BaseTool):
    """Get corpus/index statistics."""

    name = "get_stats"
    description = "Get corpus/index stats"
    detailed_help = """
    Get statistics about corpus, index, or documents.

    **Stat scopes:**
    - 'corpus': Overall document counts, page counts
    - 'index': VL index info (Jina v4, 2048-dim)
    - 'document': Per-document statistics

    **When to use:**
    - "How many documents?"
    - "Corpus statistics"
    - "Index information"

    **Method:** Metadata aggregation (fast, no search)
    """

    input_schema = GetStatsInput

    def execute_impl(self, stat_scope: str) -> ToolResult:
        if stat_scope not in ("corpus", "index", "document"):
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid stat_scope: {stat_scope}. Must be 'corpus', 'index', or 'document'",
            )

        try:
            vs_stats = self.vector_store.get_stats()
            stats = {"vector_store": vs_stats}

            doc_count = vs_stats.get("documents", 0)
            stats["unique_documents"] = doc_count

            doc_list = self.vector_store.get_document_list()
            stats["document_list"] = sorted(doc_list) if doc_list else []

            if stat_scope == "index":
                stats["documents"] = {
                    "count": doc_count,
                    "document_ids": stats["document_list"],
                }
                stats["architecture"] = "vl"
                stats["embedding_model"] = "jina-embeddings-v4"
                stats["embedding_dimensions"] = 2048

            return ToolResult(
                success=True,
                data=stats,
                metadata={
                    "stat_scope": stat_scope,
                    "stat_categories": list(stats.keys()),
                },
            )

        except Exception as e:
            logger.error(f"Get stats failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
