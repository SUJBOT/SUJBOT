"""
Get Document List Tool

Lists all documents in the vector store for orchestrator routing decisions.
"""

import logging
from typing import List

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetDocumentListInput(ToolInput):
    """Input for get_document_list tool (no parameters needed)."""
    pass


@register_tool
class GetDocumentListTool(BaseTool):
    """List all available documents in the corpus."""

    name = "get_document_list"
    description = "Get list of all documents in the vector store with categories"
    detailed_help = """
    Returns a list of all document IDs, categories, and summaries from the vector store.

    **Use cases:**
    - Orchestrator routing: understand what documents exist
    - Query scoping: check if relevant documents are indexed
    - Corpus overview: see available document coverage
    - Category awareness: identify legislation vs documentation

    **Returns:**
    - document_ids: List of unique document identifiers
    - document_count: Total number of documents
    - documents: List of documents with id, category, and summary
    - by_category: Documents grouped by category (legislation/documentation)
    """

    input_schema = GetDocumentListInput
    requires_reranker = False

    def execute_impl(self) -> ToolResult:
        """
        Get list of all documents in vector store, grouped by category.

        Returns:
            ToolResult with document list, summaries, and categories
        """
        try:
            # Get document list from vector store
            document_ids: List[str] = []
            summaries: dict = {}
            categories: dict = {}

            # Try PostgreSQL adapter method first
            if hasattr(self.vector_store, 'get_document_list'):
                document_ids = self.vector_store.get_document_list() or []

            # Try to get summaries if available
            if hasattr(self.vector_store, 'get_document_summaries'):
                summaries = self.vector_store.get_document_summaries() or {}

            # Get document categories
            if hasattr(self.vector_store, 'get_document_categories'):
                categories = self.vector_store.get_document_categories() or {}

            # Sort for consistent output
            document_ids = sorted(document_ids)

            # Build response with summaries and categories
            documents = []
            for doc_id in document_ids:
                doc_info = {
                    "id": doc_id,
                    "category": categories.get(doc_id, "documentation"),
                    "summary": summaries.get(doc_id, "No summary available"),
                }
                documents.append(doc_info)

            # Group by category for clarity
            legislation = [d for d in documents if d["category"] == "legislation"]
            documentation = [d for d in documents if d["category"] == "documentation"]

            return ToolResult(
                success=True,
                data={
                    "document_count": len(document_ids),
                    "document_ids": document_ids,
                    "documents": documents,
                    "by_category": {
                        "legislation": [d["id"] for d in legislation],
                        "documentation": [d["id"] for d in documentation],
                    },
                },
                metadata={
                    "source": "vector_store",
                    "has_summaries": bool(summaries),
                    "has_categories": bool(categories),
                },
            )

        except AttributeError as e:
            logger.error(f"Vector store missing required method: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error="Vector store not properly initialized. Check DATABASE_URL configuration.",
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Database connection failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Database connection failed: {e}. Verify DATABASE_URL in .env file.",
            )
        except Exception as e:
            logger.error(f"Unexpected error getting document list: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Internal error: {type(e).__name__}: {e}",
            )
