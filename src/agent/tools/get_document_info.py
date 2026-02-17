"""
Get Document Info Tool — VL-only.

Retrieves document information from vectors.vl_pages and vectors.documents.
"""

import logging
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetDocumentInfoInput(ToolInput):
    """Input for get_document_info tool."""

    document_id: Optional[str] = Field(None, description="Document ID (optional - if None, returns list of all documents)")
    info_type: str = Field(
        ...,
        description="Type of information: 'list' (all documents with categories - requires document_id=None), 'summary' (page summaries overview), 'metadata' (page count, category)",
    )


@register_tool
class GetDocumentInfoTool(BaseTool):
    """Get document information from VL page store."""

    name = "get_document_info"
    description = "Get document info/metadata or list all documents"
    detailed_help = """
    Retrieves document information from the VL page store.

    Info types:
    - 'list': List all indexed documents with categories (requires document_id=None)
    - 'summary': Page count and page summaries for a specific document
    - 'metadata': Page count, document category, page info

    **When to use:**
    - 'list': "What documents are available?", corpus discovery
    - 'summary': Need document overview before detailed search
    - 'metadata': Need comprehensive stats

    **Data source:** vectors.vl_pages + vectors.documents
    """
    input_schema = GetDocumentInfoInput

    def execute_impl(
        self, document_id: Optional[str] = None, info_type: str = "summary"
    ) -> ToolResult:
        try:
            # Handle 'list' info_type (all documents)
            if info_type == "list":
                if document_id is not None:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="info_type='list' requires document_id=None (lists all documents)",
                    )

                doc_ids = self.vector_store.get_document_list()
                categories = self.vector_store.get_document_categories()

                document_list = [
                    {"id": doc_id, "category": categories.get(doc_id, "unknown")}
                    for doc_id in sorted(doc_ids)
                ]

                return ToolResult(
                    success=True,
                    data={"documents": document_list, "count": len(document_list)},
                    metadata={"total_documents": len(document_list)},
                )

            # For other info_types, document_id is required
            if document_id is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"document_id is required for info_type='{info_type}' (use info_type='list' with document_id=None to list all documents)",
                )

            if info_type not in ("summary", "metadata"):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid info_type: {info_type}. Must be 'list', 'summary', or 'metadata'",
                )

            # Both 'summary' and 'metadata' need pages and categories
            pages = self._get_document_pages(document_id)
            if not pages:
                return ToolResult(
                    success=True,
                    data=None,
                    metadata={"document_id": document_id, "found": False},
                )

            categories = self.vector_store.get_document_categories()
            category = categories.get(document_id, "unknown")
            page_count = len(pages)
            base_metadata = {"document_id": document_id, "page_count": page_count}

            if info_type == "summary":
                page_summaries = [
                    {"page_number": p["page_number"], "summary": meta.get("page_summary", "")}
                    for p in pages
                    if (meta := p.get("metadata") or {}).get("page_summary")
                ]

                return ToolResult(
                    success=True,
                    data={
                        "document_id": document_id,
                        "page_count": page_count,
                        "category": category,
                        "page_summaries": page_summaries[:20],
                        "has_more": len(page_summaries) > 20,
                    },
                    metadata=base_metadata,
                )

            else:  # metadata
                pages_with_summary = sum(
                    1 for p in pages
                    if (p.get("metadata") or {}).get("page_summary")
                )

                return ToolResult(
                    success=True,
                    data={
                        "document_id": document_id,
                        "page_count": page_count,
                        "category": category,
                        "pages_with_summary": pages_with_summary,
                        "page_numbers": [p["page_number"] for p in pages],
                    },
                    metadata=base_metadata,
                )

        except Exception as e:
            logger.error(f"Get document info failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _get_document_pages(self, document_id: str):
        """Get all VL pages for a document, ordered by page_number."""
        import json as _json
        from ...storage.postgres_adapter import _run_async_safe

        async def _fetch():
            await self.vector_store._ensure_pool()
            async with self.vector_store.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT page_id, document_id, page_number, metadata
                    FROM vectors.vl_pages
                    WHERE document_id = $1
                    ORDER BY page_number
                    """,
                    document_id,
                )
                results = []
                for row in rows:
                    meta = row.get("metadata") or {}
                    # asyncpg may return jsonb as string — parse if needed
                    if isinstance(meta, str):
                        try:
                            meta = _json.loads(meta)
                        except (ValueError, TypeError):
                            meta = {}
                    results.append({
                        "page_id": row["page_id"],
                        "document_id": row["document_id"],
                        "page_number": row["page_number"],
                        "metadata": meta,
                    })
                return results

        return _run_async_safe(_fetch(), operation_name="get_document_pages")
