"""
Search Tool — VL-only.

Jina v4 cosine similarity on vectors.vl_pages -> page images (base64 PNG).
"""

import logging
from typing import Literal, Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class SearchInput(ToolInput):
    """Input for VL search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(
        5,
        description="Number of page results to return (default: 5, each page ~1600 tokens)",
        ge=1,
        le=100,
    )
    filter_document: Optional[str] = Field(
        None,
        description="Optional document ID to filter results (searches within specific document)",
    )
    filter_category: Optional[Literal["documentation", "legislation"]] = Field(
        None,
        description="Filter by document category: 'documentation' or 'legislation'",
    )


@register_tool
class SearchTool(BaseTool):
    """
    Semantic search over document corpus (VL mode).

    Jina v4 cosine similarity on page embeddings -> returns page images (base64 PNG).
    """

    name = "search"
    description = "Search documents — returns page images for multimodal analysis"
    detailed_help = """
    Semantic search using Jina v4 (2048-dim) embeddings.

    - Embeds query with Jina v4
    - Cosine similarity against page embeddings in PostgreSQL
    - Returns page images (base64 PNG) for multimodal LLM
    - Default k=5 (each page ~1600 tokens)

    **Usage:**
    - search(query="What is safety margin?", k=5)
    - search(query="...", filter_document="doc_id")
    - search(query="...", filter_category="legislation")
    """

    input_schema = SearchInput

    def execute_impl(
        self,
        query: str,
        k: int = 5,
        filter_document: Optional[str] = None,
        filter_category: Optional[Literal["documentation", "legislation"]] = None,
    ) -> ToolResult:
        """
        Execute VL search — returns page images for multimodal LLM.

        Args:
            query: Natural language query
            k: Number of results
            filter_document: Optional document ID filter
            filter_category: Optional category filter ('documentation' or 'legislation')

        Returns:
            ToolResult with page data and base64 images
        """
        logger.info(f"VL search: '{query[:50]}...' (k={k}, category={filter_category})")

        try:
            results = self.vl_retriever.search(
                query=query,
                k=k,
                document_filter=filter_document,
                category_filter=filter_category,
            )

            # Build text data for logging/history
            formatted = []
            page_images = []
            failed_pages = []

            for r in results:
                formatted.append(
                    {
                        "page_id": r.page_id,
                        "document_id": r.document_id,
                        "page_number": r.page_number,
                        "score": round(r.score, 4),
                    }
                )

                # Load base64 image for multimodal injection
                try:
                    b64_data = self.page_store.get_image_base64(r.page_id)
                    page_images.append(
                        {
                            "page_id": r.page_id,
                            "base64_data": b64_data,
                            "media_type": "image/png",
                            "page_number": r.page_number,
                            "document_id": r.document_id,
                            "score": round(r.score, 4),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to load image for {r.page_id}: {e}", exc_info=True)
                    failed_pages.append(r.page_id)

            if failed_pages and not page_images:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"All {len(failed_pages)} page images failed to load: {failed_pages}",
                )

            result_metadata = {
                "query": query,
                "k": k,
                "filter_document": filter_document,
                "filter_category": filter_category,
                "search_method": "vl_jina_v4",
                "final_count": len(formatted),
                "page_images": page_images,
            }
            if failed_pages:
                result_metadata["failed_pages"] = failed_pages

            return ToolResult(
                success=True,
                data=formatted,
                citations=[
                    f"[{i+1}] {r['document_id']} p.{r['page_number']} (score: {r['score']:.3f})"
                    for i, r in enumerate(formatted)
                ],
                metadata=result_metadata,
            )

        except Exception as e:
            logger.error(f"VL search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"VL search failed: {type(e).__name__}: {e}",
            )
