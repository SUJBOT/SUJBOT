"""
Expand Context Tool — VL-only.

Expands page context by returning adjacent pages from the same document.
"""

import logging
from typing import List

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class ExpandContextInput(ToolInput):
    """Input for expand_context tool."""

    page_ids: List[str] = Field(
        ...,
        description="List of page IDs (format: {doc_id}_p{NNN}) to expand with adjacent pages",
    )
    k: int = Field(
        3,
        description="Number of additional pages in each direction (before/after)",
        ge=1,
        le=10,
    )


@register_tool
class ExpandContextTool(BaseTool):
    """Expand page context with adjacent pages."""

    name = "expand_context"
    description = "Get adjacent pages for context expansion"
    detailed_help = """
    Expand pages with neighboring pages from the same document.

    - Input: page IDs (format: {doc_id}_p{NNN})
    - Returns: adjacent page images as base64 for multimodal LLM
    - Pages are retrieved by page_number +/- k

    **Best practices:**
    - Start with k=3, increase if needed
    - Use after search() to get more context around found pages
    """

    input_schema = ExpandContextInput

    def execute_impl(self, page_ids: List[str], k: int = 3) -> ToolResult:
        try:
            from ...vl.page_store import PageStore

            expanded_results = []
            all_page_images = []

            for page_id in page_ids:
                # Parse page_id -> (document_id, page_number)
                try:
                    document_id, page_number = PageStore.page_id_to_components(page_id)
                except ValueError as e:
                    logger.warning(f"Invalid page_id '{page_id}': {e}")
                    continue

                # Query adjacent pages from PostgreSQL
                adjacent_pages = self.vector_store.get_adjacent_vl_pages(
                    document_id=document_id,
                    page_number=page_number,
                    k=k,
                )

                # Load base64 images for multimodal injection
                for page in adjacent_pages:
                    adj_page_id = page["page_id"]
                    try:
                        b64_data = self.page_store.get_image_base64(adj_page_id)
                        all_page_images.append(
                            {
                                "page_id": adj_page_id,
                                "base64_data": b64_data,
                                "media_type": "image/png",
                                "page_number": page["page_number"],
                                "document_id": page["document_id"],
                                "position": (
                                    "before" if page["page_number"] < page_number else "after"
                                ),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load image for {adj_page_id}: {e}")

                expansion = {
                    "target_page_id": page_id,
                    "document_id": document_id,
                    "page_number": page_number,
                    "expanded_pages": [
                        {
                            "page_id": p["page_id"],
                            "document_id": p["document_id"],
                            "page_number": p["page_number"],
                            "position": "before" if p["page_number"] < page_number else "after",
                        }
                        for p in adjacent_pages
                    ],
                    "expansion_count": len(adjacent_pages),
                }
                expanded_results.append(expansion)

            # Collect unique document citations
            citations = list({r["document_id"] for r in expanded_results})

            return ToolResult(
                success=True,
                data={"expansions": expanded_results},
                citations=citations,
                metadata={
                    "page_count": len(page_ids),
                    "page_images": all_page_images,
                },
            )

        except Exception as e:
            logger.error(f"Expand context failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
