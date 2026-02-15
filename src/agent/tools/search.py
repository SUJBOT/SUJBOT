"""
Search Tool — VL-only.

Jina v4 cosine similarity on vectors.vl_pages -> page images (base64 PNG).
Supports both text queries and image queries (from attachments or existing pages).
"""

import logging
from typing import List, Literal, Optional

from pydantic import Field, model_validator

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)

# Module-level availability check — runs once at import time
try:
    from rag_confidence import score_retrieval_general as _score_retrieval_general

    _HAS_RAG_CONFIDENCE = True
except ImportError:
    _HAS_RAG_CONFIDENCE = False
    logger.debug("rag_confidence not available — QPP scoring disabled")


class SearchInput(ToolInput):
    """Input for VL search tool."""

    query: str = Field(
        "",
        description=(
            "Natural language search query. "
            "Required for text search, optional for image search (used as log context)."
        ),
    )
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
    image_attachment_index: Optional[int] = Field(
        None,
        description=(
            "Search by image from user attachments. "
            "0-based index into the attachment images list "
            "(each PDF page is a separate entry). "
            "When set, the image embedding is used instead of text query."
        ),
        ge=0,
    )
    image_page_id: Optional[str] = Field(
        None,
        description=(
            "Search by image of an existing indexed page. "
            "Use the page_id from a previous search result "
            "(e.g. 'BZ_VR1_p005'). "
            "Finds pages visually similar to the given page."
        ),
    )

    @model_validator(mode="after")
    def validate_query_or_image(self) -> "SearchInput":
        has_text = bool(self.query and self.query.strip())
        has_image = (
            self.image_attachment_index is not None or self.image_page_id is not None
        )
        if not has_text and not has_image:
            raise ValueError(
                "Either 'query' (text search) or 'image_attachment_index'/'image_page_id' "
                "(image search) must be provided."
            )
        if self.image_attachment_index is not None and self.image_page_id is not None:
            raise ValueError(
                "Cannot use both 'image_attachment_index' and 'image_page_id'. Pick one."
            )
        return self


@register_tool
class SearchTool(BaseTool):
    """
    Semantic search over document corpus (VL mode).

    Jina v4 cosine similarity on page embeddings -> returns page images (base64 PNG).
    Supports text queries AND image queries (from attachments or existing pages).
    """

    name = "search"
    description = (
        "Search documents by text or image — returns page images for multimodal analysis. "
        "For image search, set image_attachment_index (attachment image) "
        "or image_page_id (existing page)."
    )
    detailed_help = """
    Semantic search using Jina v4 (2048-dim) embeddings.

    **Text search (default):**
    - search(query="What is safety margin?", k=5)
    - search(query="...", filter_document="doc_id")
    - search(query="...", filter_category="legislation")

    **Image search (from attachment):**
    - search(query="similar pages", image_attachment_index=0, k=5)
      Uses the image at index 0 from user attachments as the query.
      Each PDF page is a separate index.

    **Image search (from existing page):**
    - search(query="similar pages", image_page_id="BZ_VR1_p005", k=5)
      Finds pages visually similar to the given indexed page.
    """

    input_schema = SearchInput

    def _resolve_image_base64(
        self,
        image_attachment_index: Optional[int],
        image_page_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Resolve image source to base64 data.

        Returns:
            (base64_data, description) or (None, error_message)
        """
        if image_attachment_index is not None:
            attachment_images: List = self._request_context.get(
                "attachment_images", []
            )
            if not attachment_images:
                return None, "No attachment images available in this message."
            if image_attachment_index >= len(attachment_images):
                return None, (
                    f"Attachment index {image_attachment_index} out of range. "
                    f"Available: 0-{len(attachment_images) - 1} "
                    f"({len(attachment_images)} images)."
                )
            img = attachment_images[image_attachment_index]
            desc = img.get("filename", "attachment")
            if img.get("page") is not None:
                desc = f"{desc} page {img['page']}"
            return img["base64_data"], desc

        if image_page_id is not None:
            try:
                b64 = self.page_store.get_image_base64(image_page_id)
                return b64, f"page {image_page_id}"
            except Exception as e:
                return None, f"Failed to load image for page '{image_page_id}': {e}"

        return None, None

    def execute_impl(
        self,
        query: str = "",
        k: int = 5,
        filter_document: Optional[str] = None,
        filter_category: Optional[Literal["documentation", "legislation"]] = None,
        image_attachment_index: Optional[int] = None,
        image_page_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute VL search — text or image query.

        Args:
            query: Natural language query (required for text, optional context for image)
            k: Number of results
            filter_document: Optional document ID filter
            filter_category: Optional category filter
            image_attachment_index: Search by attachment image (0-based index)
            image_page_id: Search by existing page image (page_id)

        Returns:
            ToolResult with page data and base64 images
        """
        is_image_search = (
            image_attachment_index is not None or image_page_id is not None
        )

        if is_image_search:
            # Resolve image source
            image_b64, image_desc = self._resolve_image_base64(
                image_attachment_index, image_page_id
            )
            if image_b64 is None:
                return ToolResult(
                    success=False, data=None, error=image_desc or "No image source"
                )

            log_ctx = query[:30] if query else image_desc
            logger.info(
                "VL image search: '%s' (k=%d, category=%s)",
                log_ctx,
                k,
                filter_category,
            )

            try:
                results = self.vl_retriever.search_by_image(
                    image_base64=image_b64,
                    k=k,
                    document_filter=filter_document,
                    category_filter=filter_category,
                )
            except Exception as e:
                logger.error(f"VL image search error: {e}", exc_info=True)
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Image search failed: {type(e).__name__}: {e}",
                )

            search_method = "vl_jina_v4_image"
            confidence_result = None  # QPP not applicable for image search
        else:
            logger.info(
                "VL search: '%s...' (k=%d, category=%s)",
                query[:50],
                k,
                filter_category,
            )

            try:
                results, query_embedding = self.vl_retriever.search_with_embedding(
                    query=query,
                    k=k,
                    document_filter=filter_document,
                    category_filter=filter_category,
                )
            except Exception as e:
                logger.error(f"VL search error: {e}", exc_info=True)
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"VL search failed: {type(e).__name__}: {e}",
                )

            search_method = "vl_jina_v4"

            # QPP confidence scoring (best-effort — never blocks search)
            confidence_result = None
            if _HAS_RAG_CONFIDENCE:
                try:
                    all_similarities = self.vector_store.get_all_vl_similarities(
                        query_embedding
                    )
                    if len(all_similarities) > 0:
                        confidence_result = _score_retrieval_general(
                            query, all_similarities
                        )
                        logger.info(
                            "QPP confidence: %.3f (%s) for '%s...'",
                            confidence_result["confidence"],
                            confidence_result["band"],
                            query[:40],
                        )
                except (RuntimeError, ValueError, OSError) as e:
                    logger.warning(
                        "QPP confidence scoring failed (non-fatal): %s", e
                    )

        # Build result data (shared for both text and image search)
        return self._build_search_result(
            results=results,
            query=query,
            k=k,
            filter_document=filter_document,
            filter_category=filter_category,
            search_method=search_method,
            confidence_result=confidence_result,
        )

    def _build_search_result(
        self,
        results,
        query: str,
        k: int,
        filter_document: Optional[str],
        filter_category: Optional[str],
        search_method: str,
        confidence_result: Optional[dict] = None,
    ) -> ToolResult:
        """Build ToolResult from VLPageResult list (shared by text and image search)."""
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
                logger.warning(
                    f"Failed to load image for {r.page_id}: {e}", exc_info=True
                )
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
            "search_method": search_method,
            "final_count": len(formatted),
            "page_images": page_images,
        }
        if failed_pages:
            result_metadata["failed_pages"] = failed_pages

        # Include adaptive-k diagnostics if available
        adaptive_config = getattr(self.vl_retriever, "adaptive_config", None)
        if adaptive_config and adaptive_config.enabled and len(results) > 0:
            scores = [r.score for r in results]
            result_metadata["adaptive_k"] = {
                "enabled": True,
                "method": adaptive_config.method,
                "final_count": len(results),
                "score_range": (round(min(scores), 4), round(max(scores), 4)),
            }

        # Include QPP retrieval confidence if available
        if confidence_result:
            result_metadata["retrieval_confidence"] = {
                "score": round(confidence_result["confidence"], 4),
                "band": confidence_result["band"],
            }

        citations = [
            f"[{i+1}] {r['document_id']} p.{r['page_number']} (score: {r['score']:.3f})"
            for i, r in enumerate(formatted)
        ]

        # Append confidence annotation for the agent
        if confidence_result:
            citations.append(
                f"[Retrieval confidence: {confidence_result['confidence']:.2f} ({confidence_result['band']})]"
            )

        return ToolResult(
            success=True,
            data=formatted,
            citations=citations,
            metadata=result_metadata,
        )
