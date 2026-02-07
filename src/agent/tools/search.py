"""
Search Tool — dual-mode (VL / OCR).

VL mode: Jina v4 cosine similarity → page images (base64 PNG).
OCR mode: HyDE + Expansion Fusion → text chunks.

Active mode determined by config.json → "architecture".
"""

import logging
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import create_fusion_retriever, format_chunk_result, generate_citation

logger = logging.getLogger(__name__)


class SearchInput(ToolInput):
    """Input for fusion search tool."""

    query: str = Field(..., description="Natural language search query")
    k: int = Field(
        10,
        description="Number of results to return (default: 10 for OCR, 5 for VL mode — page images are ~16x larger than text chunks)",
        ge=1,
        le=100,
    )
    filter_document: Optional[str] = Field(
        None,
        description="Optional document ID to filter results (searches within specific document)",
    )


@register_tool
class SearchTool(BaseTool):
    """
    Semantic search over document corpus (VL or OCR mode).

    VL mode:  Jina v4 cosine similarity → returns page images (base64 PNG).
    OCR mode: HyDE + Expansion Fusion → returns text chunks.
    """

    name = "search"
    description = "Search documents — returns page images (VL mode) or text chunks (OCR mode)"
    detailed_help = """
    Semantic search with automatic mode dispatch.

    **VL mode:**
    - Embeds query with Jina v4 (2048-dim)
    - Cosine similarity against page embeddings in PostgreSQL
    - Returns page images (base64 PNG) for multimodal LLM
    - Default k=5 (each page ~1600 tokens)

    **OCR mode:**
    - HyDE + Expansion Fusion with Qwen3-Embedding-8B (4096-dim)
    - Returns text chunks (512 tokens each)
    - Default k=10

    **Usage:**
    - search(query="What is safety margin?", k=5)
    - search(query="...", filter_document="doc_id")
    """

    input_schema = SearchInput
    requires_reranker = False  # No reranking in fusion pipeline

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_retriever = None  # Lazy initialization

    def _get_fusion_retriever(self):
        """Lazy initialization of FusionRetriever using SSOT factory."""
        if self._fusion_retriever is None:
            # Use shared factory from _utils.py (SSOT for FusionRetriever creation)
            self._fusion_retriever = create_fusion_retriever(
                vector_store=self.vector_store,
                config=self.config,
                layer=3,  # Layer 3 = chunk-level search
            )
        return self._fusion_retriever

    def execute_impl(
        self,
        query: str,
        k: int = 10,
        filter_document: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute search — dispatches to VL or OCR mode based on architecture config.

        Args:
            query: Natural language query
            k: Number of results
            filter_document: Optional document ID filter

        Returns:
            ToolResult with formatted chunks/pages and citations
        """
        if self._is_vl_mode():
            # VL pages are ~1600 tokens each vs ~100 for text chunks;
            # default to 5 unless the caller explicitly requested more
            vl_k = min(k, 5) if k == 10 else k
            return self._execute_vl(query, vl_k, filter_document)
        return self._execute_ocr(query, k, filter_document)

    def _execute_vl(
        self,
        query: str,
        k: int = 5,
        filter_document: Optional[str] = None,
    ) -> ToolResult:
        """
        VL mode: search page embeddings, return page images for multimodal LLM.

        Returns ToolResult with:
        - data: list of {page_id, document_id, page_number, score}
        - metadata.page_images: list of {page_id, base64_data, media_type, page_number, document_id, score}
        """
        logger.info(f"VL search: '{query[:50]}...' (k={k})")

        try:
            results = self.vl_retriever.search(
                query=query,
                k=k,
                document_filter=filter_document,
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
                    logger.warning(f"Failed to load image for {r.page_id}: {e}")
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

    def _execute_ocr(
        self,
        query: str,
        k: int = 10,
        filter_document: Optional[str] = None,
    ) -> ToolResult:
        """OCR mode: HyDE + Expansion Fusion search (original implementation)."""
        logger.info(f"Fusion search: '{query[:50]}...' (k={k})")

        try:
            # Get fusion retriever
            retriever = self._get_fusion_retriever()

            # Execute fusion search
            chunks = retriever.search(
                query=query,
                k=k,
                document_filter=filter_document,
            )

            # Format results
            formatted = [format_chunk_result(c) for c in chunks]
            final_count = len(formatted)

            # Generate citations
            citations = [
                generate_citation(c, i + 1, format="inline") for i, c in enumerate(formatted)
            ]

            # Build metadata
            result_metadata = {
                "query": query,
                "k": k,
                "filter_document": filter_document,
                "search_method": "hyde_expansion_fusion",
                "final_count": final_count,
                "fusion_weights": {
                    "hyde": retriever.config.hyde_weight,
                    "expansion": retriever.config.expansion_weight,
                },
            }

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata=result_metadata,
            )

        except ValueError as e:
            # Configuration error (missing API key, etc.)
            logger.error(f"Search configuration error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Configuration error: {e}. Check DEEPINFRA_API_KEY in .env",
            )

        except ConnectionError as e:
            # Database connection error
            logger.error(f"Database connection error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Database connection error: {e}",
            )

        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Search failed: {type(e).__name__}: {e}",
            )
