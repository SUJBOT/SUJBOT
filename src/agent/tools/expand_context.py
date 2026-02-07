"""
Expand Context Tool

Expand chunk/page context with adjacent or related content for both OCR and VL modes.
"""

import logging
from typing import Dict, List, Optional
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result

logger = logging.getLogger(__name__)


class ExpandContextInput(ToolInput):
    """Input for unified expand_context tool."""

    chunk_ids: List[str] = Field(
        ...,
        description=(
            "List of chunk IDs (OCR mode) or page IDs (VL mode, format: {doc_id}_p{NNN}) to expand"
        ),
    )
    expansion_mode: str = Field(
        ...,
        description=(
            "Expansion mode: 'adjacent' (before/after chunks/pages), "
            "'section' (same section, OCR only), 'similarity' (semantically similar, OCR only), "
            "'hybrid' (section + similarity, OCR only). "
            "In VL mode, all modes fall back to 'adjacent' (neighboring pages)."
        ),
    )
    k: int = Field(
        3,
        description="Number of additional chunks/pages per input (in each direction for adjacent)",
        ge=1,
        le=10,
    )


@register_tool
class ExpandContextTool(BaseTool):
    """Expand chunk context."""

    name = "expand_context"
    description = "Expand context — get adjacent pages (VL) or neighboring chunks (OCR)"
    detailed_help = """
    Expand chunks/pages with additional surrounding or related context.

    **OCR mode expansion modes:**
    - 'adjacent': Chunks immediately before/after (linear context)
    - 'section': All chunks from same section
    - 'similarity': Semantically similar chunks
    - 'hybrid': Combination of section + similarity

    **VL mode:**
    - All modes fall back to 'adjacent' (neighboring pages by page_number)
    - Input: page IDs (format: {doc_id}_p{NNN})
    - Returns: adjacent page images as base64 for multimodal LLM

    **Best practices:**
    - Use 'adjacent' for simple before/after context
    - Start with k=3, increase if needed
    """

    input_schema = ExpandContextInput

    def execute_impl(self, chunk_ids: List[str], expansion_mode: str, k: int = 3) -> ToolResult:
        if self._is_vl_mode():
            return self._execute_vl(page_ids=chunk_ids, expansion_mode=expansion_mode, k=k)

        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            # Find target chunks
            target_chunks = []
            for chunk_id in chunk_ids:
                for meta in layer3_chunks:
                    if meta.get("chunk_id") == chunk_id:
                        target_chunks.append(meta)
                        break

            if not target_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No chunks found for IDs: {chunk_ids}",
                    metadata={"chunk_ids": chunk_ids},
                )

            expanded_results = []

            for target_chunk in target_chunks:
                expansion = {
                    "target_chunk": format_chunk_result(target_chunk),
                    "expanded_chunks": [],
                }

                if expansion_mode == "adjacent":
                    # Get adjacent chunks (before/after)
                    context_chunks = self._expand_adjacent(target_chunk, layer3_chunks, k=k)
                    expansion["expanded_chunks"] = context_chunks
                    expansion["context_before"] = [
                        c for c in context_chunks if c.get("position") == "before"
                    ]
                    expansion["context_after"] = [
                        c for c in context_chunks if c.get("position") == "after"
                    ]

                elif expansion_mode == "section":
                    # Get chunks from same section
                    section_chunks = self._expand_by_section(target_chunk, layer3_chunks, k=k)
                    expansion["expanded_chunks"] = section_chunks

                elif expansion_mode == "similarity":
                    # Get semantically similar chunks
                    try:
                        similarity_chunks = self._expand_by_similarity(target_chunk, k=k)
                        expansion["expanded_chunks"] = similarity_chunks
                    except Exception as e:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Similarity expansion failed: {e}",
                        )

                elif expansion_mode == "hybrid":
                    # Combine section + similarity
                    section_chunks = self._expand_by_section(target_chunk, layer3_chunks, k=k // 2)
                    expansion["expanded_chunks"].extend(section_chunks)

                    try:
                        similarity_chunks = self._expand_by_similarity(target_chunk, k=k // 2)
                        expansion["expanded_chunks"].extend(similarity_chunks)
                    except Exception as e:
                        expansion["expansion_warning"] = f"Similarity expansion failed: {str(e)}"

                    # Remove duplicates
                    seen_ids = {target_chunk.get("chunk_id")}
                    unique_expanded = []
                    for chunk in expansion["expanded_chunks"]:
                        chunk_id = chunk.get("chunk_id")
                        if chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            unique_expanded.append(chunk)
                    expansion["expanded_chunks"] = unique_expanded

                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Invalid expansion_mode: {expansion_mode}. Must be 'adjacent', 'section', 'similarity', or 'hybrid'",
                    )

                expansion["expansion_count"] = len(expansion["expanded_chunks"])
                expanded_results.append(expansion)

            # Collect citations
            citations = []
            for result in expanded_results:
                doc_id = result["target_chunk"].get("document_id", "Unknown")
                if doc_id not in citations:
                    citations.append(doc_id)

            return ToolResult(
                success=True,
                data={"expansions": expanded_results, "expansion_mode": expansion_mode},
                citations=citations,
                metadata={
                    "chunk_count": len(target_chunks),
                    "expansion_mode": expansion_mode,
                },
            )

        except Exception as e:
            logger.error(f"Expand context failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _expand_adjacent(self, target_chunk: Dict, all_chunks: List[Dict], k: int) -> List[Dict]:
        """
        Get adjacent chunks (before/after) based on chunk_id sequence.

        IMPORTANT: We use document_id + chunk_id ordering, NOT section_id.
        This is because during PDF extraction, headers and their bullet list items
        get different section_ids (sequential: sec_383, sec_384, sec_385...) even
        though they are logically adjacent content.

        By using chunk_id sequence, we can correctly find content that follows
        a header regardless of section_id assignment.
        """
        document_id = target_chunk.get("document_id")
        chunk_id = target_chunk.get("chunk_id")

        if not document_id or not chunk_id:
            logger.warning("expand_adjacent: missing document_id or chunk_id")
            return []

        # Helper to extract numeric part from chunk_id (e.g., BZ_VR1_L3_266 -> 266)
        def extract_chunk_num(cid: str) -> int:
            try:
                parts = cid.split("_")
                return int(parts[-1]) if parts and parts[-1].isdigit() else 0
            except (ValueError, IndexError):
                return 0

        # Find ALL chunks from same document (NOT filtered by section_id)
        # This allows finding adjacent content even across section boundaries
        same_doc = [chunk for chunk in all_chunks if chunk.get("document_id") == document_id]

        # Sort by chunk_id numerically (L3_266 < L3_267 < L3_268)
        same_doc.sort(key=lambda x: extract_chunk_num(x.get("chunk_id", "")))

        # Find target position
        target_num = extract_chunk_num(chunk_id)
        target_position = None
        for pos, chunk in enumerate(same_doc):
            if extract_chunk_num(chunk.get("chunk_id", "")) == target_num:
                target_position = pos
                break

        if target_position is None:
            logger.warning(f"expand_adjacent: target chunk {chunk_id} not found in document")
            return []

        # Get context_window from config (default: 2)
        context_window = k

        # Extract context chunks
        start_pos = max(0, target_position - context_window)
        end_pos = min(len(same_doc), target_position + context_window + 1)

        context_chunks = []

        # Before chunks
        for i in range(start_pos, target_position):
            chunk = format_chunk_result(same_doc[i])
            chunk["position"] = "before"
            context_chunks.append(chunk)

        # After chunks
        for i in range(target_position + 1, end_pos):
            chunk = format_chunk_result(same_doc[i])
            chunk["position"] = "after"
            context_chunks.append(chunk)

        logger.debug(
            f"expand_adjacent: found {len(context_chunks)} adjacent chunks "
            f"for {chunk_id} (window={context_window})"
        )

        return context_chunks

    def _expand_by_section(self, target_chunk: Dict, all_chunks: List[Dict], k: int) -> List[Dict]:
        """Expand by finding neighboring chunks from same section."""
        try:
            document_id = target_chunk.get("document_id")
            section_id = target_chunk.get("section_id")
            chunk_id = target_chunk.get("chunk_id")

            if not document_id or not section_id:
                return []

            # Find chunks from same section
            same_section = [
                chunk
                for chunk in all_chunks
                if isinstance(chunk, dict)
                and chunk.get("document_id") == document_id
                and chunk.get("section_id") == section_id
                and chunk.get("chunk_id") != chunk_id
            ]

            # Sort by chunk_id
            same_section.sort(key=lambda x: x.get("chunk_id", ""))

            return [format_chunk_result(chunk) for chunk in same_section[:k]]

        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"Section-based expansion failed: {e}", exc_info=True)
            return []

    def _expand_by_similarity(self, target_chunk: Dict, k: int) -> List[Dict]:
        """Expand by finding semantically similar chunks."""
        # Validate embedder
        if (
            not self.embedder
            or not hasattr(self.embedder, "dimensions")
            or not self.embedder.dimensions
        ):
            raise ValueError("Embedder not available for similarity-based expansion")

        content = target_chunk.get("content", "")
        if not content:
            return []

        try:
            # Embed and search
            query_embedding = self.embedder.embed_texts([content])

            results = self.vector_store.hierarchical_search(
                query_text=content,
                query_embedding=query_embedding,
                k_layer3=k + 1,
            )

            chunks = results.get("layer3", [])

            # Filter out target chunk
            target_id = target_chunk.get("chunk_id")
            similar_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") != target_id]

            return [format_chunk_result(chunk) for chunk in similar_chunks[:k]]

        except Exception as e:
            logger.error(f"Similarity expansion failed: {e}", exc_info=True)
            raise

    # =========================================================================
    # VL Mode — page-based expansion
    # =========================================================================

    def _execute_vl(
        self,
        page_ids: List[str],
        expansion_mode: str,
        k: int = 3,
    ) -> ToolResult:
        """
        VL mode: expand page context by returning adjacent pages.

        In VL mode, only 'adjacent' expansion is meaningful (pages ± k).
        Other modes (section, similarity, hybrid) fall back to adjacent.
        """
        if expansion_mode != "adjacent":
            logger.info(f"VL mode: '{expansion_mode}' not supported, falling back to 'adjacent'")

        try:
            from ...vl.page_store import PageStore

            expanded_results = []
            all_page_images = []

            for page_id in page_ids:
                # Parse page_id → (document_id, page_number)
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
                data={"expansions": expanded_results, "expansion_mode": "adjacent"},
                citations=citations,
                metadata={
                    "page_count": len(page_ids),
                    "expansion_mode": "adjacent",
                    "mode": "vl",
                    "page_images": all_page_images,
                },
            )

        except Exception as e:
            logger.error(f"VL expand context failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
