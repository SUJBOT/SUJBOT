"""
VL Retrieval Pipeline

Simple retrieval for Vision-Language architecture:
  query text -> embedder embed_query() -> PostgreSQL cosine search -> page results

Supports Jina v4 cloud or local Qwen3-VL-Embedding-8B (same interface).
No HyDE/expansion fusion -- cosine similarity with L2-normalized embeddings
provides sufficient accuracy for ~500 pages.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .jina_client import JinaClient
from .local_embedder import LocalVLEmbedder
from .page_store import PageStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLPageResult:
    """Single page result from VL retrieval (immutable)."""

    page_id: str
    document_id: str
    page_number: int
    score: float
    image_path: Optional[str] = None

    def __post_init__(self):
        if self.page_number < 1:
            raise ValueError(f"page_number must be >= 1, got {self.page_number}")
        # Clamp score to [0.0, 1.0] — cosine similarity with L2-normalized
        # embeddings can be slightly negative for dissimilar content
        raw_score = self.score
        clamped = max(0.0, min(1.0, raw_score))
        if abs(raw_score - clamped) > 0.01:
            logger.warning(
                "VLPageResult score %.4f clamped to %.4f for page %s (possible embedding issue)",
                raw_score, clamped, self.page_id,
            )
        object.__setattr__(self, "score", clamped)


class VLRetriever:
    """
    Vision-Language retrieval pipeline.

    Flow: query -> embedder text embedding -> PostgreSQL cosine search -> VLPageResult list

    Accepts JinaClient or LocalVLEmbedder (duck-typed, same interface).
    """

    def __init__(
        self,
        jina_client: Union[JinaClient, LocalVLEmbedder],
        vector_store,  # PostgresVectorStoreAdapter
        page_store: PageStore,
        default_k: int = 5,
        adaptive_config=None,
    ):
        from src.retrieval.adaptive_k import AdaptiveKConfig

        self.jina_client = jina_client
        self.vector_store = vector_store
        self.page_store = page_store
        self.default_k = default_k
        self.adaptive_config: AdaptiveKConfig = adaptive_config or AdaptiveKConfig(enabled=False)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        document_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[VLPageResult]:
        """
        Search for relevant pages using VL embeddings.

        Delegates to search_with_embedding() and discards the embedding.

        Args:
            query: Natural language query text
            k: Number of results (default: self.default_k)
            document_filter: Optional document_id filter
            category_filter: Optional category filter ('documentation' or 'legislation')

        Returns:
            List of VLPageResult sorted by relevance
        """
        results, _ = self.search_with_embedding(
            query=query, k=k, document_filter=document_filter, category_filter=category_filter
        )
        return results

    def _convert_and_filter(
        self, raw_results: List[dict], k: int, label: str
    ) -> List[VLPageResult]:
        """Convert raw DB results to VLPageResult and apply adaptive-k filtering."""
        results = []
        for row in raw_results:
            page_id = row["page_id"]
            try:
                image_path = self.page_store.get_image_path(page_id)
            except Exception as e:
                logger.warning(
                    f"Could not resolve image path for page {page_id}: {e}", exc_info=True
                )
                image_path = row.get("image_path")

            results.append(
                VLPageResult(
                    page_id=page_id,
                    document_id=row["document_id"],
                    page_number=row["page_number"],
                    score=row["score"],
                    image_path=image_path,
                )
            )

        if self.adaptive_config.enabled and results:
            from src.retrieval.adaptive_k import adaptive_k_filter, AdaptiveKConfig

            effective_config = AdaptiveKConfig(
                enabled=True,
                method=self.adaptive_config.method,
                fetch_k=self.adaptive_config.fetch_k,
                min_k=self.adaptive_config.min_k,
                max_k=min(self.adaptive_config.max_k, k),
                score_gap_threshold=self.adaptive_config.score_gap_threshold,
                min_samples_for_adaptive=self.adaptive_config.min_samples_for_adaptive,
            )
            scores = [r.score for r in results]
            adaptive_result = adaptive_k_filter(results, scores, effective_config)
            results = adaptive_result.items

            logger.info(
                "Adaptive-k (%s): %d -> %d results (threshold=%.3f, method=%s)",
                label,
                adaptive_result.original_count,
                adaptive_result.filtered_count,
                adaptive_result.threshold,
                adaptive_result.method_used,
            )
        elif results:
            logger.info(
                "VL %s: -> %d pages (top score: %.3f)",
                label,
                len(results),
                results[0].score,
            )
        else:
            logger.info("VL %s: -> 0 pages", label)

        return results

    def search_with_embedding(
        self,
        query: str,
        k: Optional[int] = None,
        document_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> Tuple[List[VLPageResult], np.ndarray]:
        """
        Search for relevant pages and return the query embedding.

        Identical to search() but also returns the query embedding,
        allowing callers to reuse it without a redundant embedder API call.

        Args:
            query: Natural language query text
            k: Number of results (default: self.default_k)
            document_filter: Optional document_id filter
            category_filter: Optional category filter

        Returns:
            Tuple of (results, query_embedding) where query_embedding is
            the L2-normalized embedding as np.ndarray.
        """
        k = k or self.default_k

        if self.adaptive_config.enabled:
            fetch_k = max(self.adaptive_config.fetch_k, k)
        else:
            fetch_k = k

        query_embedding = self.jina_client.embed_query(query)

        raw_results = self.vector_store.search_vl_pages(
            query_embedding=query_embedding,
            k=fetch_k,
            document_filter=document_filter,
            category_filter=category_filter,
        )

        results = self._convert_and_filter(raw_results, k, f"search '{query[:50]}...'")

        return results, query_embedding

    def search_by_image(
        self,
        image_base64: str,
        k: Optional[int] = None,
        document_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[VLPageResult]:
        """
        Search for relevant pages using an image as query.

        Same pipeline as search() but embeds an image instead of text.

        Args:
            image_base64: Base64-encoded image data
            k: Number of results (default: self.default_k)
            document_filter: Optional document_id filter
            category_filter: Optional category filter

        Returns:
            List of VLPageResult sorted by relevance
        """
        k = k or self.default_k

        if self.adaptive_config.enabled:
            fetch_k = max(self.adaptive_config.fetch_k, k)
        else:
            fetch_k = k

        query_embedding = self.jina_client.embed_image(image_base64)

        raw_results = self.vector_store.search_vl_pages(
            query_embedding=query_embedding,
            k=fetch_k,
            document_filter=document_filter,
            category_filter=category_filter,
        )

        return self._convert_and_filter(raw_results, k, "image search")
