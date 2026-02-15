"""
VL Retrieval Pipeline

Simple retrieval for Vision-Language architecture:
  query text -> Jina embed_query() -> PostgreSQL cosine search -> page results

No HyDE/expansion fusion -- Jina v4's task-specific LoRA adapters
(retrieval.query vs retrieval.passage) already handle the query-document
asymmetry that HyDE addresses for text-only embeddings.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .jina_client import JinaClient
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
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")


class VLRetriever:
    """
    Vision-Language retrieval pipeline.

    Flow: query -> Jina text embedding -> PostgreSQL cosine search -> VLPageResult list
    """

    def __init__(
        self,
        jina_client: JinaClient,
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

        When adaptive retrieval is enabled, fetches a larger candidate pool
        and uses Otsu/GMM thresholding to find the natural score cutoff.

        Args:
            query: Natural language query text
            k: Number of results (default: self.default_k)
            document_filter: Optional document_id filter
            category_filter: Optional category filter ('documentation' or 'legislation')

        Returns:
            List of VLPageResult sorted by relevance
        """
        k = k or self.default_k

        # Determine how many candidates to fetch from DB
        if self.adaptive_config.enabled:
            fetch_k = max(self.adaptive_config.fetch_k, k)
        else:
            fetch_k = k

        # 1. Embed query using Jina v4 (retrieval.query task)
        query_embedding = self.jina_client.embed_query(query)

        # 2. Search PostgreSQL vl_pages table
        raw_results = self.vector_store.search_vl_pages(
            query_embedding=query_embedding,
            k=fetch_k,
            document_filter=document_filter,
            category_filter=category_filter,
        )

        # 3. Convert to VLPageResult objects with image paths
        results = []
        for row in raw_results:
            page_id = row["page_id"]
            try:
                image_path = self.page_store.get_image_path(page_id)
            except Exception as e:
                logger.warning(f"Could not resolve image path for page {page_id}: {e}", exc_info=True)
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

        # 4. Apply adaptive-k filtering
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
                "Adaptive-k: %d -> %d results (threshold=%.3f, method=%s)",
                adaptive_result.original_count,
                adaptive_result.filtered_count,
                adaptive_result.threshold,
                adaptive_result.method_used,
            )
        elif results:
            logger.info(
                "VL search: '%s...' -> %d pages (top score: %.3f)",
                query[:50],
                len(results),
                results[0].score,
            )
        else:
            logger.info("VL search: '%s...' -> 0 pages", query[:50])

        return results
