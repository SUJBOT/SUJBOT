"""
VL Retrieval Pipeline

Simple retrieval for Vision-Language architecture:
  query text → Jina embed_query() → PostgreSQL cosine search → page results

No HyDE/expansion fusion — Jina v4's task-specific LoRA adapters
(retrieval.query vs retrieval.passage) already handle the query-document
asymmetry that HyDE addresses for text-only embeddings.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .jina_client import JinaClient
from .page_store import PageStore

logger = logging.getLogger(__name__)


@dataclass
class VLPageResult:
    """Single page result from VL retrieval."""

    page_id: str
    document_id: str
    page_number: int
    score: float
    image_path: Optional[str] = None


class VLRetriever:
    """
    Vision-Language retrieval pipeline.

    Flow: query → Jina text embedding → PostgreSQL cosine search → VLPageResult list
    """

    def __init__(
        self,
        jina_client: JinaClient,
        vector_store,  # PostgresVectorStoreAdapter
        page_store: PageStore,
        default_k: int = 5,
    ):
        self.jina_client = jina_client
        self.vector_store = vector_store
        self.page_store = page_store
        self.default_k = default_k

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        document_filter: Optional[str] = None,
    ) -> List[VLPageResult]:
        """
        Search for relevant pages using VL embeddings.

        Args:
            query: Natural language query text
            k: Number of results (default: self.default_k)
            document_filter: Optional document_id filter

        Returns:
            List of VLPageResult sorted by relevance
        """
        k = k or self.default_k

        # 1. Embed query using Jina v4 (retrieval.query task)
        query_embedding = self.jina_client.embed_query(query)

        # 2. Search PostgreSQL vl_pages table
        raw_results = self.vector_store.search_vl_pages(
            query_embedding=query_embedding,
            k=k,
            document_filter=document_filter,
        )

        # 3. Convert to VLPageResult objects with image paths
        results = []
        for row in raw_results:
            page_id = row["page_id"]
            try:
                image_path = self.page_store.get_image_path(page_id)
            except Exception:
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

        logger.info(
            f"VL search: '{query[:50]}...' → {len(results)} pages "
            f"(top score: {results[0].score:.3f})" if results else
            f"VL search: '{query[:50]}...' → 0 pages"
        )

        return results
