"""
Graph RAG Module

Knowledge graph layer for cross-document entity/relationship reasoning.
Extracts entities from page images, builds relationship graph,
detects communities via Leiden algorithm, generates summaries.
"""

import logging
from typing import Any, Optional, Tuple

from .community_detector import CommunityDetector
from .community_summarizer import CommunitySummarizer
from .embedder import GraphEmbedder
from .entity_extractor import EntityExtractor
from .post_processor import rebuild_graph_communities
from .storage import GraphStorageAdapter

logger = logging.getLogger(__name__)

__all__ = [
    "GraphStorageAdapter",
    "GraphEmbedder",
    "EntityExtractor",
    "CommunityDetector",
    "CommunitySummarizer",
    "rebuild_graph_communities",
    "create_graph_components",
]


def create_graph_components(
    pool: Any,
    provider: Optional[Any] = None,
    embedder: Optional[GraphEmbedder] = None,
) -> Tuple[GraphStorageAdapter, Optional[EntityExtractor], CommunityDetector, Optional[CommunitySummarizer]]:
    """
    Factory for graph components (follows create_vl_components pattern).

    Args:
        pool: asyncpg connection pool (shared with vector store)
        provider: LLM provider for extraction/summarization (optional)
        embedder: Optional GraphEmbedder for semantic search (optional)

    Returns:
        Tuple of (storage, extractor, detector, summarizer)
    """
    storage = GraphStorageAdapter(pool=pool, embedder=embedder)
    extractor = EntityExtractor(provider) if provider else None
    detector = CommunityDetector()
    summarizer = CommunitySummarizer(provider) if provider else None

    logger.info(
        "Graph components initialized: extractor=%s, summarizer=%s",
        "yes" if extractor else "no",
        "yes" if summarizer else "no",
    )
    return storage, extractor, detector, summarizer
