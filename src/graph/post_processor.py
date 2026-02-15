"""
Graph Post-Processor — rebuild communities and deduplicate entities after indexing.

Orchestrates the full post-indexing pipeline:
  exact dedup → embed entities → semantic dedup → re-embed →
  community detection → community summarization → save → embed communities
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .storage import GraphStorageAdapter, _vec_to_pg

if TYPE_CHECKING:
    from .community_detector import CommunityDetector
    from .community_summarizer import CommunitySummarizer
    from .embedder import GraphEmbedder

logger = logging.getLogger(__name__)


async def rebuild_graph_communities(
    graph_storage: GraphStorageAdapter,
    community_detector: "CommunityDetector",
    community_summarizer: Optional["CommunitySummarizer"] = None,
    graph_embedder: Optional["GraphEmbedder"] = None,
    llm_provider: Optional[Any] = None,
    document_id: Optional[str] = None,
    enable_dedup: bool = True,
    semantic_threshold: float = 0.85,
) -> Dict:
    """
    Full post-indexing pipeline: dedup, community detection, summarization, embedding.

    Args:
        graph_storage: Graph storage adapter
        community_detector: Leiden community detector
        community_summarizer: Optional LLM summarizer for communities
        graph_embedder: Optional embedder for entity/community search
        llm_provider: Optional LLM provider for semantic dedup arbitration
        document_id: For logging context (which document triggered this)
        enable_dedup: Whether to run entity deduplication
        semantic_threshold: Cosine similarity threshold for semantic dedup

    Returns:
        Dict with pipeline stats
    """
    log_ctx = f" (triggered by {document_id})" if document_id else ""
    logger.info(f"Graph rebuild started{log_ctx}")
    stats: Dict[str, Any] = {}

    # Phase 1: Exact dedup
    if enable_dedup:
        try:
            exact_stats = await graph_storage.async_deduplicate_exact()
            stats["exact_dedup"] = exact_stats
        except Exception as e:
            logger.error(f"Exact dedup failed, continuing: {e}", exc_info=True)
            stats["exact_dedup"] = {"error": str(e)}

    # Phase 2: Embed entities without embeddings
    if graph_embedder:
        try:
            embed_count = await _embed_new_entities(graph_storage, graph_embedder)
            stats["entities_embedded"] = embed_count
        except Exception as e:
            logger.error(f"Entity embedding failed, continuing: {e}", exc_info=True)
            stats["entities_embedded"] = {"error": str(e)}

    # Phase 3: Semantic dedup (requires embeddings + LLM)
    if enable_dedup and graph_embedder and llm_provider:
        try:
            sem_stats = await graph_storage.async_deduplicate_semantic(
                similarity_threshold=semantic_threshold,
                llm_provider=llm_provider,
            )
            stats["semantic_dedup"] = sem_stats
        except Exception as e:
            logger.error(f"Semantic dedup failed, continuing: {e}", exc_info=True)
            stats["semantic_dedup"] = {"error": str(e)}

        # Phase 4: Re-embed merged entities (canonical got embedding cleared)
        if graph_embedder:
            try:
                re_embed_count = await _embed_new_entities(graph_storage, graph_embedder)
                stats["entities_re_embedded"] = re_embed_count
            except Exception as e:
                logger.error(f"Re-embedding failed, continuing: {e}", exc_info=True)

    # Phase 5: Community detection
    try:
        entities, relationships = await graph_storage.async_get_all()
        stats["total_entities"] = len(entities)
        stats["total_relationships"] = len(relationships)
    except Exception as e:
        logger.error(f"Failed to fetch graph data for community detection: {e}", exc_info=True)
        stats["error"] = str(e)
        return stats

    if len(entities) < 3:
        logger.info(f"Too few entities ({len(entities)}) for community detection, skipping")
        stats["communities_detected"] = 0
        return stats

    try:
        communities = await asyncio.to_thread(
            community_detector.detect,
            entities=entities,
            relationships=relationships,
            max_levels=3,
        )
        stats["communities_detected"] = len(communities)
    except Exception as e:
        logger.error(f"Community detection failed: {e}", exc_info=True)
        stats["communities_detected"] = {"error": str(e)}
        return stats

    if not communities:
        logger.info("No communities detected")
        return stats

    # Phase 6: Community summarization
    if community_summarizer:
        id_to_entity = {e["entity_id"]: e for e in entities}
        model_name = None
        try:
            model_name = community_summarizer.provider.get_model_name()
        except Exception:
            pass

        summarized = 0
        for comm in communities:
            entity_ids = comm["entity_ids"]
            comm_entities = [id_to_entity[eid] for eid in entity_ids if eid in id_to_entity]
            comm_entity_set = set(entity_ids)
            comm_rels = [
                {
                    "source": id_to_entity.get(r["source_entity_id"], {}).get("name", "?"),
                    "target": id_to_entity.get(r["target_entity_id"], {}).get("name", "?"),
                    "type": r["relationship_type"],
                }
                for r in relationships
                if r["source_entity_id"] in comm_entity_set
                and r["target_entity_id"] in comm_entity_set
            ]

            try:
                summary = await asyncio.to_thread(
                    community_summarizer.summarize, comm_entities, comm_rels
                )
                if summary:
                    comm["summary"] = summary
                    comm["summary_model"] = model_name
                    comm["title"] = summary.split(".")[0].strip()[:100]
                    summarized += 1
                else:
                    comm["title"] = f"Community L{comm['level']}"
            except Exception as e:
                logger.warning(f"Community summarization failed: {e}")
                comm["title"] = f"Community L{comm['level']}"

        stats["communities_summarized"] = summarized

    # Phase 7: Save communities (atomic DELETE+INSERT)
    try:
        saved = await graph_storage.async_save_communities(communities)
        stats["communities_saved"] = saved
        logger.info(f"Saved {saved} communities")
    except Exception as e:
        logger.error(f"Failed to save communities: {e}", exc_info=True)
        stats["communities_saved"] = {"error": str(e)}
        return stats

    # Phase 8: Embed communities
    if graph_embedder:
        try:
            comm_embed_count = await _embed_new_communities(graph_storage, graph_embedder)
            stats["communities_embedded"] = comm_embed_count
        except Exception as e:
            logger.error(f"Community embedding failed: {e}", exc_info=True)
            stats["communities_embedded"] = {"error": str(e)}

    logger.info(f"Graph rebuild complete{log_ctx}: {stats}")
    return stats


async def _embed_new_entities(
    graph_storage: GraphStorageAdapter,
    graph_embedder: "GraphEmbedder",
    batch_size: int = 256,
) -> int:
    """Embed entities that have NULL search_embedding."""
    await graph_storage._ensure_pool()

    async with graph_storage.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT entity_id, name, entity_type, coalesce(description, '') AS description "
            "FROM graph.entities WHERE search_embedding IS NULL ORDER BY entity_id"
        )

    if not rows:
        logger.info("No entities need embedding")
        return 0

    logger.info(f"Embedding {len(rows)} entities")

    # Build text representations
    texts = [f"{r['name']} ({r['entity_type']}): {r['description']}" for r in rows]
    entity_ids = [r["entity_id"] for r in rows]

    # Batch encode
    embeddings = await asyncio.to_thread(graph_embedder.encode_passages, texts, batch_size)

    # Store embeddings in batches
    total_stored = 0
    for i in range(0, len(entity_ids), batch_size):
        batch_ids = entity_ids[i : i + batch_size]
        batch_vecs = embeddings[i : i + batch_size]

        records = [(eid, _vec_to_pg(vec)) for eid, vec in zip(batch_ids, batch_vecs)]

        async with graph_storage.pool.acquire() as conn:
            await conn.executemany(
                "UPDATE graph.entities SET search_embedding = $2::vector WHERE entity_id = $1",
                records,
            )
        total_stored += len(records)

    logger.info(f"Embedded {total_stored} entities")
    return total_stored


async def _embed_new_communities(
    graph_storage: GraphStorageAdapter,
    graph_embedder: "GraphEmbedder",
    batch_size: int = 256,
) -> int:
    """Embed communities that have NULL search_embedding."""
    await graph_storage._ensure_pool()

    async with graph_storage.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT community_id, coalesce(title, '') AS title, coalesce(summary, '') AS summary "
            "FROM graph.communities WHERE search_embedding IS NULL ORDER BY community_id"
        )

    if not rows:
        logger.info("No communities need embedding")
        return 0

    logger.info(f"Embedding {len(rows)} communities")

    texts = [f"{r['title']}: {r['summary']}" for r in rows]
    community_ids = [r["community_id"] for r in rows]

    embeddings = await asyncio.to_thread(graph_embedder.encode_passages, texts, batch_size)

    total_stored = 0
    for i in range(0, len(community_ids), batch_size):
        batch_ids = community_ids[i : i + batch_size]
        batch_vecs = embeddings[i : i + batch_size]

        records = [(cid, _vec_to_pg(vec)) for cid, vec in zip(batch_ids, batch_vecs)]

        async with graph_storage.pool.acquire() as conn:
            await conn.executemany(
                "UPDATE graph.communities SET search_embedding = $2::vector WHERE community_id = $1",
                records,
            )
        total_stored += len(records)

    logger.info(f"Embedded {total_stored} communities")
    return total_stored
