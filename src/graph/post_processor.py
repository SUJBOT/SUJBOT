"""
Graph Post-Processor — rebuild communities and deduplicate entities after indexing.

Orchestrates the full post-indexing pipeline:
  exact dedup → embed entities → semantic dedup → re-embed →
  community detection → community summarization → save → embed communities
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..exceptions import GraphStoreError, ProviderError
from ..utils.async_helpers import vec_to_pgvector
from .storage import GraphStorageAdapter

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
    dedup_provider: Optional[Any] = None,
    page_store: Optional[Any] = None,
    document_id: Optional[str] = None,
    enable_dedup: bool = True,
    semantic_threshold: float = 0.85,
    llm_threshold: float = 0.75,
    auto_merge_threshold: float = 0.95,
    max_images_per_entity: int = 2,
) -> Dict:
    """
    Full post-indexing pipeline: dedup, community detection, summarization, embedding.

    Args:
        graph_storage: Graph storage adapter
        community_detector: Leiden community detector
        community_summarizer: Optional LLM summarizer for communities
        graph_embedder: Optional embedder for entity/community search
        llm_provider: Optional LLM provider for semantic dedup (fallback)
        dedup_provider: Optional separate LLM provider for dedup (preferred over llm_provider)
        page_store: Optional PageStore for multimodal dedup context
        document_id: For logging context (which document triggered this)
        enable_dedup: Whether to run entity deduplication
        semantic_threshold: Legacy threshold (used as llm_threshold if thresholds not specified)
        llm_threshold: Below this similarity, skip dedup entirely
        auto_merge_threshold: Above this similarity, auto-merge without LLM
        max_images_per_entity: Max page images per entity in dedup context

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
        except (GraphStoreError, ProviderError) as e:
            logger.error(f"Exact dedup failed, continuing: {e}", exc_info=True)
            stats["exact_dedup"] = {"error": str(e)}

    # Phase 2: Embed entities without embeddings
    if graph_embedder:
        try:
            embed_count = await _embed_new_entities(graph_storage, graph_embedder)
            stats["entities_embedded"] = embed_count
        except (GraphStoreError, ProviderError) as e:
            logger.error(f"Entity embedding failed, continuing: {e}", exc_info=True)
            stats["entities_embedded"] = {"error": str(e)}

    # Phase 3: Semantic dedup (requires embeddings; LLM optional for middle-zone)
    dedup_llm = dedup_provider or llm_provider
    if enable_dedup and graph_embedder:
        try:
            sem_stats = await graph_storage.async_deduplicate_semantic(
                llm_threshold=llm_threshold,
                auto_merge_threshold=auto_merge_threshold,
                llm_provider=dedup_llm,
                page_store=page_store,
                max_images_per_entity=max_images_per_entity,
            )
            stats["semantic_dedup"] = sem_stats
        except (GraphStoreError, ProviderError) as e:
            logger.error(f"Semantic dedup failed, continuing: {e}", exc_info=True)
            stats["semantic_dedup"] = {"error": str(e)}

        # Phase 4: Re-embed merged entities (canonical got embedding cleared)
        try:
            re_embed_count = await _embed_new_entities(graph_storage, graph_embedder)
            stats["entities_re_embedded"] = re_embed_count
        except (GraphStoreError, ProviderError) as e:
            logger.error(f"Re-embedding failed, continuing: {e}", exc_info=True)
            stats["entities_re_embedded"] = {"error": str(e)}

    # Phase 5: Community detection
    try:
        entities, relationships = await graph_storage.async_get_all()
        stats["total_entities"] = len(entities)
        stats["total_relationships"] = len(relationships)
    except GraphStoreError as e:
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
        except Exception as e:
            logger.debug(f"Could not get summarizer model name: {e}")

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
                result = await asyncio.to_thread(
                    community_summarizer.summarize, comm_entities, comm_rels
                )
                if result:
                    title, description = result
                    comm["title"] = title
                    comm["summary"] = description
                    comm["summary_model"] = model_name
                    summarized += 1
                else:
                    comm["title"] = f"Community L{comm['level']}"
            except Exception as e:
                logger.warning(f"Community summarization failed: {e}", exc_info=True)
                comm["title"] = f"Community L{comm['level']}"

        stats["communities_summarized"] = summarized

    # Phase 7: Save communities (atomic DELETE+INSERT)
    try:
        saved = await graph_storage.async_save_communities(communities)
        stats["communities_saved"] = saved
        logger.info(f"Saved {saved} communities")
    except GraphStoreError as e:
        logger.error(f"Failed to save communities: {e}", exc_info=True)
        stats["communities_saved"] = {"error": str(e)}
        return stats

    # Phase 8: Embed communities
    if graph_embedder:
        try:
            comm_embed_count = await _embed_new_communities(graph_storage, graph_embedder)
            stats["communities_embedded"] = comm_embed_count
        except (GraphStoreError, ProviderError) as e:
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
    return await _embed_rows(
        graph_storage=graph_storage,
        graph_embedder=graph_embedder,
        fetch_sql=(
            "SELECT entity_id, name, entity_type, coalesce(description, '') AS description "
            "FROM graph.entities WHERE search_embedding IS NULL ORDER BY entity_id"
        ),
        id_column="entity_id",
        text_fn=lambda r: f"{r['name']} ({r['entity_type']}): {r['description']}",
        update_sql="UPDATE graph.entities SET search_embedding = $2::vector WHERE entity_id = $1",
        label="entities",
        batch_size=batch_size,
    )


async def _embed_new_communities(
    graph_storage: GraphStorageAdapter,
    graph_embedder: "GraphEmbedder",
    batch_size: int = 256,
) -> int:
    """Embed communities that have NULL search_embedding."""
    return await _embed_rows(
        graph_storage=graph_storage,
        graph_embedder=graph_embedder,
        fetch_sql=(
            "SELECT community_id, coalesce(title, '') AS title, coalesce(summary, '') AS summary "
            "FROM graph.communities WHERE search_embedding IS NULL ORDER BY community_id"
        ),
        id_column="community_id",
        text_fn=lambda r: f"{r['title']}: {r['summary']}",
        update_sql="UPDATE graph.communities SET search_embedding = $2::vector WHERE community_id = $1",
        label="communities",
        batch_size=batch_size,
    )


async def _embed_rows(
    graph_storage: GraphStorageAdapter,
    graph_embedder: "GraphEmbedder",
    fetch_sql: str,
    id_column: str,
    text_fn,
    update_sql: str,
    label: str,
    batch_size: int = 256,
) -> int:
    """Fetch rows with NULL embeddings, encode them, and store the vectors.

    Shared implementation for entity and community embedding.
    """
    await graph_storage._ensure_pool()

    async with graph_storage.pool.acquire() as conn:
        rows = await conn.fetch(fetch_sql)

    if not rows:
        logger.info(f"No {label} need embedding")
        return 0

    logger.info(f"Embedding {len(rows)} {label}")

    texts = [text_fn(r) for r in rows]
    row_ids = [r[id_column] for r in rows]

    embeddings = await asyncio.to_thread(graph_embedder.encode_passages, texts, batch_size)

    total_stored = 0
    for i in range(0, len(row_ids), batch_size):
        batch_ids = row_ids[i : i + batch_size]
        batch_vecs = embeddings[i : i + batch_size]

        records = [(rid, vec_to_pgvector(vec)) for rid, vec in zip(batch_ids, batch_vecs)]

        async with graph_storage.pool.acquire() as conn:
            await conn.executemany(update_sql, records)
        total_stored += len(records)

    logger.info(f"Embedded {total_stored} {label}")
    return total_stored
