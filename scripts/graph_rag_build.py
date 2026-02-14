"""
Build knowledge graph from existing VL page images.

Extracts entities and relationships from page images using multimodal LLM,
then runs Leiden community detection and generates community summaries.

Usage:
    uv run python scripts/graph_rag_build.py                     # full build
    uv run python scripts/graph_rag_build.py --limit 10          # test run (10 pages)
    uv run python scripts/graph_rag_build.py --doc-id BZ_VR1     # single document
    uv run python scripts/graph_rag_build.py --skip-extraction   # communities only (re-run Leiden)
    uv run python scripts/graph_rag_build.py --skip-communities  # extraction only
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio

nest_asyncio.apply()

from backend.constants import get_variant_model
from src.agent.providers.factory import create_provider
from src.config import get_config
from src.graph import create_graph_components
from src.graph.storage import GraphStorageAdapter
from src.storage.postgres_adapter import PostgresVectorStoreAdapter
from src.vl.page_store import PageStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def extract_entities(args, graph_storage, entity_extractor, page_store, vector_store):
    """Phase 1: Extract entities and relationships from page images."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Entity Extraction")
    logger.info("=" * 60)

    # Get pages to process
    await vector_store._ensure_pool()
    async with vector_store.pool.acquire() as conn:
        if args.doc_id:
            rows = await conn.fetch(
                "SELECT page_id, document_id FROM vectors.vl_pages "
                "WHERE document_id = $1 ORDER BY page_number",
                args.doc_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT page_id, document_id FROM vectors.vl_pages ORDER BY document_id, page_number"
            )

    pages = [{"page_id": r["page_id"], "document_id": r["document_id"]} for r in rows]

    if args.limit:
        pages = pages[: args.limit]

    if not pages:
        logger.info("No pages to process.")
        return

    logger.info(f"Processing {len(pages)} pages...")

    max_consecutive_failures = 3
    consecutive_failures = 0
    total_entities = 0
    total_relationships = 0

    for i, page in enumerate(pages):
        page_id = page["page_id"]
        document_id = page["document_id"]

        try:
            result = entity_extractor.extract_from_page(page_id, page_store)
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            if entities:
                n_ent = graph_storage.add_entities(entities, document_id, source_page_id=page_id)
                total_entities += n_ent
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if relationships:
                n_rel = graph_storage.add_relationships(relationships, document_id, source_page_id=page_id)
                total_relationships += n_rel

            logger.info(
                f"[{i + 1}/{len(pages)}] {page_id}: "
                f"{len(entities)} entities, {len(relationships)} relationships"
            )

        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as e:
            logger.warning(f"[{i + 1}/{len(pages)}] {page_id}: failed — {e}")
            consecutive_failures += 1

        if consecutive_failures >= max_consecutive_failures:
            logger.error(
                f"Aborting extraction after {max_consecutive_failures} "
                f"consecutive failures (completed {i + 1}/{len(pages)})"
            )
            break

    logger.info(f"Extraction complete: {total_entities} entities, {total_relationships} relationships")


async def build_communities(args, graph_storage, community_detector, community_summarizer):
    """Phase 2: Community detection and summarization."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Community Detection")
    logger.info("=" * 60)

    entities, relationships = graph_storage.get_all_entities_and_relationships()
    logger.info(f"Graph: {len(entities)} entities, {len(relationships)} relationships")

    if len(entities) < 3:
        logger.warning("Too few entities for community detection (need >= 3)")
        return

    communities = community_detector.detect(
        entities=entities,
        relationships=relationships,
        max_levels=3,
    )

    if not communities:
        logger.warning("No communities detected")
        return

    logger.info(f"Detected {len(communities)} communities")

    # Summarize communities
    if community_summarizer:
        logger.info("=" * 60)
        logger.info("PHASE 3: Community Summarization")
        logger.info("=" * 60)

        model_name = community_summarizer.provider.get_model_name()

        for i, comm in enumerate(communities):
            entity_ids = comm["entity_ids"]

            # Get entity details for this community
            id_to_entity = {e["entity_id"]: e for e in entities}
            comm_entities = [id_to_entity[eid] for eid in entity_ids if eid in id_to_entity]

            # Get relationships within community
            comm_entity_set = set(entity_ids)
            comm_rels = [
                {
                    "source": id_to_entity.get(r["source_entity_id"], {}).get("name", "?"),
                    "target": id_to_entity.get(r["target_entity_id"], {}).get("name", "?"),
                    "type": r["relationship_type"],
                }
                for r in relationships
                if r["source_entity_id"] in comm_entity_set and r["target_entity_id"] in comm_entity_set
            ]

            try:
                summary = community_summarizer.summarize(comm_entities, comm_rels)
                if summary:
                    comm["summary"] = summary
                    comm["summary_model"] = model_name
                    # Generate title from first sentence
                    comm["title"] = summary.split(".")[0].strip()[:100]
                    logger.info(
                        f"[{i + 1}/{len(communities)}] L{comm['level']}: "
                        f"{comm['title']} ({len(entity_ids)} entities)"
                    )
                else:
                    comm["title"] = f"Community L{comm['level']}#{i}"
                    logger.warning(f"[{i + 1}/{len(communities)}] Empty summary")
            except Exception as e:
                comm["title"] = f"Community L{comm['level']}#{i}"
                logger.warning(f"[{i + 1}/{len(communities)}] Summarization failed: {e}")

    # Save communities
    saved = graph_storage.save_communities(communities)
    logger.info(f"Saved {saved} communities")


async def async_main(args):
    """Main async entry point."""
    config = get_config()
    vl_config = config.vl

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set in .env")
        sys.exit(1)

    # Create provider
    model = get_variant_model(args.variant)
    logger.info(f"Using model: {model} (variant: {args.variant})")
    provider = create_provider(model)

    # Create vector store (to read pages)
    dimensions = vl_config.dimensions if vl_config else 2048
    vector_store = PostgresVectorStoreAdapter(
        connection_string=db_url,
        dimensions=dimensions,
        architecture="vl",
    )
    await vector_store.initialize()

    # Create page store
    page_store = PageStore(
        store_dir=vl_config.page_store_dir if vl_config else "data/vl_pages",
        source_pdf_dir=vl_config.source_pdf_dir if vl_config else "data",
        dpi=vl_config.page_image_dpi if vl_config else 150,
        image_format=vl_config.page_image_format if vl_config else "png",
    )

    # Create graph components (shares vector_store pool)
    graph_storage, entity_extractor, community_detector, community_summarizer = \
        create_graph_components(pool=vector_store.pool, provider=provider)

    # Phase 1: Extract entities
    if not args.skip_extraction:
        await extract_entities(args, graph_storage, entity_extractor, page_store, vector_store)

    # Phase 2: Communities
    if not args.skip_communities:
        await build_communities(args, graph_storage, community_detector, community_summarizer)

    # Print stats
    stats = graph_storage.get_graph_stats()
    logger.info("=" * 60)
    logger.info("GRAPH STATS")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph from VL page images")
    parser.add_argument("--doc-id", type=str, default=None, help="Process only this document")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to process (for testing)")
    parser.add_argument(
        "--variant", type=str, default="remote", help="Agent variant for LLM (default: remote)"
    )
    parser.add_argument("--skip-extraction", action="store_true", help="Skip entity extraction (re-run communities only)")
    parser.add_argument("--skip-communities", action="store_true", help="Skip community detection")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
