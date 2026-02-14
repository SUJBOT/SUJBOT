"""
Build knowledge graph from existing VL page images.

Extracts entities and relationships from page images using multimodal LLM,
then runs Leiden community detection and generates community summaries.

Usage:
    uv run python scripts/graph_rag_build.py                     # full build
    uv run python scripts/graph_rag_build.py --limit 10          # test run (10 pages)
    uv run python scripts/graph_rag_build.py --doc-id BZ_VR1     # single document
    uv run python scripts/graph_rag_build.py --skip-extraction   # re-run community detection + summarization
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
from src.storage.postgres_adapter import PostgresVectorStoreAdapter
from src.vl.page_store import PageStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def _log_phase(title: str) -> None:
    """Log a phase banner."""
    logger.info(SEPARATOR)
    logger.info(title)
    logger.info(SEPARATOR)


async def extract_entities(args, graph_storage, entity_extractor, page_store, vector_store):
    """Phase 1: Extract entities and relationships from page images."""
    _log_phase("PHASE 1: Entity Extraction")

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

    MAX_CONSECUTIVE_FAILURES = 3
    consecutive_failures = 0
    total_entities = 0
    total_relationships = 0
    n_pages = len(pages)

    for i, page in enumerate(pages, 1):
        page_id = page["page_id"]
        document_id = page["document_id"]
        progress = f"[{i}/{n_pages}]"

        try:
            result = entity_extractor.extract_from_page(page_id, page_store)
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            if entities:
                total_entities += graph_storage.add_entities(
                    entities, document_id, source_page_id=page_id
                )
            if relationships:
                total_relationships += graph_storage.add_relationships(
                    relationships, document_id, source_page_id=page_id
                )

            consecutive_failures = 0
            logger.info(
                f"{progress} {page_id}: "
                f"{len(entities)} entities, {len(relationships)} relationships"
            )

        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as e:
            logger.warning(f"{progress} {page_id}: failed — {e}", exc_info=True)
            consecutive_failures += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.error(
                f"Aborting extraction after {MAX_CONSECUTIVE_FAILURES} "
                f"consecutive failures (completed {i}/{n_pages})"
            )
            break

    logger.info(
        f"Extraction complete: {total_entities} entities, {total_relationships} relationships"
    )


async def build_communities(args, graph_storage, community_detector, community_summarizer):
    """Phase 2: Community detection and summarization."""
    _log_phase("PHASE 2: Community Detection")

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

    if community_summarizer:
        _log_phase("PHASE 3: Community Summarization")

        model_name = community_summarizer.provider.get_model_name()
        id_to_entity = {e["entity_id"]: e for e in entities}
        n_communities = len(communities)

        for i, comm in enumerate(communities, 1):
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

            fallback_title = f"Community L{comm['level']}#{i - 1}"
            try:
                summary = community_summarizer.summarize(comm_entities, comm_rels)
                if summary:
                    comm["summary"] = summary
                    comm["summary_model"] = model_name
                    comm["title"] = summary.split(".")[0].strip()[:100]
                    logger.info(
                        f"[{i}/{n_communities}] L{comm['level']}: "
                        f"{comm['title']} ({len(entity_ids)} entities)"
                    )
                else:
                    comm["title"] = fallback_title
                    logger.warning(f"[{i}/{n_communities}] Empty summary")
            except Exception as e:
                comm["title"] = fallback_title
                logger.warning(f"[{i}/{n_communities}] Summarization failed: {e}")

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

    # Create provider (--model overrides --variant)
    model = args.model or get_variant_model(args.variant)
    logger.info(f"Using model: {model}")
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
    graph_storage, entity_extractor, community_detector, community_summarizer = (
        create_graph_components(pool=vector_store.pool, provider=provider)
    )

    # Phase 1: Extract entities
    if not args.skip_extraction:
        await extract_entities(args, graph_storage, entity_extractor, page_store, vector_store)

    # Phase 2: Communities
    if not args.skip_communities:
        await build_communities(args, graph_storage, community_detector, community_summarizer)

    stats = graph_storage.get_graph_stats()
    _log_phase("GRAPH STATS")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Cleanup
    await vector_store.close()


def main():
    parser = argparse.ArgumentParser(description="Build knowledge graph from VL page images")
    parser.add_argument("--doc-id", type=str, default=None, help="Process only this document")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max pages to process (for testing)"
    )
    parser.add_argument(
        "--variant", type=str, default="remote", help="Agent variant for LLM (default: remote)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model to use (overrides --variant, default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip entity extraction (re-run communities only)",
    )
    parser.add_argument("--skip-communities", action="store_true", help="Skip community detection")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
