"""
Re-embed all VL pages using the local embedding model.

Reads existing pages from the database, loads their images from PageStore,
embeds them via LocalVLEmbedder, and updates embeddings in vectors.vl_pages.

Usage:
    uv run python scripts/reembed_vl_pages.py                     # all pages
    uv run python scripts/reembed_vl_pages.py --doc-id BZ_VR1     # single doc
    uv run python scripts/reembed_vl_pages.py --limit 5            # test run
    uv run python scripts/reembed_vl_pages.py --dry-run            # preview only
    uv run python scripts/reembed_vl_pages.py --embedder jina      # use Jina instead
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio

nest_asyncio.apply()

import numpy as np

from src.config import get_config
from src.storage.postgres_adapter import PostgresVectorStoreAdapter
from src.vl.page_store import PageStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_all_pages(pool, document_id=None, limit=None):
    """Fetch page records from vectors.vl_pages."""
    query = "SELECT page_id, document_id, page_number, image_path, metadata FROM vectors.vl_pages"
    params = []

    if document_id:
        query += " WHERE document_id = $1"
        params.append(document_id)

    query += " ORDER BY document_id, page_number"

    if limit:
        query += f" LIMIT ${len(params) + 1}"
        params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [dict(r) for r in rows]


async def update_embedding(pool, page_id: str, embedding: np.ndarray):
    """Update a single page's embedding in the database."""
    embedding_str = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE vectors.vl_pages SET embedding = $1::vector WHERE page_id = $2",
            embedding_str,
            page_id,
        )


async def async_main(args):
    """Async entry point."""
    config = get_config()
    vl_config = config.vl

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set in .env")
        sys.exit(1)

    # Override dev port to production if needed
    if ":5433/" in db_url and not args.use_dev_db:
        db_url = db_url.replace(":5433/", ":5432/")
        logger.info("Overriding DATABASE_URL to production port 5432")

    dimensions = vl_config.dimensions if vl_config else 2048

    # Initialize storage
    vector_store = PostgresVectorStoreAdapter(
        connection_string=db_url,
        dimensions=dimensions,
    )
    await vector_store.initialize()

    # Initialize page store
    page_store = PageStore(
        store_dir=vl_config.page_store_dir if vl_config else "data/vl_pages",
        source_pdf_dir=vl_config.source_pdf_dir if vl_config else "data",
        dpi=vl_config.page_image_dpi if vl_config else 150,
        image_format=vl_config.page_image_format if vl_config else "png",
    )

    # Create embedder based on choice
    if args.embedder == "local":
        from src.vl.local_embedder import LocalVLEmbedder

        embedder = LocalVLEmbedder(
            base_url=vl_config.local_embedding_url if vl_config else None,
            model=(
                vl_config.local_embedding_model
                if vl_config
                else "Qwen/Qwen3-VL-Embedding-8B"
            ),
            dimensions=dimensions,
        )
        logger.info("Using local embedder")
    else:
        from src.vl.jina_client import JinaClient

        embedder = JinaClient(
            model=vl_config.jina_model if vl_config else "jina-embeddings-v4",
            dimensions=dimensions,
        )
        logger.info("Using Jina embedder")

    # Fetch pages
    pages = await fetch_all_pages(
        vector_store.pool,
        document_id=args.doc_id,
        limit=args.limit,
    )
    logger.info("Found %d pages to re-embed", len(pages))

    if not pages:
        logger.info("No pages found, exiting.")
        return

    if args.dry_run:
        logger.info("DRY RUN — would re-embed %d pages. Exiting.", len(pages))
        for p in pages[:10]:
            logger.info("  %s (doc=%s, page=%d)", p["page_id"], p["document_id"], p["page_number"])
        if len(pages) > 10:
            logger.info("  ... and %d more", len(pages) - 10)
        return

    # Re-embed in batches
    batch_size = args.batch_size
    total = len(pages)
    success = 0
    errors = 0
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch = pages[batch_start : batch_start + batch_size]
        page_images = []
        valid_pages = []

        for page in batch:
            try:
                img_bytes = page_store.get_image_bytes(page["page_id"])
                page_images.append(img_bytes)
                valid_pages.append(page)
            except Exception as e:
                logger.warning("Cannot read image for %s: %s", page["page_id"], e)
                errors += 1
                continue

        if not page_images:
            continue

        try:
            embeddings = embedder.embed_pages(page_images)
        except Exception as e:
            logger.error("Embedding failed for batch %d-%d: %s", batch_start, batch_start + len(batch), e)
            errors += len(valid_pages)
            continue

        # Update each page's embedding
        for i, page in enumerate(valid_pages):
            try:
                await update_embedding(vector_store.pool, page["page_id"], embeddings[i])
                success += 1
            except Exception as e:
                logger.error("DB update failed for %s: %s", page["page_id"], e)
                errors += 1

        elapsed = time.time() - start_time
        done = batch_start + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        logger.info(
            "Progress: %d/%d pages (%.1f pages/s, %d errors)",
            done, total, rate, errors,
        )

    elapsed = time.time() - start_time
    logger.info(
        "Re-embedding complete: %d success, %d errors, %.1fs elapsed",
        success, errors, elapsed,
    )

    embedder.close()


def main():
    parser = argparse.ArgumentParser(description="Re-embed VL pages with local embedding model")
    parser.add_argument("--doc-id", type=str, help="Only re-embed pages for this document")
    parser.add_argument("--limit", type=int, help="Max pages to process")
    parser.add_argument("--batch-size", type=int, default=4, help="Embedding batch size")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify DB")
    parser.add_argument("--use-dev-db", action="store_true", help="Use dev DB port 5433")
    parser.add_argument(
        "--embedder",
        choices=["local", "jina"],
        default="local",
        help="Embedder to use (default: local)",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
