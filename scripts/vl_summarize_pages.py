"""
Backfill page summaries for existing VL pages.

Sends each unsummarized page image to the configured VL remote model
and stores the summary in the page's metadata JSONB.

Usage:
    uv run python scripts/vl_summarize_pages.py                    # all pages
    uv run python scripts/vl_summarize_pages.py --doc-id BZ_VR1    # single doc
    uv run python scripts/vl_summarize_pages.py --limit 5          # test run
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
from src.storage.postgres_adapter import PostgresVectorStoreAdapter
from src.vl.jina_client import JinaClient
from src.vl.page_store import PageStore
from src.vl.vl_indexing import VLIndexingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def async_main(args):
    """Async entry point — all DB operations share a single event loop."""
    config = get_config()
    vl_config = config.vl

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set in .env")
        sys.exit(1)

    # Create summary provider
    if args.use_local:
        vl_idx_cfg = config.vl_indexing
        if not vl_idx_cfg:
            logger.error("--use-local requires vl_indexing section in config.json")
            sys.exit(1)
        model = vl_idx_cfg.summarization.model
        logger.info(f"Summary model (local): {model}")
    elif args.model:
        model = args.model
        logger.info(f"Summary model (override): {model}")
    else:
        model = get_variant_model(args.variant)
        logger.info(f"Summary model: {model} (variant: {args.variant})")
    summary_provider = create_provider(model)

    # Create VL components
    dimensions = vl_config.dimensions if vl_config else 2048
    vector_store = PostgresVectorStoreAdapter(
        connection_string=db_url,
        dimensions=dimensions,
    )
    await vector_store.initialize()

    page_store = PageStore(
        store_dir=vl_config.page_store_dir if vl_config else "data/vl_pages",
        source_pdf_dir=vl_config.source_pdf_dir if vl_config else "data",
        dpi=vl_config.page_image_dpi if vl_config else 150,
        image_format=vl_config.page_image_format if vl_config else "png",
    )

    jina_client = JinaClient(
        model=vl_config.jina_model if vl_config else "jina-embeddings-v4",
        dimensions=dimensions,
    )

    pipeline = VLIndexingPipeline(
        jina_client=jina_client,
        vector_store=vector_store,
        page_store=page_store,
        summary_provider=summary_provider,
    )

    # If --limit is set, only process that many pages
    if args.limit:
        pages = vector_store.get_vl_pages_without_summary(args.doc_id)
        pages = pages[: args.limit]
        if not pages:
            logger.info("No unsummarized pages found.")
            return

        model_name = summary_provider.get_model_name()
        logger.info(f"Summarizing {len(pages)} pages (limit={args.limit})...")

        success = 0
        for i, page in enumerate(pages):
            page_id = page["page_id"]
            try:
                summary = pipeline._summarize_page(page_id)
                if summary:
                    vector_store.update_vl_page_metadata(
                        page_id,
                        {"page_summary": summary, "summary_model": model_name},
                    )
                    success += 1
                    logger.info(f"[{i + 1}/{len(pages)}] {page_id}: {summary[:80]}...")
                else:
                    logger.warning(f"[{i + 1}/{len(pages)}] {page_id}: empty summary")
            except Exception as e:
                logger.warning(f"[{i + 1}/{len(pages)}] {page_id}: failed — {e}")

        logger.info(f"Done: {success}/{len(pages)} pages summarized.")
    else:
        count = pipeline.summarize_existing_pages(document_id=args.doc_id)
        logger.info(f"Total pages summarized: {count}")


def main():
    parser = argparse.ArgumentParser(description="Backfill VL page summaries")
    parser.add_argument("--doc-id", type=str, default=None, help="Summarize only this document")
    parser.add_argument("--limit", type=int, default=None, help="Max pages to summarize (for testing)")
    parser.add_argument(
        "--variant", type=str, default="remote", help="Agent variant for summary model (default: remote)"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local model from config.json vl_indexing.summarization section",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides --variant and --use-local)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
