"""
Rebuild graph communities and re-embed entities after deduplication.

Usage:
    DATABASE_URL="postgresql://postgres:sujbot_secure_password@localhost:5432/sujbot" \
    uv run python scripts/graph_rebuild_communities.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    from src.graph.storage import GraphStorageAdapter
    from src.graph.community_detector import CommunityDetector
    from src.graph.community_summarizer import CommunitySummarizer
    from src.graph.embedder import GraphEmbedder
    from src.agent.providers.factory import create_provider
    from src.graph.post_processor import rebuild_graph_communities

    graph_storage = GraphStorageAdapter(connection_string=db_url)
    community_detector = CommunityDetector()
    graph_embedder = GraphEmbedder()

    # Use Haiku for community summarization
    provider = create_provider("claude-haiku-4-5-20251001")
    community_summarizer = CommunitySummarizer(provider)

    logger.info("Starting graph community rebuild (no dedup, just embed + detect + summarize)...")

    await graph_storage._ensure_pool()

    stats = await rebuild_graph_communities(
        graph_storage=graph_storage,
        community_detector=community_detector,
        community_summarizer=community_summarizer,
        graph_embedder=graph_embedder,
        llm_provider=provider,
        enable_dedup=False,  # Skip dedup — already done
    )

    logger.info(f"Rebuild complete: {stats}")
    await graph_storage.close()


if __name__ == "__main__":
    asyncio.run(main())
