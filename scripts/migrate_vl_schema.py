"""
Migrate VL schema — create vectors.vl_pages table.

Idempotent: uses IF NOT EXISTS for all DDL statements.

Usage:
    uv run python scripts/migrate_vl_schema.py
"""

import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv


async def migrate():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    print(f"Connecting to PostgreSQL...")
    conn = await asyncpg.connect(db_url)

    ddl = """
    CREATE TABLE IF NOT EXISTS vectors.vl_pages (
        id SERIAL PRIMARY KEY,
        page_id TEXT UNIQUE NOT NULL,
        document_id TEXT NOT NULL,
        page_number INTEGER NOT NULL,
        embedding vector(2048) NOT NULL,
        image_path TEXT,
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_vl_pages_document_id
        ON vectors.vl_pages(document_id);
    CREATE INDEX IF NOT EXISTS idx_vl_pages_page_number
        ON vectors.vl_pages(document_id, page_number);
    -- No HNSW index on embedding column:
    -- ~500 vectors with exact cosine scan is <50ms and gives 100% recall.
    -- pgvector 0.8.1 HNSW limit is 2000 dims; Jina v4 uses 2048-dim.
    """

    await conn.execute(ddl)
    print("Migration complete: vectors.vl_pages table created")

    # Verify
    count = await conn.fetchval("SELECT count(*) FROM vectors.vl_pages")
    print(f"Current row count: {count}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
