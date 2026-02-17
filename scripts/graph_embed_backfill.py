#!/usr/bin/env python3
"""
Backfill search_embedding (multilingual-e5-small, 384-dim) for graph tables.

Embeds entities, relationships, and communities, then stores vectors in PostgreSQL.
Creates HNSW indexes after embedding. Safe to run multiple times — only processes
rows where search_embedding IS NULL.

Usage:
    uv run python scripts/graph_embed_backfill.py
"""

import asyncio
import os
import sys
import time

import asyncpg
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.async_helpers import vec_to_pgvector as _vec_to_pg  # noqa: E402

load_dotenv()

BATCH_SIZE = 256


async def add_columns(conn: asyncpg.Connection):
    """Add search_embedding columns if they don't exist."""
    for table in ("entities", "relationships", "communities"):
        await conn.execute(f"""
            DO $$ BEGIN
                ALTER TABLE graph.{table} ADD COLUMN search_embedding vector(384);
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
    print("Columns ready.")


async def backfill_entities(conn: asyncpg.Connection, embedder):
    rows = await conn.fetch(
        "SELECT entity_id, name, coalesce(description, '') AS description "
        "FROM graph.entities WHERE search_embedding IS NULL "
        "ORDER BY entity_id"
    )
    if not rows:
        print("Entities: all already embedded.")
        return

    print(f"Entities: embedding {len(rows)} rows...")
    texts = [f"{r['name']} {r['description']}" for r in rows]

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_rows = rows[i : i + BATCH_SIZE]
        embeddings = embedder.encode_passages(batch_texts)

        records = [
            (r["entity_id"], _vec_to_pg(embeddings[j]))
            for j, r in enumerate(batch_rows)
        ]
        await conn.executemany(
            "UPDATE graph.entities SET search_embedding = $2::vector WHERE entity_id = $1",
            records,
        )
        print(f"  Entities: {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")


async def backfill_relationships(conn: asyncpg.Connection, embedder):
    rows = await conn.fetch("""
        SELECT r.relationship_id, r.relationship_type,
               coalesce(r.description, '') AS description,
               s.name AS source_name, t.name AS target_name
        FROM graph.relationships r
        JOIN graph.entities s ON r.source_entity_id = s.entity_id
        JOIN graph.entities t ON r.target_entity_id = t.entity_id
        WHERE r.search_embedding IS NULL
        ORDER BY r.relationship_id
    """)
    if not rows:
        print("Relationships: all already embedded.")
        return

    print(f"Relationships: embedding {len(rows)} rows...")
    texts = [
        f"{r['source_name']} {r['relationship_type']} {r['target_name']} {r['description']}"
        for r in rows
    ]

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_rows = rows[i : i + BATCH_SIZE]
        embeddings = embedder.encode_passages(batch_texts)

        records = [
            (r["relationship_id"], _vec_to_pg(embeddings[j]))
            for j, r in enumerate(batch_rows)
        ]
        await conn.executemany(
            "UPDATE graph.relationships SET search_embedding = $2::vector WHERE relationship_id = $1",
            records,
        )
        print(f"  Relationships: {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")


async def backfill_communities(conn: asyncpg.Connection, embedder):
    rows = await conn.fetch(
        "SELECT community_id, coalesce(title, '') AS title, coalesce(summary, '') AS summary "
        "FROM graph.communities WHERE search_embedding IS NULL "
        "ORDER BY community_id"
    )
    if not rows:
        print("Communities: all already embedded.")
        return

    print(f"Communities: embedding {len(rows)} rows...")
    texts = [f"{r['title']} {r['summary']}" for r in rows]

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_rows = rows[i : i + BATCH_SIZE]
        embeddings = embedder.encode_passages(batch_texts)

        records = [
            (r["community_id"], _vec_to_pg(embeddings[j]))
            for j, r in enumerate(batch_rows)
        ]
        await conn.executemany(
            "UPDATE graph.communities SET search_embedding = $2::vector WHERE community_id = $1",
            records,
        )
        print(f"  Communities: {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")


async def create_indexes(conn: asyncpg.Connection):
    """Create HNSW indexes if they don't exist."""
    for table, col in [
        ("entities", "idx_entities_embedding"),
        ("relationships", "idx_relationships_embedding"),
        ("communities", "idx_communities_embedding"),
    ]:
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {col}
            ON graph.{table} USING hnsw(search_embedding vector_cosine_ops);
        """)
    print("HNSW indexes ready.")


async def main():
    from src.graph.embedder import GraphEmbedder

    dsn = os.environ.get("DATABASE_URL", "")
    dsn = dsn.replace(":5433/", ":5432/")
    if not dsn:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    print(f"Connecting to: {dsn.split('@')[-1]}")
    try:
        conn = await asyncpg.connect(dsn)
    except (asyncpg.PostgresError, OSError) as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    try:
        await add_columns(conn)

        t0 = time.time()
        embedder = GraphEmbedder()

        await backfill_entities(conn, embedder)
        await backfill_relationships(conn, embedder)
        await backfill_communities(conn, embedder)
        await create_indexes(conn)

        elapsed = time.time() - t0
        print(f"\nBackfill complete in {elapsed:.1f}s")

        # Stats
        for table in ("entities", "relationships", "communities"):
            total = await conn.fetchval(f"SELECT COUNT(*) FROM graph.{table}")
            embedded = await conn.fetchval(
                f"SELECT COUNT(*) FROM graph.{table} WHERE search_embedding IS NOT NULL"
            )
            print(f"  {table}: {embedded}/{total} embedded")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
