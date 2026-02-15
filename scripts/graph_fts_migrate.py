#!/usr/bin/env python3
"""
One-time migration: add PostgreSQL full-text search to graph schema.

Adds search_tsv TSVECTOR columns + triggers + GIN indexes to
graph.entities and graph.communities, then backfills existing rows.

Safe to run multiple times (all operations are idempotent).

Usage:
    uv run python scripts/graph_fts_migrate.py
"""

import asyncio
import os
import sys

import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def migrate():
    dsn = os.environ.get("DATABASE_URL", "")
    # Override dev port → production
    dsn = dsn.replace(":5433/", ":5432/")
    if not dsn:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    print(f"Connecting to: {dsn.split('@')[-1]}")
    conn = await asyncpg.connect(dsn)

    try:
        # 1. Ensure unaccent extension
        print("Creating unaccent extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")

        # 2. Add search_tsv columns (IF NOT EXISTS via DO block)
        print("Adding search_tsv columns...")
        await conn.execute("""
            DO $$ BEGIN
                ALTER TABLE graph.entities ADD COLUMN search_tsv TSVECTOR;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        await conn.execute("""
            DO $$ BEGIN
                ALTER TABLE graph.communities ADD COLUMN search_tsv TSVECTOR;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)

        # 3. Create trigger functions
        print("Creating trigger functions...")
        await conn.execute("""
            CREATE OR REPLACE FUNCTION graph.entities_search_tsv_trigger() RETURNS trigger AS $$
            BEGIN
                NEW.search_tsv := to_tsvector('simple', unaccent(
                    coalesce(NEW.name, '') || ' ' || coalesce(NEW.description, '')
                ));
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        await conn.execute("""
            CREATE OR REPLACE FUNCTION graph.communities_search_tsv_trigger() RETURNS trigger AS $$
            BEGIN
                NEW.search_tsv := to_tsvector('simple', unaccent(
                    coalesce(NEW.title, '') || ' ' || coalesce(NEW.summary, '')
                ));
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)

        # 4. Create triggers (drop first to make idempotent)
        print("Creating triggers...")
        await conn.execute(
            "DROP TRIGGER IF EXISTS trg_entities_search_tsv ON graph.entities;"
        )
        await conn.execute("""
            CREATE TRIGGER trg_entities_search_tsv
                BEFORE INSERT OR UPDATE OF name, description ON graph.entities
                FOR EACH ROW EXECUTE FUNCTION graph.entities_search_tsv_trigger();
        """)
        await conn.execute(
            "DROP TRIGGER IF EXISTS trg_communities_search_tsv ON graph.communities;"
        )
        await conn.execute("""
            CREATE TRIGGER trg_communities_search_tsv
                BEFORE INSERT OR UPDATE OF title, summary ON graph.communities
                FOR EACH ROW EXECUTE FUNCTION graph.communities_search_tsv_trigger();
        """)

        # 5. Create GIN indexes
        print("Creating GIN indexes...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_search_tsv
            ON graph.entities USING gin(search_tsv);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_communities_search_tsv
            ON graph.communities USING gin(search_tsv);
        """)

        # 6. Backfill existing rows
        entity_count = await conn.fetchval("SELECT COUNT(*) FROM graph.entities WHERE search_tsv IS NULL")
        print(f"Backfilling {entity_count} entities...")
        await conn.execute("""
            UPDATE graph.entities
            SET search_tsv = to_tsvector('simple', unaccent(
                coalesce(name, '') || ' ' || coalesce(description, '')
            ))
            WHERE search_tsv IS NULL;
        """)

        community_count = await conn.fetchval(
            "SELECT COUNT(*) FROM graph.communities WHERE search_tsv IS NULL"
        )
        print(f"Backfilling {community_count} communities...")
        await conn.execute("""
            UPDATE graph.communities
            SET search_tsv = to_tsvector('simple', unaccent(
                coalesce(title, '') || ' ' || coalesce(summary, '')
            ))
            WHERE search_tsv IS NULL;
        """)

        # 7. Verify
        total_entities = await conn.fetchval("SELECT COUNT(*) FROM graph.entities")
        indexed_entities = await conn.fetchval(
            "SELECT COUNT(*) FROM graph.entities WHERE search_tsv IS NOT NULL"
        )
        total_communities = await conn.fetchval("SELECT COUNT(*) FROM graph.communities")
        indexed_communities = await conn.fetchval(
            "SELECT COUNT(*) FROM graph.communities WHERE search_tsv IS NOT NULL"
        )

        print(f"\nMigration complete!")
        print(f"  Entities:    {indexed_entities}/{total_entities} indexed")
        print(f"  Communities: {indexed_communities}/{total_communities} indexed")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
