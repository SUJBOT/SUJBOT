#!/usr/bin/env python3
"""
Migrate existing graph entities to use the entity_aliases table.

Creates the entity_aliases table if it doesn't exist, then backfills aliases
from entity names. Safe to run multiple times — uses ON CONFLICT DO NOTHING.

Usage:
    uv run python scripts/graph_alias_migrate.py
"""

import asyncio
import os
import sys

import nest_asyncio

nest_asyncio.apply()

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


async def main():
    import asyncpg

    dsn = os.environ.get("DATABASE_URL", "")
    dsn = dsn.replace(":5433/", ":5432/")
    if not dsn:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    print(f"Connecting to: {dsn.split('@')[-1]}")
    conn = await asyncpg.connect(dsn)

    try:
        # Create entity_aliases table if not exists
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph.entity_aliases (
                alias_id    SERIAL PRIMARY KEY,
                entity_id   INT NOT NULL REFERENCES graph.entities(entity_id) ON DELETE CASCADE,
                alias       TEXT NOT NULL,
                alias_type  TEXT NOT NULL DEFAULT 'variant',
                language    TEXT,
                source      TEXT,
                created_at  TIMESTAMPTZ DEFAULT now()
            )
        """
        )
        await conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_aliases_unique
            ON graph.entity_aliases (lower(alias), entity_id)
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_lookup
            ON graph.entity_aliases (lower(alias))
        """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity
            ON graph.entity_aliases (entity_id)
        """
        )
        print("Table and indexes ready.")

        # Backfill: insert each entity's own name as an alias
        result = await conn.execute(
            """
            INSERT INTO graph.entity_aliases (entity_id, alias, alias_type, source)
            SELECT entity_id, name, 'variant', 'migration'
            FROM graph.entities
            ON CONFLICT (lower(alias), entity_id) DO NOTHING
        """
        )
        count = int(result.split()[-1]) if result else 0
        print(f"Backfilled {count} entity name aliases.")

        # Stats
        total_entities = await conn.fetchval("SELECT COUNT(*) FROM graph.entities")
        total_aliases = await conn.fetchval("SELECT COUNT(*) FROM graph.entity_aliases")
        print(f"\nStats: {total_entities} entities, {total_aliases} aliases")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
