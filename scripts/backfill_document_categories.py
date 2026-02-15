"""
Backfill vectors.documents table from existing vl_pages data.

Heuristic:
  - document_id starting with "Sb_" → legislation (Czech Sbírka zákonů)
  - everything else → documentation

Usage:
    uv run python scripts/backfill_document_categories.py
"""

import asyncio
import os
import re

import nest_asyncio

nest_asyncio.apply()

import asyncpg
from dotenv import load_dotenv

load_dotenv()

# Use production port 5432, not dev 5433
DATABASE_URL = os.getenv("DATABASE_URL", "").replace(":5433/", ":5432/")


def classify_document(document_id: str) -> str:
    """Classify document_id as 'legislation' or 'documentation'."""
    if re.match(r"^Sb_", document_id):
        return "legislation"
    return "documentation"


def format_display_name(document_id: str) -> str:
    """Create human-readable display name (mirrors backend logic)."""
    legal_match = re.match(r"Sb_(\d{4})_(\d+)_.*", document_id)
    if legal_match:
        year, number = legal_match.groups()
        return f"{number}/{year} Sb."
    return document_id.replace("_", " ")


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Ensure table exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS vectors.documents (
                document_id TEXT PRIMARY KEY,
                category TEXT NOT NULL DEFAULT 'documentation'
                    CHECK (category IN ('documentation', 'legislation')),
                display_name TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)

        # Get distinct document IDs from vl_pages
        rows = await conn.fetch(
            "SELECT DISTINCT document_id FROM vectors.vl_pages ORDER BY document_id"
        )
        print(f"Found {len(rows)} documents in vectors.vl_pages")

        inserted = 0
        for row in rows:
            doc_id = row["document_id"]
            category = classify_document(doc_id)
            display_name = format_display_name(doc_id)

            result = await conn.execute(
                """
                INSERT INTO vectors.documents (document_id, category, display_name)
                VALUES ($1, $2, $3)
                ON CONFLICT (document_id) DO NOTHING
                """,
                doc_id,
                category,
                display_name,
            )
            if result == "INSERT 0 1":
                inserted += 1
                print(f"  {doc_id} → {category} ({display_name})")

        print(f"\nInserted {inserted} new rows, {len(rows) - inserted} already existed")

        # Show final state
        final = await conn.fetch(
            "SELECT document_id, category, display_name FROM vectors.documents ORDER BY category, document_id"
        )
        print(f"\nFinal state ({len(final)} documents):")
        for r in final:
            print(f"  [{r['category']:<13}] {r['document_id']} — {r['display_name']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
