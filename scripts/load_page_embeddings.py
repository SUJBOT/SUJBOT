"""
Load pre-computed VL page embeddings into PostgreSQL.

Reads page_embeddings.pkl and batch-inserts into vectors.vl_pages.

Usage:
    uv run python scripts/load_page_embeddings.py
    uv run python scripts/load_page_embeddings.py --pkl-path custom_embeddings.pkl
"""

import argparse
import asyncio
import os
import pickle
import sys
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv


def parse_page_id(page_id: str) -> tuple:
    """Parse BZ_VR1_p001 → ("BZ_VR1", 1)"""
    parts = page_id.rsplit("_p", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid page_id: {page_id}")
    return parts[0], int(parts[1])


def vector_to_pgvector_string(vec: np.ndarray) -> str:
    """Convert numpy array to pgvector string format."""
    return "[" + ",".join(map(str, vec.flatten().tolist())) + "]"


async def load_embeddings(pkl_path: str):
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"ERROR: File not found: {pkl_path}")
        sys.exit(1)

    # Load pickle
    print(f"Loading embeddings from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    page_ids = data["page_ids"]
    embeddings = data["embeddings"]
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)

    print(f"Loaded {len(page_ids)} page embeddings ({embeddings.shape})")

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Connect
    print("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(db_url)

    # Batch insert
    inserted = 0
    skipped = 0

    for i, page_id in enumerate(page_ids):
        doc_id, page_num = parse_page_id(page_id)
        embedding_str = vector_to_pgvector_string(embeddings[i])

        try:
            result = await conn.execute(
                """
                INSERT INTO vectors.vl_pages (page_id, document_id, page_number, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (page_id) DO NOTHING
                """,
                page_id, doc_id, page_num, embedding_str,
            )
            if result == "INSERT 0 1":
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"ERROR inserting {page_id}: {e}")
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(page_ids)} (inserted={inserted}, skipped={skipped})")

    await conn.close()

    print(f"\nDone! Inserted: {inserted}, Skipped (duplicates): {skipped}")
    print(f"Total rows in vectors.vl_pages: {inserted + skipped}")


def main():
    parser = argparse.ArgumentParser(description="Load VL page embeddings into PostgreSQL")
    parser.add_argument(
        "--pkl-path",
        default="page_embeddings.pkl",
        help="Path to page_embeddings.pkl (default: page_embeddings.pkl)",
    )
    args = parser.parse_args()
    asyncio.run(load_embeddings(args.pkl_path))


if __name__ == "__main__":
    main()
