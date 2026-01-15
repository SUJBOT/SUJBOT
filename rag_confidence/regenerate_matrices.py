#!/usr/bin/env python3
"""
Regenerate similarity and relevance matrices using production embeddings.

Uses Qwen3-Embedding-8B (4096D) via DeepInfra to embed queries,
then computes cosine similarity against chunk embeddings from PostgreSQL.

Usage:
    uv run python rag_confidence/regenerate_matrices.py

Output:
    - rag_confidence/synthetic_similarity_matrix.npz
    - rag_confidence/synthetic_relevance_matrix.npz
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import psycopg
from psycopg.rows import dict_row
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# DeepInfra embedding model config
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIMENSIONS = 4096
BATCH_SIZE = 50


def get_db_connection():
    """Get PostgreSQL connection from environment."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg.connect(db_url)

    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "sujbot"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def get_deepinfra_client():
    """Initialize DeepInfra client (OpenAI-compatible)."""
    api_key = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY environment variable required")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        timeout=60,
        max_retries=3,
    )


def load_chunk_embeddings(conn) -> tuple[np.ndarray, list[str]]:
    """Load all chunk embeddings from PostgreSQL."""
    logger.info("Loading chunk embeddings from PostgreSQL...")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT chunk_id, embedding::text
            FROM vectors.layer3
            ORDER BY chunk_id
        """)
        rows = cur.fetchall()

    chunk_ids = [row["chunk_id"] for row in rows]

    # Parse embeddings (stored as vector type, cast to text in query)
    embeddings = []
    for row in rows:
        emb_str = row["embedding"]
        if isinstance(emb_str, str):
            emb_str = emb_str.strip("[]")
            emb = np.array([float(x) for x in emb_str.split(",")])
        else:
            emb = np.array(emb_str)
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)

    logger.info(f"Loaded {len(chunk_ids)} chunks, embedding shape: {embeddings.shape}")
    return embeddings, chunk_ids


def load_queries(dataset_path: Path) -> list[dict]:
    """Load queries from synthetic eval dataset."""
    logger.info(f"Loading queries from {dataset_path}")

    with open(dataset_path) as f:
        data = json.load(f)

    queries = data["queries"]
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def embed_queries(queries: list[dict], client: OpenAI) -> np.ndarray:
    """Embed query texts using DeepInfra Qwen3-Embedding-8B."""
    logger.info(f"Embedding {len(queries)} queries with {EMBEDDING_MODEL}...")

    query_texts = [q["query_text"] for q in queries]
    all_embeddings = []

    # Process in batches
    for i in range(0, len(query_texts), BATCH_SIZE):
        batch = query_texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(query_texts) - 1) // BATCH_SIZE + 1

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            encoding_format="float"
        )

        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Query embeddings shape: {embeddings.shape}")
    return embeddings


def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix between queries and chunks."""
    logger.info("Computing similarity matrix...")

    # Normalize embeddings for cosine similarity
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

    query_normalized = query_embeddings / np.where(query_norms > 0, query_norms, 1)
    chunk_normalized = chunk_embeddings / np.where(chunk_norms > 0, chunk_norms, 1)

    # Compute similarity (batch matrix multiplication)
    similarity_matrix = query_normalized @ chunk_normalized.T

    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

    return similarity_matrix


def build_relevance_matrix(
    queries: list[dict],
    chunk_ids: list[str],
) -> np.ndarray:
    """Build binary relevance matrix from query annotations."""
    logger.info("Building relevance matrix...")

    n_queries = len(queries)
    n_chunks = len(chunk_ids)

    chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    relevance_matrix = np.zeros((n_queries, n_chunks), dtype=np.int8)

    missing_chunks = set()
    found_count = 0

    for i, query in enumerate(queries):
        for rel_chunk_id in query["relevant_chunk_ids"]:
            if rel_chunk_id in chunk_id_to_idx:
                relevance_matrix[i, chunk_id_to_idx[rel_chunk_id]] = 1
                found_count += 1
            else:
                missing_chunks.add(rel_chunk_id)

    if missing_chunks:
        logger.warning(
            f"Missing {len(missing_chunks)} chunk IDs from database. "
            f"Examples: {list(missing_chunks)[:5]}"
        )

    logger.info(f"Relevance matrix: {found_count} relevant pairs across {n_queries} queries")
    logger.info(f"Avg relevant chunks per query: {relevance_matrix.sum(axis=1).mean():.2f}")

    return relevance_matrix


def main():
    data_dir = Path(__file__).parent
    dataset_path = data_dir / "synthetic_eval_dataset.json"

    # Load queries
    queries = load_queries(dataset_path)

    # Connect to database and load chunk embeddings
    logger.info("Connecting to PostgreSQL...")
    conn = get_db_connection()

    try:
        chunk_embeddings, chunk_ids = load_chunk_embeddings(conn)
    finally:
        conn.close()

    # Verify dimensions match expected
    if chunk_embeddings.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Dimension mismatch: expected {EMBEDDING_DIMENSIONS}, "
            f"got {chunk_embeddings.shape[1]}"
        )

    # Initialize DeepInfra client
    client = get_deepinfra_client()

    # Embed queries
    query_embeddings = embed_queries(queries, client)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(query_embeddings, chunk_embeddings)

    # Build relevance matrix
    relevance_matrix = build_relevance_matrix(queries, chunk_ids)

    # Save matrices
    query_ids = np.array([q["query_id"] for q in queries])
    chunk_ids_arr = np.array(chunk_ids)

    sim_path = data_dir / "synthetic_similarity_matrix.npz"
    rel_path = data_dir / "synthetic_relevance_matrix.npz"

    logger.info(f"Saving similarity matrix to {sim_path}")
    np.savez_compressed(
        sim_path,
        similarity=similarity_matrix,
        query_ids=query_ids,
        chunk_ids=chunk_ids_arr,
    )

    logger.info(f"Saving relevance matrix to {rel_path}")
    np.savez_compressed(
        rel_path,
        relevance=relevance_matrix,
        query_ids=query_ids,
        chunk_ids=chunk_ids_arr,
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    print(f"Queries:           {len(queries)}")
    print(f"Chunks:            {len(chunk_ids)}")
    print(f"Embedding model:   {EMBEDDING_MODEL}")
    print(f"Dimensions:        {EMBEDDING_DIMENSIONS}")
    print(f"Similarity range:  [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(f"Relevant pairs:    {relevance_matrix.sum()}")
    print("=" * 60)

    # Quick Recall@K check
    print("\nRecall@K (should be HIGH if embeddings are good):")
    for k in [1, 5, 10, 20, 50, 100]:
        recall_at_k = []
        for i in range(len(queries)):
            top_k_indices = np.argpartition(-similarity_matrix[i], k-1)[:k]
            relevant_in_top_k = relevance_matrix[i, top_k_indices].sum()
            total_relevant = relevance_matrix[i].sum()
            if total_relevant > 0:
                recall_at_k.append(relevant_in_top_k / total_relevant)
        avg_recall = np.mean(recall_at_k) if recall_at_k else 0
        print(f"  Recall@{k:3d}: {avg_recall*100:5.1f}%")

    # Rank statistics
    ranks = []
    for i in range(similarity_matrix.shape[0]):
        sorted_indices = np.argsort(-similarity_matrix[i])
        rank_of_each = np.argsort(sorted_indices)
        relevant_mask = relevance_matrix[i] > 0
        if relevant_mask.any():
            relevant_ranks = rank_of_each[relevant_mask]
            ranks.extend(relevant_ranks.tolist())

    if ranks:
        ranks = np.array(ranks)
        print(f"\nRelevant chunk rank statistics:")
        print(f"  Mean rank:   {ranks.mean():.1f}")
        print(f"  Median rank: {np.median(ranks):.0f}")
        print(f"  % in top-10: {(ranks < 10).mean()*100:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
