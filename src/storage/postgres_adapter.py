"""
PostgreSQL Vector Store Adapter with pgvector

VL-only implementation using PostgreSQL + pgvector for vector similarity search.
Uses Jina v4 embeddings (2048-dim) with exact cosine scan on vectors.vl_pages.
"""

import asyncio
import json
import asyncpg
import numpy as np
from typing import List, Dict, Optional, Any
import logging

from .conversation_mixin import ConversationStorageMixin
from .vector_store_adapter import VectorStoreAdapter
from ..exceptions import DatabaseConnectionError
from ..utils.async_helpers import run_async_safe

# Backward-compatible alias for callers that import the old name
_run_async_safe = run_async_safe

logger = logging.getLogger(__name__)


class PostgresVectorStoreAdapter(ConversationStorageMixin, VectorStoreAdapter):
    """
    PostgreSQL + pgvector VL-only vector store.

    Uses vectors.vl_pages table with Jina v4 2048-dim embeddings.
    Exact cosine scan (no ANN index needed for ~500 pages).
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 20,
        dimensions: int = 2048,  # Jina v4
    ):
        """
        Initialize PostgreSQL adapter.

        Args:
            connection_string: PostgreSQL DSN (postgresql://user:pass@host:port/db)
            pool_size: Connection pool size (default: 20)
            dimensions: Embedding dimensionality (default: 2048 for Jina v4)
        """
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.dimensions = dimensions
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize connection pool."""
        if not self._initialized:
            await self._initialize_pool()
            self._initialized = True

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("PostgresVectorStoreAdapter connection pool closed")

    async def _initialize_pool(self):
        """Create asyncpg connection pool with retry logic."""
        import time

        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Attempting to connect to PostgreSQL (attempt {attempt}/{max_retries})..."
                )

                self.pool = await asyncpg.create_pool(
                    dsn=self.connection_string,
                    min_size=4,  # Optimized: 4 parallel searches need 4 connections
                    max_size=self.pool_size,
                    max_queries=50000,
                    max_inactive_connection_lifetime=300,
                    command_timeout=30,  # Optimized: fail fast for stuck queries
                    statement_cache_size=100,  # Optimized: cache prepared statements
                )
                logger.info(f"PostgreSQL connection pool created (min=4, max={self.pool_size})")

                # Verify pgvector extension is installed
                async with self.pool.acquire() as conn:
                    extensions = await conn.fetch(
                        "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'age')"
                    )
                    ext_names = {row["extname"] for row in extensions}
                    if "vector" not in ext_names:
                        raise RuntimeError("pgvector extension not installed")
                    logger.info("PostgreSQL extensions verified: vector")

                return  # Success!

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Connection attempt {attempt} failed: {e}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to initialize PostgreSQL pool after {max_retries} attempts: {e}"
                    )
                    raise DatabaseConnectionError(
                        f"PostgreSQL connection failed after {max_retries} attempts: {e}",
                        cause=e,
                    ) from e

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity (pgvector requires this)."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _vector_to_pgvector_string(self, vec: np.ndarray) -> str:
        """
        Convert numpy array to PostgreSQL vector string format.

        asyncpg doesn't automatically serialize Python lists to pgvector type,
        so we need to manually convert to string format: '[1.0,2.0,3.0]'
        """
        # Flatten array if needed (handles 2D arrays from some embedding models)
        if isinstance(vec, np.ndarray):
            vec = vec.flatten().tolist()
        elif isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
            # Handle nested lists (e.g., [[0.1, 0.2, ...]])
            vec = vec[0] if len(vec) == 1 else sum(vec, [])

        return "[" + ",".join(map(str, vec)) + "]"

    async def _ensure_pool(self):
        """Ensure connection pool is initialized (thread-safe)."""
        if self.pool is None:
            async with self._init_lock:
                # Double-check after acquiring lock
                if self.pool is None:
                    await self.initialize()

    def get_document_list(self, category_filter: Optional[str] = None) -> List[str]:
        """Get list of unique document IDs, optionally filtered by category."""
        return run_async_safe(self._async_get_document_list(category_filter))

    async def _async_get_document_list(self, category_filter: Optional[str] = None) -> List[str]:
        """Async get document list from vl_pages."""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            base = "SELECT DISTINCT document_id FROM vectors.vl_pages"

            if category_filter:
                query = (
                    f"{base} WHERE document_id IN "
                    f"(SELECT document_id FROM vectors.documents WHERE category = $1) "
                    f"ORDER BY document_id"
                )
                rows = await conn.fetch(query, category_filter)
            else:
                rows = await conn.fetch(f"{base} ORDER BY document_id")

            return [row["document_id"] for row in rows]

    def get_document_categories(self) -> Dict[str, str]:
        """Get mapping of document_id -> category from vectors.documents."""
        return run_async_safe(self._async_get_document_categories())

    async def _async_get_document_categories(self) -> Dict[str, str]:
        """Async get document categories."""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT document_id, category FROM vectors.documents"
            )
            return {row["document_id"]: row["category"] for row in rows}

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return run_async_safe(self._async_get_stats())

    async def _async_get_stats(self) -> Dict[str, Any]:
        """Async get stats — VL-only."""
        await self._ensure_pool()
        async with self.pool.acquire() as conn:
            vl_count = await conn.fetchval("SELECT COUNT(*) FROM vectors.vl_pages")
            doc_count = await conn.fetchval(
                "SELECT COUNT(DISTINCT document_id) FROM vectors.vl_pages"
            )
            return {
                "documents": doc_count,
                "total_vectors": vl_count,
                "vl_pages_count": vl_count,
                "dimensions": self.dimensions,
                "backend": "postgresql",
                "architecture": "vl",
            }

    # ============================================================================
    # VL (Vision-Language) Page Embeddings
    # ============================================================================

    _VALID_CATEGORIES = {"documentation", "legislation"}

    def search_vl_pages(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        document_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Search VL page embeddings (2048-dim Jina v4) by cosine similarity.

        Args:
            query_embedding: Query embedding (2048-dim, L2-normalized)
            k: Number of results
            document_filter: Optional document ID filter
            category_filter: Optional category filter ('documentation' or 'legislation')

        Returns:
            List of dicts with page_id, document_id, page_number, score, image_path, metadata
        """
        return run_async_safe(
            self._async_search_vl_pages(query_embedding, k, document_filter, category_filter),
            operation_name="search_vl_pages",
        )

    async def _async_search_vl_pages(
        self,
        query_embedding: np.ndarray,
        k: int,
        document_filter: Optional[str],
        category_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Async VL page search."""
        await self._ensure_pool()
        query_vec = self._normalize_vector(query_embedding)
        query_str = self._vector_to_pgvector_string(query_vec)

        params: list = [query_str]
        where_parts: list = []
        param_idx = 1

        if document_filter:
            param_idx += 1
            where_parts.append(f"document_id = ${param_idx}")
            params.append(document_filter)

        if category_filter:
            if category_filter not in self._VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid category_filter: {category_filter!r}. Must be one of {self._VALID_CATEGORIES}"
                )
            param_idx += 1
            where_parts.append(
                f"document_id IN (SELECT document_id FROM vectors.documents WHERE category = ${param_idx})"
            )
            params.append(category_filter)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        param_idx += 1
        params.append(k)

        sql = f"""
            SELECT page_id, document_id, page_number, image_path, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM vectors.vl_pages
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${param_idx}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            result = {
                "page_id": row["page_id"],
                "document_id": row["document_id"],
                "page_number": row["page_number"],
                "score": float(row["score"]),
                "image_path": row.get("image_path"),
                "metadata": row.get("metadata") or {},
            }
            results.append(result)

        return results

    def get_adjacent_vl_pages(
        self,
        document_id: str,
        page_number: int,
        k: int = 3,
    ) -> List[Dict]:
        """
        Get adjacent VL pages (page_number +/- k) from the same document.

        Args:
            document_id: Document to search within
            page_number: Center page number
            k: Number of pages in each direction

        Returns:
            List of dicts with page_id, document_id, page_number, image_path, metadata
            (excludes the center page itself, ordered by page_number)
        """
        return run_async_safe(
            self._async_get_adjacent_vl_pages(document_id, page_number, k),
            operation_name="get_adjacent_vl_pages",
        )

    async def _async_get_adjacent_vl_pages(
        self,
        document_id: str,
        page_number: int,
        k: int,
    ) -> List[Dict]:
        """Async implementation of adjacent VL page retrieval."""
        await self._ensure_pool()

        sql = """
            SELECT page_id, document_id, page_number, image_path, metadata
            FROM vectors.vl_pages
            WHERE document_id = $1
              AND page_number BETWEEN $2 AND $3
              AND page_number != $4
            ORDER BY page_number
        """
        page_from = page_number - k
        page_to = page_number + k

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, document_id, page_from, page_to, page_number)

        return [
            {
                "page_id": row["page_id"],
                "document_id": row["document_id"],
                "page_number": row["page_number"],
                "image_path": row.get("image_path"),
                "metadata": row.get("metadata") or {},
            }
            for row in rows
        ]

    def update_vl_page_metadata(self, page_id: str, metadata_patch: Dict[str, Any]) -> None:
        """
        Merge a metadata patch into an existing VL page's metadata JSONB.

        Uses PostgreSQL's || operator for shallow top-level merge --
        adds/overwrites keys without deleting existing ones.

        Args:
            page_id: Page identifier (e.g., "BZ_VR1_p001")
            metadata_patch: Dict of keys to merge (e.g., {"page_summary": "...", "summary_model": "..."})
        """
        run_async_safe(
            self._async_update_vl_page_metadata(page_id, metadata_patch),
            operation_name="update_vl_page_metadata",
        )

    async def _async_update_vl_page_metadata(
        self, page_id: str, metadata_patch: Dict[str, Any]
    ) -> None:
        """Async implementation of update_vl_page_metadata."""

        await self._ensure_pool()
        patch_json = json.dumps(metadata_patch)

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE vectors.vl_pages
                SET metadata = metadata || $2::jsonb
                WHERE page_id = $1
                """,
                page_id,
                patch_json,
            )
            if result == "UPDATE 0":
                logger.warning(f"update_vl_page_metadata: no row found for page_id={page_id}")

    def get_vl_pages_without_summary(
        self, document_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get VL pages that don't have a page_summary in their metadata.

        Args:
            document_id: Optional filter to a single document

        Returns:
            List of dicts with page_id, document_id, page_number
        """
        return run_async_safe(
            self._async_get_vl_pages_without_summary(document_id),
            operation_name="get_vl_pages_without_summary",
        )

    async def _async_get_vl_pages_without_summary(
        self, document_id: Optional[str] = None
    ) -> List[Dict]:
        """Async implementation of get_vl_pages_without_summary."""
        await self._ensure_pool()

        sql = """
            SELECT page_id, document_id, page_number
            FROM vectors.vl_pages
            WHERE metadata->>'page_summary' IS NULL
        """
        params: list = []

        if document_id:
            sql += " AND document_id = $1"
            params.append(document_id)

        sql += " ORDER BY document_id, page_number"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "page_id": row["page_id"],
                "document_id": row["document_id"],
                "page_number": row["page_number"],
            }
            for row in rows
        ]

    def add_vl_pages(
        self,
        pages: List[Dict],
        embeddings: np.ndarray,
    ) -> int:
        """
        Batch insert VL page embeddings.

        Args:
            pages: List of dicts with page_id, document_id, page_number, image_path, metadata
            embeddings: Embedding matrix (N x 2048)

        Returns:
            Number of rows inserted
        """
        return run_async_safe(
            self._async_add_vl_pages(pages, embeddings),
            operation_name="add_vl_pages",
        )

    async def _async_add_vl_pages(
        self,
        pages: List[Dict],
        embeddings: np.ndarray,
    ) -> int:
        """Async batch insert VL page embeddings."""

        await self._ensure_pool()

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)

        records = []
        for i, page in enumerate(pages):
            embedding_str = self._vector_to_pgvector_string(embeddings[i])
            metadata_json = json.dumps(page.get("metadata", {}))
            records.append(
                (
                    page["page_id"],
                    page["document_id"],
                    page["page_number"],
                    embedding_str,
                    page.get("image_path"),
                    metadata_json,
                )
            )

        sql = """
            INSERT INTO vectors.vl_pages
                (page_id, document_id, page_number, embedding, image_path, metadata)
            VALUES ($1, $2, $3, $4::vector, $5, $6::jsonb)
            ON CONFLICT (page_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                page_number = EXCLUDED.page_number,
                image_path = EXCLUDED.image_path,
                metadata = EXCLUDED.metadata
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(sql, records)

        logger.info(f"VL pages: inserted batch of {len(records)} pages")
        return len(records)

    # ============================================================================
    # Cleanup
    # ============================================================================

    def __del__(self):
        """Close connection pool on destruction."""
        if self.pool:
            try:
                # Try graceful close - may fail during Python shutdown
                run_async_safe(self.pool.close())
            except Exception as e:
                # Log errors during destruction for debugging
                # (don't raise - Python may be shutting down)
                try:
                    logger.debug(f"Error closing PostgreSQL pool during cleanup: {e}")
                except Exception:
                    # Logger may also be unavailable during shutdown
                    pass


# ====================================================================================
# PostgreSQL Storage Adapter for Authentication & User Data
# ====================================================================================


class PostgreSQLStorageAdapter(ConversationStorageMixin):
    """
    PostgreSQL connection pool for auth/user data.

    Separate from PostgresVectorStoreAdapter (vector search).
    This adapter provides connection pooling for:
    - auth.users (user accounts)
    - auth.conversations (chat sessions)
    - auth.messages (message history)

    Used by AuthQueries for user management and conversation storage.
    Conversation CRUD methods inherited from ConversationStorageMixin.
    """

    def __init__(self):
        """Initialize adapter (connection pool created in initialize())."""
        import os

        self.pool: Optional[asyncpg.Pool] = None
        self.db_url = os.getenv("DATABASE_URL")

        if not self.db_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set in .env file (e.g., postgresql://postgres:password@postgres:5432/sujbot)"
            )

    async def initialize(self):
        """Create connection pool."""
        logger.info("Creating PostgreSQL connection pool for auth/user storage...")

        self.pool = await asyncpg.create_pool(
            self.db_url, min_size=2, max_size=10, command_timeout=60
        )

        # Test connection
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"PostgreSQL connected: {version}")

        logger.info("Connection pool created (2-10 connections)")

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
