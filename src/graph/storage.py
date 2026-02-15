"""
Graph Storage Adapter — PostgreSQL backend for knowledge graph.

Stores entities, relationships, and communities in the `graph` schema.
Follows PostgresVectorStoreAdapter pattern: asyncpg pool + sync wrappers.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .embedder import GraphEmbedder

import asyncpg
import numpy as np

from ..exceptions import DatabaseConnectionError, GraphStoreError

logger = logging.getLogger(__name__)

# Column lists reused across entity queries
_ENTITY_COLS = "entity_id, name, entity_type, description, document_id"


def _vec_to_pg(vec: np.ndarray) -> str:
    """Convert numpy array to pgvector string '[0.1,0.2,...]'."""
    return "[" + ",".join(map(str, vec.flatten().tolist())) + "]"


def _entity_from_row(row: asyncpg.Record) -> Dict[str, Any]:
    """Convert an asyncpg Record to an entity dict. Shared by all entity queries."""
    return {
        "entity_id": row["entity_id"],
        "name": row["name"],
        "entity_type": row["entity_type"],
        "description": row["description"],
        "document_id": row["document_id"],
    }


def _community_from_row(row: asyncpg.Record) -> Dict[str, Any]:
    """Convert an asyncpg Record to a community dict. Shared by all community queries."""
    result = {
        "community_id": row["community_id"],
        "level": row["level"],
        "title": row["title"],
        "summary": row["summary"],
        "entity_ids": row["entity_ids"],
        "metadata": row["metadata"] or {},
    }
    if "score" in row.keys():
        result["score"] = float(row["score"])
    return result


def _in_placeholders(values, start: int = 1) -> str:
    """Build '$1, $2, ...' placeholders for an IN clause."""
    return ", ".join(f"${start + i}" for i in range(len(values)))


def _run_async_safe(coro, timeout: float = 30.0, operation_name: str = "graph operation"):
    """Safely run async coroutine from sync context (same pattern as postgres_adapter.py)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop is None:
            return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
        else:
            return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout ({timeout}s) during {operation_name}")
        raise TimeoutError(f"'{operation_name}' timed out after {timeout}s") from e
    except RuntimeError as e:
        if "This event loop is already running" in str(e):
            raise RuntimeError(
                f"Cannot run '{operation_name}' synchronously inside an async context. "
                "Use the async method directly or apply nest_asyncio."
            ) from e
        raise


def _parse_command_count(result: Optional[str]) -> int:
    """Parse row count from asyncpg command result (e.g. 'DELETE 5' → 5)."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError, AttributeError):
        return 0


def _parse_dedup_verdict(text: str) -> bool:
    """Parse LLM dedup verdict from structured JSON or plain text fallback."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    try:
        data = json.loads(text)
        return str(data.get("verdict", "")).strip().upper() == "YES"
    except (json.JSONDecodeError, AttributeError):
        # Fallback: plain text starts with YES
        return text.upper().startswith("YES")


class GraphStorageAdapter:
    """
    PostgreSQL storage for knowledge graph (entities, relationships, communities).

    Uses asyncpg connection pool. Can share pool with PostgresVectorStoreAdapter
    or create its own.
    """

    def __init__(
        self,
        pool: Optional[asyncpg.Pool] = None,
        connection_string: Optional[str] = None,
        embedder: Optional["GraphEmbedder"] = None,
    ):
        """
        Initialize with existing pool or connection string.

        Args:
            pool: Existing asyncpg pool (preferred — avoids duplicate connections).
                  When provided, the caller owns the pool lifecycle (we won't close it).
            connection_string: PostgreSQL DSN (creates new pool if pool is None).
                  When used, this adapter owns and will close the pool.
            embedder: Optional GraphEmbedder for semantic search on entities/communities.
                When provided, search methods use cosine similarity on embeddings.
                When None, falls back to full-text search.
        """
        self.pool = pool
        self._connection_string = connection_string
        self._owns_pool = pool is None
        self._embedder = embedder

    async def initialize(self):
        """Create connection pool if not provided."""
        await self._ensure_pool()

    async def _ensure_pool(self):
        """Lazily create connection pool on first use."""
        if self.pool is None:
            if not self._connection_string:
                raise DatabaseConnectionError("Either pool or connection_string must be provided")
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=self._connection_string,
                    min_size=2,
                    max_size=10,
                    command_timeout=30,
                )
                logger.info("Graph storage pool created")
            except (asyncpg.PostgresError, OSError) as e:
                raise DatabaseConnectionError(
                    f"Failed to create graph storage pool: {e}",
                    cause=e,
                ) from e

    async def close(self):
        """Close pool if we own it."""
        if self._owns_pool and self.pool:
            await self.pool.close()

    # =========================================================================
    # Entity & Relationship Storage
    # =========================================================================

    def add_entities(
        self, entities: List[Dict], document_id: str, source_page_id: Optional[str] = None
    ) -> int:
        """Bulk upsert entities (sync). Returns count of records processed."""
        return _run_async_safe(
            self.async_add_entities(entities, document_id, source_page_id),
            operation_name="add_entities",
        )

    async def async_add_entities(
        self, entities: List[Dict], document_id: str, source_page_id: Optional[str] = None
    ) -> int:
        if not entities:
            return 0
        await self._ensure_pool()

        records = [
            (
                e["name"],
                e["type"],
                e.get("description"),
                source_page_id,
                document_id,
                json.dumps(e.get("metadata", {})),
            )
            for e in entities
        ]

        sql = """
            INSERT INTO graph.entities (name, entity_type, description, source_page_id, document_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (name, entity_type, document_id) DO UPDATE SET
                description = COALESCE(EXCLUDED.description, graph.entities.description),
                source_page_id = COALESCE(EXCLUDED.source_page_id, graph.entities.source_page_id),
                metadata = graph.entities.metadata || EXCLUDED.metadata
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(sql, records)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to add {len(records)} entities for document {document_id}: {e}",
                cause=e,
            ) from e

        return len(records)

    def add_relationships(
        self, relationships: List[Dict], document_id: str, source_page_id: Optional[str] = None
    ) -> int:
        """Bulk insert relationships (sync). Resolves entity names to IDs."""
        return _run_async_safe(
            self.async_add_relationships(relationships, document_id, source_page_id),
            operation_name="add_relationships",
        )

    @staticmethod
    async def _resolve_entity_id(
        conn: asyncpg.Connection, name: str, document_id: str
    ) -> Optional[int]:
        """Resolve entity name to ID: document-scoped first, then cross-document fallback."""
        entity_id = await conn.fetchval(
            "SELECT entity_id FROM graph.entities WHERE name = $1 AND document_id = $2 LIMIT 1",
            name,
            document_id,
        )
        if entity_id is None:
            entity_id = await conn.fetchval(
                "SELECT entity_id FROM graph.entities WHERE name = $1 LIMIT 1", name
            )
        return entity_id

    async def async_add_relationships(
        self, relationships: List[Dict], document_id: str, source_page_id: Optional[str] = None
    ) -> int:
        if not relationships:
            return 0
        await self._ensure_pool()

        try:
            async with self.pool.acquire() as conn:
                inserted = 0
                skipped = 0
                for r in relationships:
                    source_id = await self._resolve_entity_id(conn, r["source"], document_id)
                    target_id = await self._resolve_entity_id(conn, r["target"], document_id)

                    if source_id is None or target_id is None:
                        logger.debug(
                            f"Skipping relationship: cannot resolve "
                            f"'{r['source']}' ({source_id}) -> '{r['target']}' ({target_id})"
                        )
                        skipped += 1
                        continue

                    await conn.execute(
                        """
                        INSERT INTO graph.relationships
                            (source_entity_id, target_entity_id, relationship_type, description,
                             weight, source_page_id, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                        """,
                        source_id,
                        target_id,
                        r["type"],
                        r.get("description"),
                        r.get("weight", 1.0),
                        source_page_id,
                        json.dumps(r.get("metadata", {})),
                    )
                    inserted += 1

            if skipped > 0:
                logger.warning(
                    f"Skipped {skipped}/{len(relationships)} relationships "
                    f"for document {document_id} (unresolved entities)"
                )
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to add relationships for document {document_id}: {e}",
                cause=e,
            ) from e

        return inserted

    # =========================================================================
    # Entity Search
    # =========================================================================

    def search_entities(
        self, query: str, entity_type: Optional[str] = None, limit: int = 10
    ) -> List[Dict]:
        """Search entities by embedding similarity (or FTS fallback)."""
        return _run_async_safe(
            self.async_search_entities(query, entity_type, limit),
            operation_name="search_entities",
        )

    async def async_search_entities(
        self, query: str, entity_type: Optional[str], limit: int
    ) -> List[Dict]:
        await self._ensure_pool()

        if self._embedder:
            return await self._search_entities_by_embedding(query, entity_type, limit)
        logger.debug("No embedder configured, falling back to FTS for entity search")
        return await self._search_entities_by_fts(query, entity_type, limit)

    async def _search_entities_by_embedding(
        self, query: str, entity_type: Optional[str], limit: int
    ) -> List[Dict]:
        vec = await asyncio.to_thread(self._embedder.encode_query, query)
        query_vec = _vec_to_pg(vec)
        params: list = [query_vec]
        param_idx = 1

        type_filter = ""
        if entity_type:
            param_idx += 1
            type_filter = f"AND entity_type = ${param_idx}"
            params.append(entity_type)

        param_idx += 1
        params.append(limit)

        sql = f"""
            SELECT {_ENTITY_COLS}, metadata,
                   1 - (search_embedding <=> $1::vector) AS score
            FROM graph.entities
            WHERE search_embedding IS NOT NULL
            {type_filter}
            ORDER BY search_embedding <=> $1::vector
            LIMIT ${param_idx}
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Entity embedding search failed for query '{query}': {e}", cause=e
            ) from e

        return [
            {
                **_entity_from_row(row),
                "metadata": row["metadata"] or {},
                "score": float(row["score"]),
            }
            for row in rows
        ]

    async def _search_entities_by_fts(
        self, query: str, entity_type: Optional[str], limit: int
    ) -> List[Dict]:
        params: list = [query]
        param_idx = 1

        type_filter = ""
        if entity_type:
            param_idx += 1
            type_filter = f"AND entity_type = ${param_idx}"
            params.append(entity_type)

        param_idx += 1
        params.append(limit)

        sql = f"""
            SELECT {_ENTITY_COLS}, metadata,
                   ts_rank(search_tsv, plainto_tsquery('simple', unaccent($1))) AS score
            FROM graph.entities
            WHERE search_tsv @@ plainto_tsquery('simple', unaccent($1))
            {type_filter}
            ORDER BY score DESC
            LIMIT ${param_idx}
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Entity FTS search failed for query '{query}': {e}", cause=e
            ) from e

        return [
            {
                **_entity_from_row(row),
                "metadata": row["metadata"] or {},
                "score": float(row["score"]),
            }
            for row in rows
        ]

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    def get_entity_relationships(self, entity_id: int, depth: int = 1) -> Dict:
        """Get N-hop neighborhood of an entity using recursive CTE."""
        depth = min(depth, 5)  # Cap depth to prevent runaway recursive CTEs
        return _run_async_safe(
            self.async_get_entity_relationships(entity_id, depth),
            operation_name="get_entity_relationships",
        )

    async def async_get_entity_relationships(self, entity_id: int, depth: int) -> Dict:
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                # Get the root entity
                root = await conn.fetchrow(
                    f"SELECT {_ENTITY_COLS} FROM graph.entities WHERE entity_id = $1",
                    entity_id,
                )
                if not root:
                    return {"entity": None, "relationships": [], "connected_entities": []}

                # Recursive CTE for N-hop traversal.
                # Uses a visited array per traversal path to avoid cycles.
                # The final DISTINCT ON deduplicates relationships found via multiple paths.
                sql = """
                    WITH RECURSIVE traversal AS (
                        -- Base: direct relationships from/to root entity
                        SELECT
                            r.relationship_id, r.source_entity_id, r.target_entity_id,
                            r.relationship_type, r.description AS rel_description, r.weight,
                            1 AS hop,
                            ARRAY[$1::int] || ARRAY[
                                CASE WHEN r.source_entity_id = $1::int
                                     THEN r.target_entity_id
                                     ELSE r.source_entity_id END
                            ] AS visited
                        FROM graph.relationships r
                        WHERE r.source_entity_id = $1::int OR r.target_entity_id = $1::int

                        UNION ALL

                        -- Recursive: follow edges from frontier entities (not yet visited)
                        SELECT
                            r.relationship_id, r.source_entity_id, r.target_entity_id,
                            r.relationship_type, r.description, r.weight,
                            t.hop + 1,
                            t.visited || ARRAY[
                                CASE WHEN r.source_entity_id = ANY(t.visited)
                                     THEN r.target_entity_id
                                     ELSE r.source_entity_id END
                            ]
                        FROM graph.relationships r
                        JOIN traversal t ON (
                            (r.source_entity_id = ANY(t.visited) AND NOT r.target_entity_id = ANY(t.visited))
                            OR (r.target_entity_id = ANY(t.visited) AND NOT r.source_entity_id = ANY(t.visited))
                        )
                        WHERE t.hop < $2
                    )
                    SELECT DISTINCT ON (relationship_id)
                        relationship_id, source_entity_id, target_entity_id,
                        relationship_type, rel_description, weight, hop
                    FROM traversal
                    ORDER BY relationship_id, hop
                """
                rel_rows = await conn.fetch(sql, entity_id, depth)

                # Collect all entity IDs (always non-empty since entity_id is included)
                entity_ids = {entity_id}
                for row in rel_rows:
                    entity_ids.add(row["source_entity_id"])
                    entity_ids.add(row["target_entity_id"])

                # Fetch entity details
                entity_rows = await conn.fetch(
                    f"SELECT {_ENTITY_COLS} FROM graph.entities "
                    f"WHERE entity_id IN ({_in_placeholders(entity_ids)})",
                    *entity_ids,
                )

                entities_map = {row["entity_id"]: _entity_from_row(row) for row in entity_rows}
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Graph traversal failed for entity {entity_id}: {e}", cause=e
            ) from e

        relationships = [
            {
                "source": entities_map.get(row["source_entity_id"], {}).get("name", "?"),
                "target": entities_map.get(row["target_entity_id"], {}).get("name", "?"),
                "type": row["relationship_type"],
                "description": row["rel_description"],
                "weight": float(row["weight"]),
                "hop": row["hop"],
            }
            for row in rel_rows
        ]

        connected = [e for eid, e in entities_map.items() if eid != entity_id]

        return {
            "entity": entities_map.get(entity_id),
            "relationships": relationships,
            "connected_entities": connected,
        }

    # =========================================================================
    # Community Operations
    # =========================================================================

    def save_communities(self, communities: List[Dict]) -> int:
        """Store Leiden community detection results."""
        return _run_async_safe(
            self.async_save_communities(communities),
            operation_name="save_communities",
        )

    async def async_save_communities(self, communities: List[Dict]) -> int:
        if not communities:
            return 0
        await self._ensure_pool()

        records = [
            (
                c["level"],
                c.get("title"),
                c.get("summary"),
                c.get("summary_model"),
                c["entity_ids"],
                json.dumps(c.get("metadata", {})),
            )
            for c in communities
        ]

        try:
            async with self.pool.acquire() as conn:
                # Atomic delete ALL + insert within a transaction.
                # If insert fails, the transaction rolls back and old communities are preserved.
                async with conn.transaction():
                    await conn.execute("DELETE FROM graph.communities")
                    await conn.executemany(
                        """
                        INSERT INTO graph.communities (level, title, summary, summary_model, entity_ids, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                        """,
                        records,
                    )
        except asyncpg.PostgresError as e:
            raise GraphStoreError(f"Failed to save {len(records)} communities: {e}", cause=e) from e

        return len(records)

    def get_communities(self, level: int = 0) -> List[Dict]:
        """Get all communities at a given level."""
        return _run_async_safe(
            self.async_get_communities(level),
            operation_name="get_communities",
        )

    async def async_get_communities(self, level: int) -> List[Dict]:
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT community_id, level, title, summary, summary_model, entity_ids, metadata
                    FROM graph.communities
                    WHERE level = $1
                    ORDER BY array_length(entity_ids, 1) DESC NULLS LAST
                    """,
                    level,
                )
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to get communities at level {level}: {e}", cause=e
            ) from e

        return [_community_from_row(row) for row in rows]

    def search_communities(self, query: str, level: int = 0, limit: int = 10) -> List[Dict]:
        """Search communities by embedding similarity (or FTS fallback)."""
        return _run_async_safe(
            self.async_search_communities(query, level, limit),
            operation_name="search_communities",
        )

    async def async_search_communities(self, query: str, level: int, limit: int) -> List[Dict]:
        await self._ensure_pool()

        if self._embedder:
            return await self._search_communities_by_embedding(query, level, limit)
        logger.debug("No embedder configured, falling back to FTS for community search")
        return await self._search_communities_by_fts(query, level, limit)

    async def _search_communities_by_embedding(
        self, query: str, level: int, limit: int
    ) -> List[Dict]:
        vec = await asyncio.to_thread(self._embedder.encode_query, query)
        query_vec = _vec_to_pg(vec)
        sql = """
            SELECT community_id, level, title, summary, summary_model, entity_ids, metadata,
                   1 - (search_embedding <=> $1::vector) AS score
            FROM graph.communities
            WHERE search_embedding IS NOT NULL AND level = $2
            ORDER BY search_embedding <=> $1::vector
            LIMIT $3
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, query_vec, level, limit)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Community embedding search failed for query '{query}': {e}", cause=e
            ) from e

        return [_community_from_row(row) for row in rows]

    async def _search_communities_by_fts(self, query: str, level: int, limit: int) -> List[Dict]:
        sql = """
            SELECT community_id, level, title, summary, summary_model, entity_ids, metadata,
                   ts_rank(search_tsv, plainto_tsquery('simple', unaccent($1))) AS score
            FROM graph.communities
            WHERE search_tsv @@ plainto_tsquery('simple', unaccent($1))
              AND level = $2
            ORDER BY score DESC
            LIMIT $3
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, query, level, limit)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Community FTS search failed for query '{query}': {e}", cause=e
            ) from e

        return [_community_from_row(row) for row in rows]

    def get_community_entities(self, community_id: int) -> List[Dict]:
        """Get entity details for a community."""
        return _run_async_safe(
            self.async_get_community_entities(community_id),
            operation_name="get_community_entities",
        )

    async def async_get_community_entities(self, community_id: int) -> List[Dict]:
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT entity_ids FROM graph.communities WHERE community_id = $1",
                    community_id,
                )
                if not row or not row["entity_ids"]:
                    return []

                entity_ids = row["entity_ids"]
                entity_rows = await conn.fetch(
                    f"SELECT {_ENTITY_COLS} FROM graph.entities "
                    f"WHERE entity_id IN ({_in_placeholders(entity_ids)})",
                    *entity_ids,
                )
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to get entities for community {community_id}: {e}", cause=e
            ) from e

        return [_entity_from_row(row) for row in entity_rows]

    # =========================================================================
    # Document Management
    # =========================================================================

    def delete_document_graph(self, document_id: str) -> int:
        """Delete all graph data for a document. Returns deleted entity count."""
        return _run_async_safe(
            self.async_delete_document_graph(document_id),
            operation_name="delete_document_graph",
        )

    async def async_delete_document_graph(self, document_id: str) -> int:
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                # Relationships cascade-delete with entities
                result = await conn.execute(
                    "DELETE FROM graph.entities WHERE document_id = $1",
                    document_id,
                )
                # asyncpg returns e.g. "DELETE 5" — parse defensively
                try:
                    count = int(result.split()[-1])
                except (ValueError, IndexError):
                    logger.warning(f"Unexpected DELETE result format: {result!r}")
                    count = 0
                if count > 0:
                    logger.info(f"Deleted {count} entities for document {document_id}")
                return count
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to delete graph for document {document_id}: {e}", cause=e
            ) from e

    # =========================================================================
    # Bulk Access (for igraph construction)
    # =========================================================================

    def get_all_entities_and_relationships(self) -> tuple:
        """Get all entities and relationships for igraph construction."""
        return _run_async_safe(
            self.async_get_all(),
            operation_name="get_all_entities_and_relationships",
        )

    async def async_get_all(self) -> tuple:
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                entity_rows = await conn.fetch(
                    f"SELECT {_ENTITY_COLS} FROM graph.entities ORDER BY entity_id"
                )
                rel_rows = await conn.fetch(
                    "SELECT relationship_id, source_entity_id, target_entity_id, "
                    "relationship_type, weight FROM graph.relationships"
                )
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to fetch all entities and relationships: {e}", cause=e
            ) from e

        entities = [_entity_from_row(r) for r in entity_rows]
        relationships = [
            {
                "relationship_id": r["relationship_id"],
                "source_entity_id": r["source_entity_id"],
                "target_entity_id": r["target_entity_id"],
                "relationship_type": r["relationship_type"],
                "weight": float(r["weight"]),
            }
            for r in rel_rows
        ]

        return entities, relationships

    # =========================================================================
    # Stats
    # =========================================================================

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return _run_async_safe(
            self.async_get_graph_stats(),
            operation_name="get_graph_stats",
        )

    async def async_get_graph_stats(self) -> Dict[str, Any]:
        await self._ensure_pool()
        try:
            async with self.pool.acquire() as conn:
                entity_count = await conn.fetchval("SELECT COUNT(*) FROM graph.entities")
                rel_count = await conn.fetchval("SELECT COUNT(*) FROM graph.relationships")
                community_count = await conn.fetchval("SELECT COUNT(*) FROM graph.communities")
                doc_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT document_id) FROM graph.entities"
                )
        except asyncpg.PostgresError as e:
            raise GraphStoreError(f"Failed to get graph stats: {e}", cause=e) from e

        return {
            "entities": entity_count,
            "relationships": rel_count,
            "communities": community_count,
            "documents_with_graph": doc_count,
        }

    # =========================================================================
    # Entity Deduplication
    # =========================================================================

    async def _merge_entity_group(
        self,
        conn,
        canonical_id: int,
        duplicate_ids: List[int],
        merged_description: Optional[str] = None,
        canonical_name: Optional[str] = None,
    ) -> int:
        """Merge duplicate entities into canonical within an open transaction.

        Remaps relationships, removes self-references and duplicate edges,
        updates canonical entity, and deletes duplicates.

        Args:
            conn: Active asyncpg connection (must be inside a transaction).
            canonical_id: Entity ID to keep.
            duplicate_ids: Entity IDs to merge into canonical and delete.
            merged_description: If provided, set as canonical's description.
                When None, only the search_embedding is cleared.
            canonical_name: If provided, rename canonical entity to this name.

        Returns:
            Number of duplicate entity rows deleted.
        """
        # Remap relationships from duplicates to canonical
        for dup_id in duplicate_ids:
            await conn.execute(
                "UPDATE graph.relationships SET source_entity_id = $1 WHERE source_entity_id = $2",
                canonical_id,
                dup_id,
            )
            await conn.execute(
                "UPDATE graph.relationships SET target_entity_id = $1 WHERE target_entity_id = $2",
                canonical_id,
                dup_id,
            )

        # Remove self-referencing relationships created by remap (scoped to canonical)
        await conn.execute(
            "DELETE FROM graph.relationships "
            "WHERE source_entity_id = target_entity_id AND source_entity_id = $1",
            canonical_id,
        )

        # Deduplicate relationship edges involving canonical (keep lowest relationship_id)
        await conn.execute(
            """
            DELETE FROM graph.relationships r1
            USING graph.relationships r2
            WHERE r1.source_entity_id = r2.source_entity_id
              AND r1.target_entity_id = r2.target_entity_id
              AND r1.relationship_type = r2.relationship_type
              AND r1.relationship_id > r2.relationship_id
              AND (r1.source_entity_id = $1 OR r1.target_entity_id = $1)
            """,
            canonical_id,
        )

        # Update canonical: set name/description if provided, always clear embedding
        update_parts = ["search_embedding = NULL"]
        params: List[Any] = []
        if canonical_name is not None:
            params.append(canonical_name)
            update_parts.append(f"name = ${len(params)}")
        if merged_description is not None:
            params.append(merged_description)
            update_parts.append(f"description = ${len(params)}")
        params.append(canonical_id)
        await conn.execute(
            f"UPDATE graph.entities SET {', '.join(update_parts)} WHERE entity_id = ${len(params)}",
            *params,
        )

        # Delete duplicate entity rows
        placeholders = ", ".join(f"${i + 1}" for i in range(len(duplicate_ids)))
        result = await conn.execute(
            f"DELETE FROM graph.entities WHERE entity_id IN ({placeholders})",
            *duplicate_ids,
        )
        return _parse_command_count(result)

    def deduplicate_exact(self) -> Dict:
        """Merge entities with identical (name, entity_type) across documents (sync)."""
        return _run_async_safe(
            self.async_deduplicate_exact(),
            timeout=120.0,
            operation_name="deduplicate_exact",
        )

    async def async_deduplicate_exact(self) -> Dict:
        """
        Merge entities with identical (name, entity_type) across different documents.

        For each duplicate group, delegates to ``_merge_entity_group`` which
        remaps relationships, removes self-references and duplicate edges,
        updates the canonical entity description, and deletes duplicates.

        Returns:
            Dict with merge stats: groups_merged, entities_removed
        """
        await self._ensure_pool()

        # Find groups with identical (name, entity_type) — case-insensitive, including within same document
        find_groups_sql = """
            SELECT lower(name) AS name_key, entity_type,
                   array_agg(entity_id ORDER BY length(coalesce(description, '')) DESC, entity_id ASC) AS entity_ids,
                   array_agg(coalesce(description, '') ORDER BY length(coalesce(description, '')) DESC, entity_id ASC) AS descriptions,
                   (array_agg(name ORDER BY length(coalesce(description, '')) DESC, entity_id ASC))[1] AS canonical_name
            FROM graph.entities
            GROUP BY lower(name), entity_type
            HAVING COUNT(*) > 1
        """

        try:
            async with self.pool.acquire() as conn:
                groups = await conn.fetch(find_groups_sql)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(f"Failed to find duplicate entity groups: {e}", cause=e) from e

        if not groups:
            logger.info("No exact duplicates found")
            return {"groups_merged": 0, "entities_removed": 0}

        logger.info(f"Found {len(groups)} exact duplicate groups to merge")

        total_removed = 0
        groups_succeeded = 0

        for group in groups:
            entity_ids = list(group["entity_ids"])
            descriptions = list(group["descriptions"])
            canonical_id = entity_ids[0]
            duplicate_ids = entity_ids[1:]
            canonical_name = group.get("canonical_name")

            if not duplicate_ids:
                continue

            # Build merged description (concatenate distinct, max 2000 chars)
            distinct_descs = list(dict.fromkeys(d for d in descriptions if d))
            merged_desc = " | ".join(distinct_descs)[:2000] if distinct_descs else None

            try:
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        removed = await self._merge_entity_group(
                            conn,
                            canonical_id,
                            duplicate_ids,
                            merged_description=merged_desc,
                            canonical_name=canonical_name,
                        )
                        total_removed += removed
                        groups_succeeded += 1

            except asyncpg.PostgresError as e:
                logger.error(
                    f"Failed to merge exact duplicates for '{group['name_key']}' ({group['entity_type']}): {e}",
                    exc_info=True,
                )
                # Continue with next group -- partial failure doesn't corrupt

        stats = {
            "groups_merged": groups_succeeded,
            "entities_removed": total_removed,
        }
        logger.info(f"Exact dedup complete: {stats}")
        return stats

    def deduplicate_semantic(
        self, similarity_threshold: float = 0.85, llm_provider: Optional[Any] = None
    ) -> Dict:
        """Merge semantically similar entities using embedding NN + LLM arbiter (sync)."""
        return _run_async_safe(
            self.async_deduplicate_semantic(similarity_threshold, llm_provider),
            timeout=300.0,
            operation_name="deduplicate_semantic",
        )

    async def async_deduplicate_semantic(
        self, similarity_threshold: float = 0.85, llm_provider: Optional[Any] = None
    ) -> Dict:
        """
        Merge semantically similar entities using embedding nearest-neighbor + LLM arbiter.

        1. Find candidate pairs above similarity threshold via pgvector KNN
           (same entity_type required, uses CROSS JOIN LATERAL for batch NN)
        2. Ask LLM to confirm each candidate pair
        3. Build Union-Find from confirmed pairs (transitive closure)
        4. Merge each group using same logic as exact dedup

        Args:
            similarity_threshold: Cosine similarity threshold for candidate pairs
            llm_provider: LLM provider for arbitration (required for actual merges)

        Returns:
            Dict with merge stats
        """
        await self._ensure_pool()

        if not llm_provider:
            logger.info("No LLM provider for semantic dedup — skipping")
            return {"candidates_found": 0, "llm_confirmed": 0, "entities_removed": 0}

        # Load dedup prompt (try relative to source first, then cwd)
        prompt_path = (
            Path(__file__).resolve().parent.parent.parent / "prompts" / "graph_entity_dedup.txt"
        )
        if not prompt_path.exists():
            prompt_path = Path("prompts") / "graph_entity_dedup.txt"
        if not prompt_path.exists():
            logger.error(f"Dedup prompt not found at {prompt_path}, skipping semantic dedup")
            return {
                "candidates_found": 0,
                "llm_confirmed": 0,
                "entities_removed": 0,
                "error": f"Prompt file not found: {prompt_path}",
            }
        prompt_template = prompt_path.read_text(encoding="utf-8")

        # Find candidate pairs via pgvector KNN (batch nearest-neighbor)
        candidate_sql = """
            SELECT e1.entity_id AS id1, e2.entity_id AS id2,
                   e1.name AS name1, e2.name AS name2,
                   e1.entity_type AS type1, e2.entity_type AS type2,
                   e1.description AS desc1, e2.description AS desc2,
                   1 - (e1.search_embedding <=> e2.search_embedding) AS similarity
            FROM graph.entities e1
            CROSS JOIN LATERAL (
                SELECT entity_id, name, entity_type, description, search_embedding
                FROM graph.entities e2
                WHERE e2.entity_type = e1.entity_type
                  AND e2.entity_id > e1.entity_id
                  AND e2.search_embedding IS NOT NULL
                ORDER BY e2.search_embedding <=> e1.search_embedding
                LIMIT 5
            ) e2
            WHERE e1.search_embedding IS NOT NULL
              AND 1 - (e1.search_embedding <=> e2.search_embedding) >= $1
        """

        try:
            async with self.pool.acquire() as conn:
                candidates = await conn.fetch(candidate_sql, similarity_threshold)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Failed to find semantic duplicate candidates: {e}", cause=e
            ) from e

        if not candidates:
            logger.info("No semantic duplicate candidates found")
            return {"candidates_found": 0, "llm_confirmed": 0, "entities_removed": 0}

        logger.info(
            f"Found {len(candidates)} semantic duplicate candidates (threshold={similarity_threshold})"
        )

        # LLM arbitration for each candidate pair
        confirmed_pairs = []
        for cand in candidates:
            prompt = prompt_template.format(
                name1=cand["name1"],
                type1=cand["type1"],
                desc1=cand["desc1"] or "No description",
                name2=cand["name2"],
                type2=cand["type2"],
                desc2=cand["desc2"] or "No description",
            )

            try:
                response = await asyncio.to_thread(
                    llm_provider.create_message,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                    system="",
                    max_tokens=100,
                    temperature=0.0,
                )
                answer = (response.text or "").strip()
                is_yes = _parse_dedup_verdict(answer)

                if is_yes:
                    confirmed_pairs.append((cand["id1"], cand["id2"]))
                    logger.info(
                        f"LLM confirmed merge: '{cand['name1']}' ≈ '{cand['name2']}' "
                        f"(sim={cand['similarity']:.3f})"
                    )
                else:
                    logger.debug(
                        f"LLM rejected merge: '{cand['name1']}' vs '{cand['name2']}' "
                        f"(sim={cand['similarity']:.3f}): {answer[:80]}"
                    )
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception as e:
                logger.warning(
                    f"LLM arbitration failed for '{cand['name1']}' vs '{cand['name2']}': {e}",
                    exc_info=True,
                )

        if not confirmed_pairs:
            logger.info("No semantic duplicates confirmed by LLM")
            return {
                "candidates_found": len(candidates),
                "llm_confirmed": 0,
                "entities_removed": 0,
            }

        # Build Union-Find from confirmed pairs for transitive closure
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # Lower ID becomes root (canonical)
                if ra > rb:
                    ra, rb = rb, ra
                parent[rb] = ra

        for a, b in confirmed_pairs:
            union(a, b)

        # Group by canonical
        merge_groups: Dict[int, List[int]] = {}
        all_ids = set()
        for a, b in confirmed_pairs:
            all_ids.add(a)
            all_ids.add(b)
        for eid in all_ids:
            root = find(eid)
            merge_groups.setdefault(root, []).append(eid)

        # Deduplicate and sort each group
        for root in merge_groups:
            merge_groups[root] = sorted(set(merge_groups[root]))

        # Fetch entity names/descriptions for canonical name selection
        all_entity_ids = set()
        for group_ids in merge_groups.values():
            all_entity_ids.update(group_ids)
        entity_info: Dict[int, Dict] = {}
        if all_entity_ids:
            placeholders = ", ".join(f"${i + 1}" for i in range(len(all_entity_ids)))
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT entity_id, name, coalesce(description, '') AS description "
                    f"FROM graph.entities WHERE entity_id IN ({placeholders})",
                    *sorted(all_entity_ids),
                )
            entity_info = {r["entity_id"]: dict(r) for r in rows}

        # Merge each group using shared merge logic
        total_removed = 0
        groups_succeeded = 0
        for canonical_id, group_ids in merge_groups.items():
            duplicate_ids = [eid for eid in group_ids if eid != canonical_id]
            if not duplicate_ids:
                continue

            # Pick the longest name as canonical (prefer full name over abbreviation)
            best_name = max(
                (entity_info[eid]["name"] for eid in group_ids if eid in entity_info),
                key=len,
                default=None,
            )

            # Merge descriptions
            descs = [entity_info[eid]["description"] for eid in group_ids if eid in entity_info and entity_info[eid]["description"]]
            distinct_descs = list(dict.fromkeys(descs))
            merged_desc = " | ".join(distinct_descs)[:2000] if distinct_descs else None

            try:
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        removed = await self._merge_entity_group(
                            conn,
                            canonical_id,
                            duplicate_ids,
                            merged_description=merged_desc,
                            canonical_name=best_name,
                        )
                        total_removed += removed
                        groups_succeeded += 1

            except asyncpg.PostgresError as e:
                logger.error(
                    f"Failed to merge semantic group (canonical={canonical_id}): {e}", exc_info=True
                )

        stats = {
            "candidates_found": len(candidates),
            "llm_confirmed": len(confirmed_pairs),
            "groups_merged": groups_succeeded,
            "entities_removed": total_removed,
        }
        logger.info(f"Semantic dedup complete: {stats}")
        return stats
