"""
Graph Storage Adapter — PostgreSQL backend for knowledge graph.

Stores entities, relationships, and communities in the `graph` schema.
Follows PostgresVectorStoreAdapter pattern: asyncpg pool + sync wrappers.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg

from ..exceptions import DatabaseConnectionError, GraphStoreError

logger = logging.getLogger(__name__)

# Column lists reused across entity queries
_ENTITY_COLS = "entity_id, name, entity_type, description, document_id"


def _entity_from_row(row: asyncpg.Record) -> Dict[str, Any]:
    """Convert an asyncpg Record to an entity dict. Shared by all entity queries."""
    return {
        "entity_id": row["entity_id"],
        "name": row["name"],
        "entity_type": row["entity_type"],
        "description": row["description"],
        "document_id": row["document_id"],
    }


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


class GraphStorageAdapter:
    """
    PostgreSQL storage for knowledge graph (entities, relationships, communities).

    Uses asyncpg connection pool. Can share pool with PostgresVectorStoreAdapter
    or create its own.
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None, connection_string: Optional[str] = None):
        """
        Initialize with existing pool or connection string.

        Args:
            pool: Existing asyncpg pool (preferred — avoids duplicate connections).
                When provided, this adapter does NOT own the pool (close() is a no-op).
            connection_string: PostgreSQL DSN (creates new pool if pool is None).
                When pool is None and connection_string is provided, this adapter
                creates and owns a new pool (close() will terminate it).
        """
        self.pool = pool
        self._connection_string = connection_string
        self._owns_pool = pool is None

    async def initialize(self):
        """Create connection pool if not provided."""
        await self._ensure_pool()

    async def _ensure_pool(self):
        """Lazily create connection pool on first use."""
        if self.pool is None:
            if not self._connection_string:
                raise DatabaseConnectionError(
                    "Either pool or connection_string must be provided"
                )
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

    def add_entities(self, entities: List[Dict], document_id: str, source_page_id: Optional[str] = None) -> int:
        """Bulk upsert entities. Returns count of records processed (attempted, not DB-reported)."""
        return _run_async_safe(
            self._async_add_entities(entities, document_id, source_page_id),
            operation_name="add_entities",
        )

    async def _async_add_entities(
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
        """Bulk insert relationships. Resolves entity names to IDs."""
        return _run_async_safe(
            self._async_add_relationships(relationships, document_id, source_page_id),
            operation_name="add_relationships",
        )

    @staticmethod
    async def _resolve_entity_id(
        conn: asyncpg.Connection, name: str, document_id: str
    ) -> Optional[int]:
        """Resolve entity name to ID: document-scoped first, then cross-document fallback."""
        entity_id = await conn.fetchval(
            "SELECT entity_id FROM graph.entities WHERE name = $1 AND document_id = $2 LIMIT 1",
            name, document_id,
        )
        if entity_id is None:
            entity_id = await conn.fetchval(
                "SELECT entity_id FROM graph.entities WHERE name = $1 LIMIT 1", name
            )
        return entity_id

    async def _async_add_relationships(
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
        """Search entities by name using trigram similarity."""
        return _run_async_safe(
            self._async_search_entities(query, entity_type, limit),
            operation_name="search_entities",
        )

    async def _async_search_entities(
        self, query: str, entity_type: Optional[str], limit: int
    ) -> List[Dict]:
        await self._ensure_pool()
        params: list = [query]
        where_parts = ["similarity(name, $1) > 0.1"]
        param_idx = 1

        if entity_type:
            param_idx += 1
            where_parts.append(f"entity_type = ${param_idx}")
            params.append(entity_type)

        param_idx += 1
        params.append(limit)

        sql = f"""
            SELECT {_ENTITY_COLS}, metadata, similarity(name, $1) AS sim
            FROM graph.entities
            WHERE {' AND '.join(where_parts)}
            ORDER BY sim DESC
            LIMIT ${param_idx}
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except asyncpg.PostgresError as e:
            raise GraphStoreError(
                f"Entity search failed for query '{query}': {e}", cause=e
            ) from e

        return [
            {**_entity_from_row(row), "metadata": row["metadata"] or {}, "similarity": float(row["sim"])}
            for row in rows
        ]

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    def get_entity_relationships(self, entity_id: int, depth: int = 1) -> Dict:
        """Get N-hop neighborhood of an entity using recursive CTE."""
        depth = min(depth, 5)  # Cap depth to prevent runaway recursive CTEs
        return _run_async_safe(
            self._async_get_entity_relationships(entity_id, depth),
            operation_name="get_entity_relationships",
        )

    async def _async_get_entity_relationships(self, entity_id: int, depth: int) -> Dict:
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
            self._async_save_communities(communities),
            operation_name="save_communities",
        )

    async def _async_save_communities(self, communities: List[Dict]) -> int:
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
            raise GraphStoreError(
                f"Failed to save {len(records)} communities: {e}", cause=e
            ) from e

        return len(records)

    def get_communities(self, level: int = 0) -> List[Dict]:
        """Get all communities at a given level."""
        return _run_async_safe(
            self._async_get_communities(level),
            operation_name="get_communities",
        )

    async def _async_get_communities(self, level: int) -> List[Dict]:
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

        return [
            {
                "community_id": row["community_id"],
                "level": row["level"],
                "title": row["title"],
                "summary": row["summary"],
                "entity_ids": row["entity_ids"],
                "metadata": row["metadata"] or {},
            }
            for row in rows
        ]

    def get_community_entities(self, community_id: int) -> List[Dict]:
        """Get entity details for a community."""
        return _run_async_safe(
            self._async_get_community_entities(community_id),
            operation_name="get_community_entities",
        )

    async def _async_get_community_entities(self, community_id: int) -> List[Dict]:
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
            self._async_delete_document_graph(document_id),
            operation_name="delete_document_graph",
        )

    async def _async_delete_document_graph(self, document_id: str) -> int:
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
            self._async_get_all(),
            operation_name="get_all_entities_and_relationships",
        )

    async def _async_get_all(self) -> tuple:
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
            self._async_get_graph_stats(),
            operation_name="get_graph_stats",
        )

    async def _async_get_graph_stats(self) -> Dict[str, Any]:
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
            raise GraphStoreError(
                f"Failed to get graph stats: {e}", cause=e
            ) from e

        return {
            "entities": entity_count,
            "relationships": rel_count,
            "communities": community_count,
            "documents_with_graph": doc_count,
        }
