"""
Storage Abstraction Layer for SUJBOT

Provides unified interface for PostgreSQL pgvector storage (VL-only).

Key Components:
- VectorStoreAdapter: Abstract interface for vector stores
- PostgresVectorStoreAdapter: PostgreSQL + pgvector implementation

Usage:
    from src.storage import create_vector_store_adapter

    # Create adapter
    adapter = create_vector_store_adapter(
        backend="postgresql",
        connection_string="postgresql://..."
    )

    # VL page search
    results = adapter.search_vl_pages(query_emb, k=5)
"""

from .vector_store_adapter import VectorStoreAdapter

# PostgreSQL adapter (requires asyncpg)
try:
    from .postgres_adapter import PostgresVectorStoreAdapter
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    PostgresVectorStoreAdapter = None

__all__ = [
    "VectorStoreAdapter",
    "PostgresVectorStoreAdapter",
    "create_vector_store_adapter",
    "load_vector_store_adapter",
]


def _validate_postgresql_backend(**kwargs) -> tuple:
    """
    Validate PostgreSQL backend configuration.

    Returns:
        Tuple of (connection_string, pool_size, dimensions)

    Raises:
        ImportError: If asyncpg not installed
        ValueError: If connection_string missing
    """
    if not POSTGRES_AVAILABLE:
        raise ImportError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install asyncpg"
        )

    connection_string = kwargs.get("connection_string") or kwargs.get("path")
    if not connection_string:
        raise ValueError("PostgreSQL backend requires 'connection_string' argument")

    pool_size = kwargs.get("pool_size", 20)
    dimensions = kwargs.get("dimensions", 2048)  # Jina v4

    return connection_string, pool_size, dimensions


def create_vector_store_adapter(backend: str = "postgresql", **kwargs) -> VectorStoreAdapter:
    """
    Factory function to create vector store adapter.

    Args:
        backend: Backend type (only "postgresql" supported)
        **kwargs: Backend-specific arguments
            - connection_string: PostgreSQL connection string
            - pool_size: Connection pool size (default: 20)
            - dimensions: Embedding dimensions (default: 2048 for Jina v4)

    Returns:
        VectorStoreAdapter instance
    """
    if backend != "postgresql":
        raise ValueError(f"Unknown backend: {backend}. Only 'postgresql' is supported.")

    connection_string, pool_size, dimensions = _validate_postgresql_backend(**kwargs)
    return PostgresVectorStoreAdapter(
        connection_string=connection_string,
        pool_size=pool_size,
        dimensions=dimensions,
    )


async def load_vector_store_adapter(backend: str = "postgresql", path: str = None, **kwargs) -> VectorStoreAdapter:
    """
    Load vector store adapter from persistent storage.

    Args:
        backend: Backend type (only "postgresql" supported)
        path: Unused for PostgreSQL (kept for interface compatibility)
        **kwargs: Additional backend-specific arguments
            - connection_string: PostgreSQL connection string
            - pool_size: Connection pool size (default: 20)
            - dimensions: Embedding dimensions (default: 2048)

    Returns:
        VectorStoreAdapter instance
    """
    if backend != "postgresql":
        raise ValueError(f"Unknown backend: {backend}. Only 'postgresql' is supported.")

    # Allow path as fallback for connection_string (interface compatibility)
    if path and "connection_string" not in kwargs:
        kwargs["connection_string"] = path

    # Drop legacy architecture param if passed
    kwargs.pop("architecture", None)
    connection_string, pool_size, dimensions = _validate_postgresql_backend(**kwargs)
    adapter = PostgresVectorStoreAdapter(
        connection_string=connection_string,
        pool_size=pool_size,
        dimensions=dimensions,
    )
    await adapter.initialize()
    return adapter
