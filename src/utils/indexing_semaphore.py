"""
Indexing Concurrency Semaphore — reserves model capacity for production queries.

Limits the number of concurrent indexing requests to local vLLM models so that
production user queries always have available KV cache slots.

Usage:
    sem = get_indexing_semaphore()
    async with sem.for_provider("local_llm"):
        response = await asyncio.to_thread(provider.create_message, ...)
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class _NoOpSemaphore:
    """Context manager that never blocks — used for remote providers."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


_NOOP = _NoOpSemaphore()


class IndexingSemaphore:
    """
    Limits indexing concurrency to reserve model capacity for production queries.

    The 30B model has 8 vLLM slots total. With max_30b=2, indexing uses at most 2,
    leaving 6 for user queries. The 8B model has 16 slots, so max_8b=8 leaves 8
    for routing.
    """

    def __init__(self, max_30b: int = 2, max_8b: int = 8):
        self._sem_30b = asyncio.Semaphore(max_30b)
        self._sem_8b = asyncio.Semaphore(max_8b)
        self._max_30b = max_30b
        self._max_8b = max_8b
        logger.info(
            "IndexingSemaphore initialized: 30B=%d slots, 8B=%d slots",
            max_30b,
            max_8b,
        )

    def for_provider(self, provider_name: str) -> "_NoOpSemaphore | asyncio.Semaphore":
        """Return the appropriate semaphore for a provider.

        Args:
            provider_name: Provider identifier ('local_llm', 'local_llm_8b', 'anthropic', etc.)

        Returns:
            Semaphore for local providers, no-op for remote providers.
        """
        if provider_name == "local_llm":
            return self._sem_30b
        if provider_name == "local_llm_8b":
            return self._sem_8b
        return _NOOP


# Global singleton
_semaphore: Optional[IndexingSemaphore] = None


def get_indexing_semaphore() -> IndexingSemaphore:
    """Get the global indexing semaphore (lazy-creates with defaults if not set)."""
    global _semaphore
    if _semaphore is None:
        _semaphore = IndexingSemaphore()
    return _semaphore


def set_indexing_semaphore(semaphore: IndexingSemaphore) -> None:
    """Set the global indexing semaphore (called from backend startup)."""
    global _semaphore
    _semaphore = semaphore
