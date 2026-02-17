"""
Shared async utilities.

Provides run_async_safe() for safely calling async code from sync context,
and vec_to_pgvector() for converting numpy arrays to pgvector string format.
"""

import asyncio
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_async_safe(
    coro: Any,
    timeout: float = 30.0,
    operation_name: str = "async operation",
) -> Any:
    """
    Safely run async coroutine from sync context.

    Handles two scenarios:
    1. No running event loop: Uses asyncio.run() directly
    2. Already in async context: Uses nest_asyncio (applied at startup in backend/main.py)

    Args:
        coro: Async coroutine to execute
        timeout: Timeout in seconds (default: 30)
        operation_name: Name of operation for error messages

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If execution exceeds timeout
        RuntimeError: If called inside a running loop without nest_asyncio
    """
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


def vec_to_pgvector(vec: np.ndarray) -> str:
    """Convert numpy array to pgvector string format '[0.1,0.2,...]'."""
    return "[" + ",".join(map(str, vec.flatten().tolist())) + "]"
