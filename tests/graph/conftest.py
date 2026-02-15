import pytest


@pytest.fixture
def anyio_backend():
    """Restrict anyio to asyncio (trio not installed)."""
    return "asyncio"
