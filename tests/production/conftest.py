"""
Shared fixtures for production tests.

These fixtures provide authenticated HTTP clients for testing production services.
"""

import os
import pytest
import httpx
from typing import Generator


# Configuration from environment
PROD_BASE_URL = os.getenv("PROD_BASE_URL", "http://localhost:8000")
PROD_TEST_USER = os.getenv("PROD_TEST_USER", "test@example.com")
PROD_TEST_PASSWORD = os.getenv("PROD_TEST_PASSWORD", "TestPassword123!")


@pytest.fixture(scope="session")
def base_url() -> str:
    """Get base URL for API."""
    return PROD_BASE_URL


@pytest.fixture(scope="session")
def http_client(base_url: str) -> Generator[httpx.Client, None, None]:
    """Create HTTP client for API requests (unauthenticated)."""
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def auth_cookies(http_client: httpx.Client) -> dict:
    """
    Get authentication cookies by logging in.

    Returns empty dict if login fails (tests will be skipped).
    """
    try:
        response = http_client.post(
            "/auth/login",
            json={"email": PROD_TEST_USER, "password": PROD_TEST_PASSWORD}
        )
        if response.status_code == 200:
            return dict(response.cookies)
    except httpx.HTTPError:
        pass
    return {}


@pytest.fixture(scope="session")
def auth_client(base_url: str, auth_cookies: dict) -> Generator[httpx.Client, None, None]:
    """Create authenticated HTTP client."""
    with httpx.Client(
        base_url=base_url,
        timeout=30.0,
        cookies=auth_cookies
    ) as client:
        yield client


@pytest.fixture
def requires_auth(auth_cookies: dict):
    """Skip test if authentication failed."""
    if not auth_cookies:
        pytest.skip("Authentication required - set PROD_TEST_USER and PROD_TEST_PASSWORD")


@pytest.fixture
def requires_services(http_client: httpx.Client):
    """Skip test if services are not running."""
    try:
        response = http_client.get("/health", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Services not healthy: {response.status_code}")
    except httpx.HTTPError as e:
        pytest.skip(f"Services not running: {e}")
