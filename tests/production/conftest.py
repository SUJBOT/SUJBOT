"""
Shared fixtures for production tests.

These fixtures provide authenticated HTTP clients for testing production services.
"""

import json
import os
import subprocess
import warnings
import pytest
import httpx
from typing import Generator, Optional


# Configuration from environment (NO DEFAULT PASSWORD for security)
PROD_BASE_URL = os.getenv("PROD_BASE_URL", "http://localhost:8000")
PROD_TEST_USER = os.getenv("PROD_TEST_USER")
PROD_TEST_PASSWORD = os.getenv("PROD_TEST_PASSWORD")


def run_docker_command(
    args: list,
    timeout: int = 10,
    container_fallback: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Run a docker command with proper error handling.

    Args:
        args: Command arguments (e.g., ["docker", "exec", "container", ...])
        timeout: Command timeout in seconds
        container_fallback: If provided and first container fails, try this one

    Returns:
        CompletedProcess result

    Raises:
        pytest.skip: If Docker is not available
        pytest.fail: If command times out
    """
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)

        # If failed and fallback provided, try fallback container
        if result.returncode != 0 and container_fallback:
            # Replace container name in args
            fallback_args = [
                container_fallback if arg == args[2] else arg
                for i, arg in enumerate(args)
            ]
            # Actually find and replace the container name (it's typically args[2])
            for i, arg in enumerate(args):
                if "sujbot_" in arg and "_dev_" not in arg:
                    fallback_args = args.copy()
                    fallback_args[i] = container_fallback
                    break

            warnings.warn(
                f"Primary container failed ({args[2]}), trying fallback ({container_fallback})"
            )
            result = subprocess.run(fallback_args, capture_output=True, timeout=timeout)

        return result

    except subprocess.TimeoutExpired:
        pytest.fail(f"Docker command timed out after {timeout}s: {' '.join(args)}")
    except FileNotFoundError:
        pytest.skip("Docker CLI not found - is Docker installed?")
    except PermissionError:
        pytest.skip(
            "Permission denied running Docker - "
            "ensure current user is in docker group"
        )


def get_container_name(base_name: str) -> str:
    """
    Get the actual running container name (prod or dev).

    Args:
        base_name: Base container name (e.g., "sujbot_postgres")

    Returns:
        The actual running container name
    """
    dev_name = base_name.replace("sujbot_", "sujbot_dev_")

    # Check which container is running
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        timeout=5
    )
    running = result.stdout.decode().strip().split("\n")

    if base_name in running:
        return base_name
    elif dev_name in running:
        return dev_name
    else:
        # Return base name, tests will fail with clear message
        return base_name


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

    Returns empty dict if:
    - Credentials not configured (PROD_TEST_USER/PASSWORD not set)
    - Login returns 401 (invalid credentials)

    Fails the test if:
    - Server is unreachable (network error)
    - Server returns unexpected error (5xx)
    """
    # Check if credentials are configured
    if not PROD_TEST_USER or not PROD_TEST_PASSWORD:
        warnings.warn(
            "PROD_TEST_USER and PROD_TEST_PASSWORD not set. "
            "Authenticated tests will be skipped."
        )
        return {}

    try:
        response = http_client.post(
            "/auth/login",
            json={"email": PROD_TEST_USER, "password": PROD_TEST_PASSWORD}
        )

        if response.status_code == 200:
            return dict(response.cookies)
        elif response.status_code == 401:
            warnings.warn(
                f"Authentication failed (401) for user {PROD_TEST_USER}. "
                "Check PROD_TEST_USER and PROD_TEST_PASSWORD environment variables."
            )
            return {}
        else:
            # Unexpected status - this is a real problem
            pytest.fail(
                f"Unexpected response from /auth/login: {response.status_code} - "
                f"{response.text[:200]}"
            )

    except httpx.ConnectError as e:
        pytest.fail(f"Cannot connect to {http_client.base_url}: {e}")
    except httpx.TimeoutException as e:
        pytest.fail(f"Login request timed out: {e}")
    except httpx.HTTPError as e:
        pytest.fail(f"HTTP error during login: {type(e).__name__}: {e}")

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
        pytest.skip(
            "Authentication required - set PROD_TEST_USER and PROD_TEST_PASSWORD "
            "environment variables with valid credentials"
        )


@pytest.fixture
def requires_services(http_client: httpx.Client):
    """Skip test if services are not running."""
    try:
        response = http_client.get("/health", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Services not healthy: {response.status_code}")
    except httpx.ConnectError as e:
        pytest.skip(f"Services not running (connection refused): {e}")
    except httpx.TimeoutException as e:
        pytest.skip(f"Services not responding (timeout): {e}")
    except httpx.HTTPError as e:
        pytest.skip(f"Services error: {type(e).__name__}: {e}")


def parse_sse_events(response, track_failures: bool = True) -> tuple[list, list]:
    """
    Parse SSE events from streaming response.

    Args:
        response: httpx streaming response
        track_failures: If True, track JSON parse failures

    Returns:
        Tuple of (events list, parse_failures list)
    """
    events = []
    parse_failures = []
    current_event = None

    for line in response.iter_lines():
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                data = {"event_type": current_event, "data": None}
                if data_str:
                    data["data"] = json.loads(data_str)
                events.append(data)
            except json.JSONDecodeError as e:
                if track_failures:
                    parse_failures.append({
                        "line": line[:100],
                        "error": str(e),
                        "event_type": current_event
                    })

    return events, parse_failures


def assert_no_excessive_parse_failures(
    parse_failures: list,
    total_events: int,
    threshold: float = 0.1
):
    """
    Assert that JSON parse failure rate is below threshold.

    Args:
        parse_failures: List of parse failures
        total_events: Total number of events (successful + failed)
        threshold: Maximum acceptable failure rate (default 10%)
    """
    if not parse_failures:
        return

    if total_events == 0:
        total_events = len(parse_failures)

    failure_rate = len(parse_failures) / total_events

    if failure_rate > threshold:
        pytest.fail(
            f"Excessive JSON parse failures: {len(parse_failures)}/{total_events} "
            f"({failure_rate:.1%} > {threshold:.0%}). "
            f"First failure: {parse_failures[0]}"
        )
    elif parse_failures:
        warnings.warn(
            f"{len(parse_failures)} JSON parse failures during streaming "
            f"(within acceptable threshold)"
        )
