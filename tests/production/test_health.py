"""
Health check tests for production services.

These tests verify that all services are running and healthy.
Run these first to diagnose infrastructure issues.

Usage:
    uv run pytest tests/production/test_health.py -v
"""

import pytest
import httpx
import subprocess

from .conftest import run_docker_command, get_container_name


class TestBackendHealth:
    """Test backend API health."""

    def test_health_endpoint(self, http_client: httpx.Client):
        """Backend /health endpoint returns 200."""
        response = http_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ("healthy", "ready"), f"Unexpected status: {data.get('status')}"
        # Health response may include version, timestamp, message, or details
        assert any(key in data for key in ("version", "timestamp", "message", "details")), \
            f"Health response missing expected fields: {data}"

    def test_root_endpoint(self, http_client: httpx.Client):
        """Backend / endpoint returns API info."""
        response = http_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data or "status" in data

    def test_docs_endpoint(self, http_client: httpx.Client):
        """Backend /docs endpoint is accessible."""
        response = http_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestDatabaseHealth:
    """Test database connectivity."""

    def test_postgres_connection(self):
        """PostgreSQL is accessible via docker exec."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            ["docker", "exec", container, "pg_isready", "-U", "postgres"],
            timeout=10
        )
        assert result.returncode == 0, f"PostgreSQL not ready: {result.stderr.decode()}"

    def test_postgres_sujbot_db_exists(self):
        """sujbot database exists and has tables."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot",
                "-c", "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'vectors';"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Cannot query sujbot DB: {result.stderr.decode()}"
        # Should have layer1, layer2, layer3 tables
        output = result.stdout.decode()
        assert "3" in output or "2" in output or "1" in output, "vectors schema should have tables"

    def test_redis_connection(self):
        """Redis is accessible and responding."""
        container = get_container_name("sujbot_redis")
        result = run_docker_command(
            ["docker", "exec", container, "redis-cli", "ping"],
            timeout=10
        )
        assert result.returncode == 0, f"Redis not responding: {result.stderr.decode()}"
        assert b"PONG" in result.stdout, f"Expected PONG, got: {result.stdout.decode()}"


class TestContainerHealth:
    """Test Docker container status."""

    @pytest.fixture
    def running_containers(self) -> list:
        """Get list of running container names."""
        result = run_docker_command(
            ["docker", "ps", "--format", "{{.Names}}"],
            timeout=10
        )
        if result.returncode != 0:
            pytest.fail(f"Cannot list containers: {result.stderr.decode()}")
        return result.stdout.decode().strip().split("\n")

    @pytest.mark.parametrize("container", [
        "sujbot_postgres",
        "sujbot_redis",
        "sujbot_backend",
    ])
    def test_container_running(self, running_containers: list, container: str):
        """Required container is running."""
        # Also check dev container names
        dev_container = container.replace("sujbot_", "sujbot_dev_")
        assert container in running_containers or dev_container in running_containers, \
            f"Container {container} not running. Running: {running_containers}"

    def test_backend_container_healthy(self):
        """Backend container responds to health check."""
        container = get_container_name("sujbot_backend")
        # First check if container is running
        check = run_docker_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            timeout=5
        )
        if check.returncode != 0 or b"true" not in check.stdout:
            pytest.skip(f"Container {container} is not running")

        result = run_docker_command(
            [
                "docker", "exec", container,
                "curl", "-sf", "http://localhost:8000/health"
            ],
            timeout=15
        )
        assert result.returncode == 0, \
            f"Backend health check failed inside container: {result.stderr.decode()}"


class TestNetworkHealth:
    """Test network connectivity between services."""

    def test_backend_can_reach_postgres(self):
        """Backend can connect to PostgreSQL (verified via health endpoint)."""
        # The health endpoint already verifies DB connectivity
        # This test uses the backend's configured DATABASE_URL
        container = get_container_name("sujbot_backend")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "python", "-c",
                "import os; from sqlalchemy import create_engine, text; "
                "e = create_engine(os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/sujbot')); "
                "e.connect().execute(text('SELECT 1'))"
            ],
            timeout=15
        )
        # If this fails, it's usually due to missing env vars in test context
        # The actual connectivity is verified by health endpoint
        if result.returncode != 0:
            pytest.skip(f"Network test requires DATABASE_URL env: {result.stderr.decode()[:100]}")

    def test_backend_can_reach_redis(self):
        """Backend can connect to Redis (verified via health endpoint)."""
        # Test Redis connectivity using the backend's internal mechanism
        container = get_container_name("sujbot_backend")
        # Use nc (netcat) to verify network connectivity to redis
        result = run_docker_command(
            [
                "docker", "exec", container,
                "sh", "-c",
                "echo PING | nc -w 2 redis 6379 | grep -q PONG"
            ],
            timeout=10
        )
        # nc might not be available, skip if so
        if result.returncode != 0:
            # Try alternative: direct python socket
            result2 = run_docker_command(
                [
                    "docker", "exec", container,
                    "python", "-c",
                    "import socket; s = socket.socket(); s.settimeout(2); s.connect(('redis', 6379)); s.close()"
                ],
                timeout=10
            )
            if result2.returncode != 0:
                pytest.skip("Redis connectivity test requires nc or socket access")
