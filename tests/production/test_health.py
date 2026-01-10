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
        assert data["status"] in ("healthy", "ready")
        assert "version" in data or "timestamp" in data

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
        """Backend can connect to PostgreSQL."""
        container = get_container_name("sujbot_backend")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "python", "-c",
                "import asyncpg; import asyncio; "
                "asyncio.run(asyncpg.connect('postgresql://postgres:postgres@postgres:5432/sujbot'))"
            ],
            timeout=15
        )
        assert result.returncode == 0, f"Backend cannot reach PostgreSQL: {result.stderr.decode()}"

    def test_backend_can_reach_redis(self):
        """Backend can connect to Redis."""
        container = get_container_name("sujbot_backend")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "python", "-c",
                "import redis; r = redis.Redis(host='redis', port=6379); r.ping()"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Backend cannot reach Redis: {result.stderr.decode()}"
