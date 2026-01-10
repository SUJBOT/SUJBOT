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
        result = subprocess.run(
            ["docker", "exec", "sujbot_postgres", "pg_isready", "-U", "postgres"],
            capture_output=True,
            timeout=10
        )
        assert result.returncode == 0, f"PostgreSQL not ready: {result.stderr.decode()}"

    def test_postgres_sujbot_db_exists(self):
        """sujbot database exists and has tables."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot",
                "-c", "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'vectors';"
            ],
            capture_output=True,
            timeout=10
        )
        assert result.returncode == 0, f"Cannot query sujbot DB: {result.stderr.decode()}"
        # Should have layer1, layer2, layer3 tables
        output = result.stdout.decode()
        assert "3" in output or "2" in output or "1" in output, "vectors schema should have tables"

    def test_redis_connection(self):
        """Redis is accessible and responding."""
        result = subprocess.run(
            ["docker", "exec", "sujbot_redis", "redis-cli", "ping"],
            capture_output=True,
            timeout=10
        )
        assert result.returncode == 0
        assert b"PONG" in result.stdout


class TestContainerHealth:
    """Test Docker container status."""

    @pytest.fixture
    def running_containers(self) -> list:
        """Get list of running container names."""
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            timeout=10
        )
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
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_backend",
                "curl", "-sf", "http://localhost:8000/health"
            ],
            capture_output=True,
            timeout=15
        )
        # Try dev container if prod fails
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_backend",
                    "curl", "-sf", "http://localhost:8000/health"
                ],
                capture_output=True,
                timeout=15
            )
        assert result.returncode == 0, "Backend health check failed inside container"


class TestNetworkHealth:
    """Test network connectivity between services."""

    def test_backend_can_reach_postgres(self):
        """Backend can connect to PostgreSQL."""
        # This is implicitly tested by /health if it queries DB
        # But we can also check explicitly
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_backend",
                "python", "-c",
                "import asyncpg; import asyncio; "
                "asyncio.run(asyncpg.connect('postgresql://postgres:postgres@postgres:5432/sujbot'))"
            ],
            capture_output=True,
            timeout=15
        )
        if result.returncode != 0:
            # Try dev container
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_backend",
                    "python", "-c",
                    "import asyncpg; import asyncio; "
                    "asyncio.run(asyncpg.connect('postgresql://postgres:postgres@postgres:5432/sujbot'))"
                ],
                capture_output=True,
                timeout=15
            )
        assert result.returncode == 0, f"Backend cannot reach PostgreSQL: {result.stderr.decode()}"

    def test_backend_can_reach_redis(self):
        """Backend can connect to Redis."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_backend",
                "python", "-c",
                "import redis; r = redis.Redis(host='redis', port=6379); r.ping()"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_backend",
                    "python", "-c",
                    "import redis; r = redis.Redis(host='redis', port=6379); r.ping()"
                ],
                capture_output=True,
                timeout=10
            )
        assert result.returncode == 0, f"Backend cannot reach Redis: {result.stderr.decode()}"
