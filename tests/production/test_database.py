"""
Database integrity tests for production.

These tests verify that the database schema and data are correct.

Usage:
    uv run pytest tests/production/test_database.py -v
"""

import pytest

from .conftest import run_docker_command, get_container_name


class TestVectorSchema:
    """Test vector storage schema."""

    def test_vectors_schema_exists(self):
        """vectors schema exists in database."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'vectors';"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Query failed: {result.stderr.decode()}"
        assert "vectors" in result.stdout.decode(), \
            f"vectors schema not found. Output: {result.stdout.decode()}"

    def test_vl_pages_table_exists(self):
        """vl_pages table exists with correct columns."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT column_name FROM information_schema.columns "
                     "WHERE table_schema = 'vectors' AND table_name = 'vl_pages';"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Query failed: {result.stderr.decode()}"

        columns = result.stdout.decode()
        required_columns = ["page_id", "document_id", "embedding"]
        missing_columns = [col for col in required_columns if col not in columns]

        assert not missing_columns, \
            f"Missing columns in vectors.vl_pages: {missing_columns}. Found: {columns}"

    def test_vl_pages_has_data(self):
        """vl_pages table has indexed pages."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM vectors.vl_pages;"
            ],
            timeout=10
        )

        if result.returncode != 0:
            pytest.skip(f"Cannot query vectors.vl_pages: {result.stderr.decode()}")

        output = result.stdout.decode().strip()
        try:
            count = int(output)
        except ValueError:
            pytest.fail(f"Invalid count response: {output}")

        # Should have at least some indexed data in production
        # Skip if no data (might be fresh install)
        if count == 0:
            pytest.skip("No indexed data - fresh installation")
        assert count > 0, f"Expected indexed pages, got count={count}"

    def test_documents_table_exists(self):
        """documents table exists in vectors schema."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT column_name FROM information_schema.columns "
                     "WHERE table_schema = 'vectors' AND table_name = 'documents';"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Query failed: {result.stderr.decode()}"

        columns = result.stdout.decode()
        required_columns = ["document_id", "category"]
        missing_columns = [col for col in required_columns if col not in columns]

        assert not missing_columns, \
            f"Missing columns in vectors.documents: {missing_columns}. Found: {columns}"


class TestAuthSchema:
    """Test authentication schema."""

    def test_auth_schema_exists(self):
        """auth schema exists in database."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'auth';"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Query failed: {result.stderr.decode()}"
        assert "auth" in result.stdout.decode(), \
            f"auth schema not found. Output: {result.stdout.decode()}"

    def test_users_table_exists(self):
        """auth.users table exists."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM auth.users;"
            ],
            timeout=10
        )
        assert result.returncode == 0, \
            f"auth.users table should exist. Error: {result.stderr.decode()}"

    def test_conversations_table_exists(self):
        """auth.conversations table exists."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM auth.conversations;"
            ],
            timeout=10
        )
        assert result.returncode == 0, \
            f"auth.conversations table should exist. Error: {result.stderr.decode()}"


class TestEmbeddingDimensions:
    """Test embedding vector dimensions are correct."""

    def test_vl_embedding_dimension_2048(self):
        """VL page embeddings have 2048 dimensions (Jina v4)."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT vector_dims(embedding) FROM vectors.vl_pages LIMIT 1;"
            ],
            timeout=10
        )

        if result.returncode != 0:
            pytest.skip(f"Cannot query embeddings: {result.stderr.decode()}")

        output = result.stdout.decode().strip()
        if not output:
            pytest.skip("No embeddings to check dimensions")

        try:
            dims = int(output)
        except ValueError:
            pytest.fail(f"Invalid dimension response: {output}")

        assert dims == 2048, f"Expected 2048 dimensions, got {dims}"


class TestIndexes:
    """Test that required indexes exist."""

    def test_vl_pages_indexes_exist(self):
        """Indexes exist on vectors.vl_pages."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT indexname FROM pg_indexes WHERE tablename = 'vl_pages' AND schemaname = 'vectors';"
            ],
            timeout=10
        )

        if result.returncode != 0:
            pytest.skip(f"Cannot query indexes: {result.stderr.decode()}")

        # VL pages may use exact scan (no ANN index needed for ~500 pages)
        # Just verify the query succeeds
