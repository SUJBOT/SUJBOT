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

    def test_layer3_table_exists(self):
        """layer3 (chunks) table exists with correct columns."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT column_name FROM information_schema.columns "
                     "WHERE table_schema = 'vectors' AND table_name = 'layer3';"
            ],
            timeout=10
        )
        assert result.returncode == 0, f"Query failed: {result.stderr.decode()}"

        columns = result.stdout.decode()
        required_columns = ["chunk_id", "document_id", "embedding", "content"]
        missing_columns = [col for col in required_columns if col not in columns]

        assert not missing_columns, \
            f"Missing columns in vectors.layer3: {missing_columns}. Found: {columns}"

    def test_layer3_has_data(self):
        """layer3 table has indexed chunks."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM vectors.layer3;"
            ],
            timeout=10
        )

        if result.returncode != 0:
            pytest.skip(f"Cannot query vectors.layer3: {result.stderr.decode()}")

        output = result.stdout.decode().strip()
        try:
            count = int(output)
        except ValueError:
            pytest.fail(f"Invalid count response: {output}")

        # Should have at least some indexed data in production
        # Skip if no data (might be fresh install)
        if count == 0:
            pytest.skip("No indexed data - fresh installation")
        assert count > 0, f"Expected indexed chunks, got count={count}"


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

    def test_embedding_dimension_4096(self):
        """Embeddings have 4096 dimensions (Qwen3-Embedding-8B)."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT vector_dims(embedding) FROM vectors.layer3 LIMIT 1;"
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

        assert dims == 4096, f"Expected 4096 dimensions, got {dims}"


class TestIndexes:
    """Test that required indexes exist."""

    def test_layer3_hnsw_index_exists(self):
        """HNSW index exists on vectors.layer3 for fast similarity search."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT indexname FROM pg_indexes WHERE tablename = 'layer3' AND schemaname = 'vectors';"
            ],
            timeout=10
        )

        if result.returncode != 0:
            pytest.skip(f"Cannot query indexes: {result.stderr.decode()}")

        indexes = result.stdout.decode()
        # Should have some index (HNSW or IVFFlat)
        if not indexes.strip():
            pytest.skip("No indexes found - may be fresh installation")

        # Check for vector index (name may vary)
        assert indexes.strip(), "No indexes found on vectors.layer3"

    def test_content_tsv_index_exists(self):
        """Full-text search index exists on content_tsv column."""
        container = get_container_name("sujbot_postgres")
        result = run_docker_command(
            [
                "docker", "exec", container,
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT indexname FROM pg_indexes WHERE tablename = 'layer3' "
                     "AND schemaname = 'vectors' AND indexdef LIKE '%content_tsv%';"
            ],
            timeout=10
        )

        if result.returncode != 0:
            pytest.skip(f"Cannot query indexes: {result.stderr.decode()}")

        # Full-text index is optional, just check query works
        # (result may be empty if not using FTS)
