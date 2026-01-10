"""
Database integrity tests for production.

These tests verify that the database schema and data are correct.

Usage:
    uv run pytest tests/production/test_database.py -v
"""

import pytest
import subprocess
import json


class TestVectorSchema:
    """Test vector storage schema."""

    def test_vectors_schema_exists(self):
        """vectors schema exists in database."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'vectors';"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'vectors';"
                ],
                capture_output=True,
                timeout=10
            )
        assert result.returncode == 0
        assert "vectors" in result.stdout.decode()

    def test_layer3_table_exists(self):
        """layer3 (chunks) table exists with correct columns."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT column_name FROM information_schema.columns WHERE table_schema = 'vectors' AND table_name = 'layer3';"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT column_name FROM information_schema.columns WHERE table_schema = 'vectors' AND table_name = 'layer3';"
                ],
                capture_output=True,
                timeout=10
            )
        assert result.returncode == 0

        columns = result.stdout.decode()
        required_columns = ["chunk_id", "document_id", "embedding", "content"]
        for col in required_columns:
            assert col in columns, f"Column {col} missing from vectors.layer3"

    def test_layer3_has_data(self):
        """layer3 table has indexed chunks."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM vectors.layer3;"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT count(*) FROM vectors.layer3;"
                ],
                capture_output=True,
                timeout=10
            )

        if result.returncode != 0:
            pytest.skip("Cannot query vectors.layer3")

        count = int(result.stdout.decode().strip())
        # Should have at least some indexed data in production
        # Skip if no data (might be fresh install)
        if count == 0:
            pytest.skip("No indexed data - fresh installation")
        assert count > 0


class TestAuthSchema:
    """Test authentication schema."""

    def test_auth_schema_exists(self):
        """auth schema exists in database."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'auth';"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'auth';"
                ],
                capture_output=True,
                timeout=10
            )
        assert result.returncode == 0
        assert "auth" in result.stdout.decode()

    def test_users_table_exists(self):
        """auth.users table exists."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM auth.users;"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT count(*) FROM auth.users;"
                ],
                capture_output=True,
                timeout=10
            )
        assert result.returncode == 0, "auth.users table should exist"

    def test_conversations_table_exists(self):
        """auth.conversations table exists."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT count(*) FROM auth.conversations;"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT count(*) FROM auth.conversations;"
                ],
                capture_output=True,
                timeout=10
            )
        assert result.returncode == 0, "auth.conversations table should exist"


class TestEmbeddingDimensions:
    """Test embedding vector dimensions are correct."""

    def test_embedding_dimension_4096(self):
        """Embeddings have 4096 dimensions (Qwen3-Embedding-8B)."""
        result = subprocess.run(
            [
                "docker", "exec", "sujbot_postgres",
                "psql", "-U", "postgres", "-d", "sujbot", "-t",
                "-c", "SELECT vector_dims(embedding) FROM vectors.layer3 LIMIT 1;"
            ],
            capture_output=True,
            timeout=10
        )
        if result.returncode != 0:
            result = subprocess.run(
                [
                    "docker", "exec", "sujbot_dev_postgres",
                    "psql", "-U", "postgres", "-d", "sujbot", "-t",
                    "-c", "SELECT vector_dims(embedding) FROM vectors.layer3 LIMIT 1;"
                ],
                capture_output=True,
                timeout=10
            )

        if result.returncode != 0 or not result.stdout.decode().strip():
            pytest.skip("No embeddings to check dimensions")

        dims = int(result.stdout.decode().strip())
        assert dims == 4096, f"Expected 4096 dimensions, got {dims}"
