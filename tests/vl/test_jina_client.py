"""
Tests for JinaClient — embedding, caching, normalization, error handling.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.exceptions import JinaAPIError
from src.vl.jina_client import JINA_DIMENSIONS, JinaClient


@pytest.fixture
def mock_response():
    """Create a mock httpx response with valid embedding."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [{"embedding": [0.1] * JINA_DIMENSIONS}],
        "model": "jina-embeddings-v4",
        "usage": {"total_tokens": 10},
    }
    return response


@pytest.fixture
def client():
    """Create JinaClient with test API key."""
    with patch.dict(os.environ, {"JINA_API_KEY": "test-jina-key"}):
        return JinaClient(cache_max_size=3)


class TestJinaClientInit:
    """Initialization tests."""

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JINA_API_KEY", None)
            with pytest.raises(JinaAPIError, match="JINA_API_KEY"):
                JinaClient()

    def test_custom_dimensions(self):
        with patch.dict(os.environ, {"JINA_API_KEY": "test"}):
            client = JinaClient(dimensions=1024)
            assert client.dimensions == 1024


class TestL2Normalize:
    """L2 normalization tests."""

    def test_nonzero_vector_has_unit_norm(self, client):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normalized = client._l2_normalize(vec)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-5

    def test_zero_vector_returns_zero(self, client):
        vec = np.zeros(10, dtype=np.float32)
        normalized = client._l2_normalize(vec)
        assert np.allclose(normalized, vec)

    def test_already_normalized_unchanged(self, client):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        normalized = client._l2_normalize(vec)
        assert np.allclose(normalized, vec)


class TestEmbedQuery:
    """embed_query tests."""

    def test_embed_query_returns_array(self, client, mock_response):
        client._client.post = MagicMock(return_value=mock_response)

        result = client.embed_query("test query")
        assert isinstance(result, np.ndarray)
        assert result.shape == (JINA_DIMENSIONS,)
        # Should be L2-normalized
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_embed_query_cache_hit(self, client, mock_response):
        """Second call with same query should use cache, not API."""
        client._client.post = MagicMock(return_value=mock_response)

        # First call → API
        result1 = client.embed_query("test query")
        # Second call → cache
        result2 = client.embed_query("test query")

        assert np.allclose(result1, result2)
        # post should only have been called once (second was cached)
        assert client._client.post.call_count == 1

    def test_embed_query_cache_eviction(self, client, mock_response):
        """Cache with max_size=3 should evict oldest entry on 4th unique query."""
        client._client.post = MagicMock(return_value=mock_response)

        # Fill cache
        client.embed_query("query1")
        client.embed_query("query2")
        client.embed_query("query3")
        assert len(client._query_cache) == 3

        # This should evict query1
        client.embed_query("query4")
        assert len(client._query_cache) == 3

    def test_embed_query_http_error_raises_jina_error(self, client):
        """HTTP errors should be wrapped in JinaAPIError."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )
        client._client.post = MagicMock(return_value=mock_response)

        with pytest.raises(JinaAPIError, match="401"):
            client.embed_query("test")

    def test_embed_query_missing_data_field_raises(self, client):
        """Response missing 'data' field should raise JinaAPIError."""
        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.raise_for_status.return_value = None
        bad_response.json.return_value = {"error": "bad request"}
        client._client.post = MagicMock(return_value=bad_response)

        with pytest.raises(JinaAPIError, match="missing 'data'"):
            client.embed_query("test")

    def test_embed_query_missing_embedding_field_raises(self, client):
        """Response with data but missing 'embedding' should raise JinaAPIError."""
        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.raise_for_status.return_value = None
        bad_response.json.return_value = {"data": [{"object": "embedding"}]}
        client._client.post = MagicMock(return_value=bad_response)

        with pytest.raises(JinaAPIError, match="missing 'embedding'"):
            client.embed_query("test")


class TestEmbedPages:
    """embed_pages tests."""

    def test_embed_pages_batching(self, client):
        """10 images with batch_size=8 should make 2 API calls."""
        client.batch_size = 8

        def make_batch_response(*args, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            # Return as many embeddings as images in the batch
            payload = kwargs.get("json") or (args[1] if len(args) > 1 else {})
            num_items = len(payload.get("input", []))
            resp.json.return_value = {
                "data": [{"embedding": [0.5] * JINA_DIMENSIONS} for _ in range(num_items)],
            }
            return resp

        client._client.post = MagicMock(side_effect=make_batch_response)

        # 10 fake images
        images = [b"fake_png_bytes"] * 10
        result = client.embed_pages(images)

        assert result.shape == (10, JINA_DIMENSIONS)
        # 10 images / batch_size 8 = 2 API calls
        assert client._client.post.call_count == 2
