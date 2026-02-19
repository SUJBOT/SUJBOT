"""
Local VL Embedding Client (drop-in replacement for JinaClient)

Calls a local vLLM or compatible server for text and image embeddings.
Uses the OpenAI-compatible /v1/embeddings endpoint.

Designed for Qwen3-VL-Embedding-8B served via vLLM on GB10 (DGX Spark).

Interface matches JinaClient:
- embed_query(text) -> np.ndarray
- embed_image(base64_data) -> np.ndarray
- embed_pages(page_images) -> np.ndarray
- close() -> None
"""

import base64
import hashlib
import logging
import os
from typing import List, Optional

import httpx
import numpy as np

from ..exceptions import EmbeddingError
from ..utils.cache import LRUCache

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_EMBEDDING_URL = "http://localhost:8081/v1"
DEFAULT_LOCAL_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_DIMENSIONS = 4096
BATCH_SIZE = 4


class LocalVLEmbedder:
    """
    Local embedding client for VL architecture.

    Drop-in replacement for JinaClient. Calls a local vLLM server
    with OpenAI-compatible /v1/embeddings endpoint.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = BATCH_SIZE,
        cache_max_size: int = 500,
    ):
        # Env var takes priority (Docker overrides config.json's localhost URL)
        self.base_url = (
            os.getenv("LOCAL_EMBEDDING_BASE_URL")
            or base_url
            or DEFAULT_LOCAL_EMBEDDING_URL
        )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._query_cache: LRUCache[np.ndarray] = LRUCache(
            max_size=cache_max_size, name="local_embed_query_cache"
        )
        self._client = httpx.Client(timeout=120.0)
        self._embed_url = f"{self.base_url.rstrip('/')}/embeddings"

        logger.info(
            "LocalVLEmbedder initialized: %s (%d-dim) at %s",
            model, dimensions, self.base_url,
        )

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _post_embeddings(self, payload: dict, context: str = "") -> dict:
        """POST to local embedding server."""
        try:
            response = self._client.post(
                self._embed_url,
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(
                f"Local embedding server returned {e.response.status_code} ({context})",
                details={
                    "status": e.response.status_code,
                    "body": e.response.text[:500],
                    "url": self._embed_url,
                },
                cause=e,
            )
        except httpx.ConnectError as e:
            raise EmbeddingError(
                f"Cannot connect to local embedding server at {self._embed_url}. "
                f"Is the vLLM server running? ({context})",
                details={"url": self._embed_url},
                cause=e,
            )
        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Local embedding request failed ({context}): {e}",
                details={"url": self._embed_url},
                cause=e,
            )

    def _extract_embedding(self, data: dict, index: int = 0) -> np.ndarray:
        """Extract and validate embedding from API response."""
        if "data" not in data or not data["data"]:
            raise EmbeddingError(
                "Local embedding server returned unexpected response: missing 'data'",
                details={"response_keys": list(data.keys())},
            )
        if index >= len(data["data"]):
            raise EmbeddingError(
                f"Expected at least {index + 1} embeddings, got {len(data['data'])}",
            )
        item = data["data"][index]
        if "embedding" not in item:
            raise EmbeddingError(
                "Local embedding response item missing 'embedding' field",
                details={"item_keys": list(item.keys())},
            )
        vec = np.array(item["embedding"], dtype=np.float32)
        return self._l2_normalize(vec)

    def close(self) -> None:
        """Close the persistent HTTP client."""
        self._client.close()

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a text query.

        Args:
            text: Query text

        Returns:
            L2-normalized embedding array of shape (dimensions,)
        """
        cache_key = self._cache_key(text)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float",
        }

        data = self._post_embeddings(payload, context="query")
        embedding = self._extract_embedding(data)

        self._query_cache.set(cache_key, embedding)
        return embedding

    def embed_image(self, base64_data: str) -> np.ndarray:
        """
        Embed a single image for retrieval query (image-to-page search).

        Args:
            base64_data: Base64-encoded image data. Accepts raw base64 or
                         full data URI (data:image/png;base64,...)

        Returns:
            L2-normalized embedding array of shape (dimensions,)
        """
        if base64_data.startswith("data:"):
            data_uri = base64_data
        else:
            data_uri = f"data:image/png;base64,{base64_data}"

        # Pass data URI as plain string input (OpenAI-compatible embedding server)
        payload = {
            "model": self.model,
            "input": data_uri,
            "encoding_format": "float",
        }

        data = self._post_embeddings(payload, context="image_query")
        return self._extract_embedding(data)

    def embed_pages(self, page_images: List[bytes]) -> np.ndarray:
        """
        Embed page images (for indexing).

        Args:
            page_images: List of PNG image bytes

        Returns:
            L2-normalized embedding matrix of shape (N, dimensions)
        """
        all_embeddings = []

        for batch_start in range(0, len(page_images), self.batch_size):
            batch = page_images[batch_start : batch_start + self.batch_size]

            # Pass data URIs as plain string inputs (OpenAI-compatible embedding server)
            inputs = []
            for img_bytes in batch:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                inputs.append(f"data:image/png;base64,{b64}")

            payload = {
                "model": self.model,
                "input": inputs,
                "encoding_format": "float",
            }

            data = self._post_embeddings(
                payload,
                context=f"page batch {batch_start}-{batch_start + len(batch)}",
            )

            if "data" not in data or not data["data"]:
                raise EmbeddingError(
                    "Local embedding server returned unexpected response for page batch",
                    details={"batch_start": batch_start},
                )

            for item in data["data"]:
                if "embedding" not in item:
                    raise EmbeddingError(
                        "Local embedding response item missing 'embedding'",
                        details={"batch_start": batch_start},
                    )
                vec = np.array(item["embedding"], dtype=np.float32)
                all_embeddings.append(self._l2_normalize(vec))

            logger.debug(
                "Embedded page batch %d-%d (%d pages)",
                batch_start, batch_start + len(batch), len(batch),
            )

        return np.array(all_embeddings, dtype=np.float32)
