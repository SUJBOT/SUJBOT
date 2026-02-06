"""
Jina Embeddings v4 API Client

Provides text and image embedding via Jina's multimodal embedding API.
Uses task-specific LoRA adapters for asymmetric retrieval:
- `retrieval.query` for text queries
- `retrieval.passage` for page images (base64-encoded)

Dimensions: 2048 (matching pre-indexed page embeddings).
"""

import base64
import hashlib
import logging
import os
from typing import Dict, List, Optional

import httpx
import numpy as np

from ..exceptions import JinaAPIError

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v4"
JINA_DIMENSIONS = 2048
BATCH_SIZE = 8


class JinaClient:
    """
    Async-ready client for Jina Embeddings v4 API.

    Supports two embedding modes:
    - Text queries (retrieval.query task)
    - Page images (retrieval.passage task, base64-encoded)

    Includes MD5-keyed query cache for repeated queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = JINA_MODEL,
        dimensions: int = JINA_DIMENSIONS,
        batch_size: int = BATCH_SIZE,
        cache_max_size: int = 500,
    ):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise JinaAPIError("JINA_API_KEY not set in environment or constructor")

        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._query_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = cache_max_size

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize to match pre-indexed embeddings."""
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a text query using retrieval.query task.

        Args:
            text: Query text

        Returns:
            L2-normalized embedding array of shape (dimensions,)
        """
        cache_key = self._cache_key(text)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        payload = {
            "model": self.model,
            "task": "retrieval.query",
            "dimensions": self.dimensions,
            "input": [{"text": text}],
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    JINA_API_URL, json=payload, headers=self._headers()
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise JinaAPIError(
                f"Jina API returned {e.response.status_code}",
                details={"status": e.response.status_code, "body": e.response.text[:500]},
                cause=e,
            )
        except httpx.RequestError as e:
            raise JinaAPIError(
                f"Jina API request failed: {e}",
                cause=e,
            )

        # Validate response structure
        if "data" not in data or not data["data"]:
            raise JinaAPIError(
                "Jina API returned unexpected response: missing 'data' field",
                details={"response_keys": list(data.keys())},
            )
        if "embedding" not in data["data"][0]:
            raise JinaAPIError(
                "Jina API returned unexpected response: missing 'embedding' in data[0]",
                details={"data_keys": list(data["data"][0].keys())},
            )

        embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
        embedding = self._l2_normalize(embedding)

        # Cache (evict oldest if full)
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[cache_key] = embedding

        return embedding

    def embed_pages(self, page_images: List[bytes]) -> np.ndarray:
        """
        Embed page images using retrieval.passage task.

        Args:
            page_images: List of PNG image bytes

        Returns:
            L2-normalized embedding matrix of shape (N, dimensions)
        """
        all_embeddings = []

        for batch_start in range(0, len(page_images), self.batch_size):
            batch = page_images[batch_start : batch_start + self.batch_size]

            input_items = []
            for img_bytes in batch:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                input_items.append({"image": f"data:image/png;base64,{b64}"})

            payload = {
                "model": self.model,
                "task": "retrieval.passage",
                "dimensions": self.dimensions,
                "input": input_items,
            }

            try:
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        JINA_API_URL, json=payload, headers=self._headers()
                    )
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as e:
                raise JinaAPIError(
                    f"Jina API returned {e.response.status_code} for image batch",
                    details={
                        "status": e.response.status_code,
                        "batch_start": batch_start,
                        "batch_size": len(batch),
                    },
                    cause=e,
                )
            except httpx.RequestError as e:
                raise JinaAPIError(
                    f"Jina API image embedding request failed: {e}",
                    cause=e,
                )

            # Validate response structure
            if "data" not in data or not data["data"]:
                raise JinaAPIError(
                    "Jina API returned unexpected response for image batch: missing 'data'",
                    details={"batch_start": batch_start, "response_keys": list(data.keys())},
                )

            for item in data["data"]:
                if "embedding" not in item:
                    raise JinaAPIError(
                        "Jina API response item missing 'embedding' field",
                        details={"item_keys": list(item.keys()), "batch_start": batch_start},
                    )
                vec = np.array(item["embedding"], dtype=np.float32)
                all_embeddings.append(self._l2_normalize(vec))

            logger.debug(
                f"Embedded image batch {batch_start}-{batch_start + len(batch)} "
                f"({len(batch)} pages)"
            )

        return np.array(all_embeddings, dtype=np.float32)
