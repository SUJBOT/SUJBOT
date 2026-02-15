"""
Graph Embedder — multilingual-e5-small for entity/relationship/community search.

Lazy-loads the model on first use. Produces 384-dim normalized vectors.
E5 models require prefixes: "query: " for search queries, "passage: " for stored text.
"""

import logging
import threading
from typing import List

import numpy as np

from ..exceptions import EmbeddingError

logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384


class GraphEmbedder:
    """Encodes short text phrases using multilingual-e5-small (384-dim)."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading {MODEL_NAME}...")
                self._model = SentenceTransformer(MODEL_NAME)
                dim = self._model.get_sentence_embedding_dimension()
                if dim != EMBEDDING_DIM:
                    raise EmbeddingError(
                        f"Expected {EMBEDDING_DIM}-dim embeddings, got {dim}-dim from {MODEL_NAME}"
                    )
                logger.info(f"Loaded {MODEL_NAME} ({EMBEDDING_DIM}-dim)")
            except EmbeddingError:
                raise
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load {MODEL_NAME}: {e}", cause=e
                ) from e

    def encode_passages(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts for storage (with 'passage: ' prefix)."""
        self._load_model()
        try:
            prefixed = [f"passage: {t}" for t in texts]
            return self._model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to encode {len(texts)} passages: {e}", cause=e
            ) from e

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single search query (with 'query: ' prefix)."""
        self._load_model()
        try:
            return self._model.encode(f"query: {text}", normalize_embeddings=True)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to encode query '{text[:50]}': {e}", cause=e
            ) from e
