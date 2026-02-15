"""
Abstract Vector Store Adapter Interface

Defines the contract that all vector store implementations must follow.
VL-only architecture — searches vectors.vl_pages via cosine similarity.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import numpy as np


class VectorStoreAdapter(ABC):
    """
    Abstract interface for VL vector storage backends.

    All RAG tools depend on this interface. VL mode uses Jina v4 (2048-dim)
    embeddings with exact cosine scan on vectors.vl_pages.
    """

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            {
                'documents': int,
                'total_vectors': int,
                'vl_pages_count': int,
                'dimensions': int,
                'backend': str,
                'architecture': str,
            }
        """
        pass
