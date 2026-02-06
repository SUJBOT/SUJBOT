"""
Vision-Language (VL) RAG Architecture

Alternative to OCR-based RAG: embeds PDF pages as images via Jina Embeddings v4,
retrieves relevant pages by cosine similarity, and sends page images to Claude's
vision API for answer generation.

Switchable via config.json: "architecture": "ocr" | "vl"
"""

from .jina_client import JinaClient
from .page_store import PageStore
from .vl_indexing import VLIndexingPipeline
from .vl_retriever import VLPageResult, VLRetriever

__all__ = [
    "JinaClient",
    "PageStore",
    "VLIndexingPipeline",
    "VLPageResult",
    "VLRetriever",
]
