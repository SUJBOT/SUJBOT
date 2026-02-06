"""
Vision-Language (VL) RAG Architecture

Alternative to OCR-based RAG: embeds PDF pages as images via Jina Embeddings v4,
retrieves relevant pages by cosine similarity, and sends page images to Claude's
vision API for answer generation.

Switchable via config.json: "architecture": "ocr" | "vl"
"""

import logging
from typing import Any, Dict, Tuple

from .jina_client import JinaClient
from .page_store import PageStore
from .vl_indexing import VLIndexingPipeline
from .vl_retriever import VLPageResult, VLRetriever

logger = logging.getLogger(__name__)

__all__ = [
    "JinaClient",
    "PageStore",
    "VLIndexingPipeline",
    "VLPageResult",
    "VLRetriever",
    "create_vl_components",
]


def create_vl_components(
    vl_config: Dict[str, Any],
    vector_store: Any,
) -> Tuple[VLRetriever, PageStore]:
    """
    Create VL components from config (SSOT factory function).

    Used by both SingleAgentRunner and MultiAgentRunner to avoid
    duplicate initialization code.

    Args:
        vl_config: VL configuration dict from config.json["vl"]
        vector_store: PostgreSQL vector store adapter

    Returns:
        Tuple of (VLRetriever, PageStore)
    """
    jina_client = JinaClient(
        model=vl_config.get("jina_model", "jina-embeddings-v4"),
        dimensions=vl_config.get("dimensions", 2048),
    )
    page_store = PageStore(
        store_dir=vl_config.get("page_store_dir", "data/vl_pages"),
        source_pdf_dir=vl_config.get("source_pdf_dir", "data"),
        dpi=vl_config.get("page_image_dpi", 150),
        image_format=vl_config.get("page_image_format", "png"),
    )
    vl_retriever = VLRetriever(
        jina_client=jina_client,
        vector_store=vector_store,
        page_store=page_store,
        default_k=vl_config.get("default_k", 5),
    )
    logger.info(
        "VL components initialized: Jina v4 (%d-dim), page store at %s",
        vl_config.get("dimensions", 2048),
        vl_config.get("page_store_dir", "data/vl_pages"),
    )
    return vl_retriever, page_store
