"""
Vision-Language (VL) RAG Architecture

Embeds PDF pages as images via configurable embedder (Jina v4 cloud or
Qwen3-VL-Embedding-8B local), retrieves relevant pages by cosine similarity,
and sends page images to a vision-capable LLM for answer generation.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from .jina_client import JinaClient
from .local_embedder import LocalVLEmbedder
from .page_store import PageStore
from .vl_indexing import VLIndexingPipeline
from .vl_retriever import VLPageResult, VLRetriever

logger = logging.getLogger(__name__)

__all__ = [
    "JinaClient",
    "LocalVLEmbedder",
    "PageStore",
    "VLIndexingPipeline",
    "VLPageResult",
    "VLRetriever",
    "create_vl_components",
]


def _create_embedder(vl_config: Dict[str, Any]):
    """
    Create embedder based on config (Jina cloud or local vLLM).

    Config key "embedder": "jina" (default) or "local".
    """
    embedder_type = vl_config.get("embedder", "jina")

    if embedder_type == "local":
        embedder = LocalVLEmbedder(
            base_url=vl_config.get("local_embedding_url"),
            model=vl_config.get("local_embedding_model", "Qwen/Qwen3-VL-Embedding-8B"),
            dimensions=vl_config.get("dimensions", 2048),
        )
        logger.info(
            "Using local embedder: %s (%d-dim) at %s",
            embedder.model,
            embedder.dimensions,
            embedder.base_url,
        )
        return embedder

    # Default: Jina cloud
    embedder = JinaClient(
        model=vl_config.get("jina_model", "jina-embeddings-v4"),
        dimensions=vl_config.get("dimensions", 2048),
    )
    logger.info(
        "Using Jina embedder: %s (%d-dim)",
        vl_config.get("jina_model", "jina-embeddings-v4"),
        vl_config.get("dimensions", 2048),
    )
    return embedder


def create_vl_components(
    vl_config: Dict[str, Any],
    vector_store: Any,
    adaptive_config: Optional[Any] = None,
) -> Tuple[VLRetriever, PageStore]:
    """
    Create VL components from config (SSOT factory function).

    Used by SingleAgentRunner to avoid duplicate initialization code.

    Embedder selection via config key "embedder": "jina" (default) or "local".

    Args:
        vl_config: VL configuration dict from config.json["vl"]
        vector_store: PostgreSQL vector store adapter
        adaptive_config: Optional AdaptiveKConfig for score thresholding

    Returns:
        Tuple of (VLRetriever, PageStore)
    """
    embedder = _create_embedder(vl_config)
    page_store = PageStore(
        store_dir=vl_config.get("page_store_dir", "data/vl_pages"),
        source_pdf_dir=vl_config.get("source_pdf_dir", "data"),
        dpi=vl_config.get("page_image_dpi", 150),
        image_format=vl_config.get("page_image_format", "png"),
    )
    vl_retriever = VLRetriever(
        jina_client=embedder,
        vector_store=vector_store,
        page_store=page_store,
        default_k=vl_config.get("default_k", 5),
        adaptive_config=adaptive_config,
    )
    embedder_name = vl_config.get("embedder", "jina")
    logger.info(
        "VL components initialized: %s embedder (%d-dim), page store at %s%s",
        embedder_name,
        vl_config.get("dimensions", 2048),
        vl_config.get("page_store_dir", "data/vl_pages"),
        f", adaptive-k={adaptive_config.method}" if adaptive_config and adaptive_config.enabled else "",
    )
    return vl_retriever, page_store
