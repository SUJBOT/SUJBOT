"""
VL Indexing Pipeline

Two modes:
1. load_precomputed_embeddings(pkl_path) — load page_embeddings.pkl into PostgreSQL
2. index_pdf(pdf_path, document_id) — render pages → Jina API → PostgreSQL

Uses PageStore for rendering, JinaClient for embedding,
PostgresVectorStoreAdapter for storage.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .jina_client import JinaClient
from .page_store import PageStore

logger = logging.getLogger(__name__)


class VLIndexingPipeline:
    """
    Indexes PDF pages as image embeddings for VL retrieval.
    """

    def __init__(
        self,
        jina_client: JinaClient,
        vector_store,  # PostgresVectorStoreAdapter
        page_store: PageStore,
    ):
        self.jina_client = jina_client
        self.vector_store = vector_store
        self.page_store = page_store

    def load_precomputed_embeddings(self, pkl_path: str) -> int:
        """
        Load pre-computed page embeddings from pickle file into PostgreSQL.

        Expected format:
            {
                "page_ids": ["BZ_VR1_p001", "BZ_VR1_p002", ...],
                "embeddings": ndarray(N, 2048)
            }

        Args:
            pkl_path: Path to page_embeddings.pkl

        Returns:
            Number of pages inserted
        """
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {pkl_path}")

        logger.info(f"Loading precomputed embeddings from {pkl_path}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        page_ids = data["page_ids"]
        embeddings = data["embeddings"]

        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)

        logger.info(f"Loaded {len(page_ids)} page embeddings ({embeddings.shape})")

        # Build page records
        pages = []
        for page_id in page_ids:
            doc_id, page_num = PageStore.page_id_to_components(page_id)
            pages.append({
                "page_id": page_id,
                "document_id": doc_id,
                "page_number": page_num,
                "image_path": None,  # Will be set when images are rendered
                "metadata": {},
            })

        inserted = self.vector_store.add_vl_pages(pages, embeddings)
        logger.info(f"Inserted {inserted} page embeddings into PostgreSQL")
        return inserted

    def index_pdf(
        self,
        pdf_path: str,
        document_id: Optional[str] = None,
    ) -> int:
        """
        Index a PDF: render pages → embed via Jina → store in PostgreSQL.

        Args:
            pdf_path: Path to PDF file
            document_id: Document identifier (default: filename stem)

        Returns:
            Number of pages indexed
        """
        pdf_path = Path(pdf_path)
        if document_id is None:
            document_id = pdf_path.stem

        # 1. Render pages to images
        logger.info(f"Rendering pages from {pdf_path.name}...")
        page_ids = self.page_store.render_pdf_pages(str(pdf_path), document_id)

        if not page_ids:
            logger.warning(f"No pages rendered from {pdf_path}")
            return 0

        # 2. Load image bytes for embedding
        page_images = []
        for page_id in page_ids:
            img_bytes = self.page_store.get_image_bytes(page_id)
            page_images.append(img_bytes)

        # 3. Embed via Jina
        logger.info(f"Embedding {len(page_images)} page images via Jina...")
        embeddings = self.jina_client.embed_pages(page_images)

        # 4. Store in PostgreSQL
        pages = []
        for page_id in page_ids:
            doc_id, page_num = PageStore.page_id_to_components(page_id)
            pages.append({
                "page_id": page_id,
                "document_id": doc_id,
                "page_number": page_num,
                "image_path": self.page_store.get_image_path(page_id),
                "metadata": {},
            })

        inserted = self.vector_store.add_vl_pages(pages, embeddings)
        logger.info(f"Indexed {inserted} pages from {pdf_path.name}")
        return inserted
