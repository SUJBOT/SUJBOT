"""
VL Indexing Pipeline

Two modes:
1. load_precomputed_embeddings(pkl_path) — load page_embeddings.pkl into PostgreSQL
2. index_pdf(pdf_path, document_id) — render pages → Jina API → PostgreSQL

Optionally summarizes each page image via a multimodal LLM provider.

Uses PageStore for rendering, JinaClient for embedding,
PostgresVectorStoreAdapter for storage.
"""

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..exceptions import JinaAPIError, PageRenderError, StorageError
from .jina_client import JinaClient
from .page_store import PageStore

if TYPE_CHECKING:
    from ..agent.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "vl_page_summary.txt"


class VLIndexingPipeline:
    """
    Indexes PDF pages as image embeddings for VL retrieval.

    Optionally generates page summaries when a summary_provider is given.
    """

    def __init__(
        self,
        jina_client: JinaClient,
        vector_store,  # PostgresVectorStoreAdapter
        page_store: PageStore,
        summary_provider: Optional["BaseProvider"] = None,
    ):
        self.jina_client = jina_client
        self.vector_store = vector_store
        self.page_store = page_store
        self.summary_provider = summary_provider

    def _summarize_page(self, page_id: str) -> Optional[str]:
        """
        Generate a text summary for a single page image.

        Args:
            page_id: Page identifier (e.g., "BZ_VR1_p001")

        Returns:
            Summary text, or None if summarization fails
        """
        if not self.summary_provider:
            return None

        image_b64 = self.page_store.get_image_base64(page_id)
        prompt_text = _SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{self.page_store.image_format}",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        response = self.summary_provider.create_message(
            messages=messages,
            tools=[],
            system="",
            max_tokens=500,
            temperature=0.0,
        )
        return response.text.strip() if response.text else None

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
            pages.append(
                {
                    "page_id": page_id,
                    "document_id": doc_id,
                    "page_number": page_num,
                    "image_path": None,  # Will be set when images are rendered
                    "metadata": {},
                }
            )

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
        try:
            embeddings = self.jina_client.embed_pages(page_images)
        except JinaAPIError as e:
            raise PageRenderError(
                f"Pages rendered ({len(page_ids)}) but Jina embedding failed: {e}",
                details={"pdf_path": str(pdf_path), "pages_rendered": len(page_ids)},
                cause=e,
            ) from e

        # 4. Summarize pages (optional, if summary_provider is set)
        summaries: dict[str, Optional[str]] = {}
        if self.summary_provider:
            model_name = self.summary_provider.get_model_name()
            logger.info(f"Summarizing {len(page_ids)} pages via {model_name}...")
            for i, page_id in enumerate(page_ids):
                try:
                    summaries[page_id] = self._summarize_page(page_id)
                    logger.debug(f"Summarized page {i + 1}/{len(page_ids)}: {page_id}")
                except Exception as e:
                    logger.warning(f"Failed to summarize {page_id}: {e}")
                    summaries[page_id] = None
            summarized = sum(1 for v in summaries.values() if v)
            logger.info(f"Summarized {summarized}/{len(page_ids)} pages")

        # 5. Store in PostgreSQL
        pages = []
        for page_id in page_ids:
            doc_id, page_num = PageStore.page_id_to_components(page_id)
            metadata: dict = {}
            if summaries.get(page_id):
                metadata["page_summary"] = summaries[page_id]
                metadata["summary_model"] = self.summary_provider.get_model_name()
            pages.append(
                {
                    "page_id": page_id,
                    "document_id": doc_id,
                    "page_number": page_num,
                    "image_path": self.page_store.get_image_path(page_id),
                    "metadata": metadata,
                }
            )

        try:
            inserted = self.vector_store.add_vl_pages(pages, embeddings)
        except Exception as e:
            raise StorageError(
                f"Embeddings computed ({len(page_ids)} pages) but database insert failed: {e}",
                details={"pdf_path": str(pdf_path), "pages_embedded": len(page_ids)},
                cause=e,
            ) from e

        logger.info(f"Indexed {inserted} pages from {pdf_path.name}")
        return inserted

    def summarize_existing_pages(self, document_id: Optional[str] = None) -> int:
        """
        Backfill summaries for existing pages that lack one.

        Queries unsummarized pages, generates summaries via the summary_provider,
        and patches metadata in-place.

        Args:
            document_id: Optional filter to a single document

        Returns:
            Number of pages successfully summarized
        """
        if not self.summary_provider:
            logger.warning("No summary_provider configured — cannot summarize.")
            return 0

        pages = self.vector_store.get_vl_pages_without_summary(document_id)
        if not pages:
            logger.info("All pages already have summaries.")
            return 0

        model_name = self.summary_provider.get_model_name()
        logger.info(
            f"Summarizing {len(pages)} pages via {model_name}..."
        )

        success = 0
        for i, page in enumerate(pages):
            page_id = page["page_id"]
            try:
                summary = self._summarize_page(page_id)
                if summary:
                    self.vector_store.update_vl_page_metadata(
                        page_id,
                        {"page_summary": summary, "summary_model": model_name},
                    )
                    success += 1
                    logger.info(
                        f"[{i + 1}/{len(pages)}] {page_id}: "
                        f"{summary[:80]}..."
                    )
                else:
                    logger.warning(f"[{i + 1}/{len(pages)}] {page_id}: empty summary")
            except Exception as e:
                logger.warning(f"[{i + 1}/{len(pages)}] {page_id}: failed — {e}")

        logger.info(f"Summarization complete: {success}/{len(pages)} pages.")
        return success
