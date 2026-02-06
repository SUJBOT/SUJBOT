"""
Page Image Storage/Loading

Renders PDF pages to PNG images and provides access for:
- VL indexing (render → embed)
- VL retrieval (load base64 → send to Claude vision API)

Uses PyMuPDF (fitz) for PDF rendering at configurable DPI.
Stores images at: data/vl_pages/{document_id}/page_{NNN}.png
"""

import base64
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from ..exceptions import PageRenderError

logger = logging.getLogger(__name__)

DEFAULT_DPI = 150
DEFAULT_FORMAT = "png"
DEFAULT_STORE_DIR = "data/vl_pages"


class PageStore:
    """
    Manages rendered PDF page images on disk.

    Supports lazy rendering: if a requested image doesn't exist,
    it will be rendered from the source PDF on demand.
    """

    def __init__(
        self,
        store_dir: str = DEFAULT_STORE_DIR,
        source_pdf_dir: str = "data",
        dpi: int = DEFAULT_DPI,
        image_format: str = DEFAULT_FORMAT,
    ):
        self.store_dir = Path(store_dir)
        self.source_pdf_dir = Path(source_pdf_dir)
        self.dpi = dpi
        self.image_format = image_format
        self.store_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def page_id_to_components(page_id: str) -> Tuple[str, int]:
        """
        Parse page_id into document_id and page_number.

        Format: {document_id}_p{NNN}
        Example: BZ_VR1_p001 → ("BZ_VR1", 1)
        """
        # Find the last _p followed by digits
        parts = page_id.rsplit("_p", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"Invalid page_id format: {page_id} (expected {{doc_id}}_p{{NNN}})")
        return parts[0], int(parts[1])

    @staticmethod
    def make_page_id(document_id: str, page_number: int) -> str:
        """Create page_id from components."""
        return f"{document_id}_p{page_number:03d}"

    def _image_path(self, document_id: str, page_number: int) -> Path:
        """Get filesystem path for a page image."""
        doc_dir = self.store_dir / document_id
        return doc_dir / f"page_{page_number:03d}.{self.image_format}"

    def get_image_path(self, page_id: str) -> str:
        """
        Get filesystem path for a page image.

        If image doesn't exist, attempts lazy rendering from source PDF.
        """
        doc_id, page_num = self.page_id_to_components(page_id)
        path = self._image_path(doc_id, page_num)

        if not path.exists():
            # Lazy render from source PDF
            self._lazy_render_page(doc_id, page_num)

        return str(path)

    @lru_cache(maxsize=200)
    def get_image_base64(self, page_id: str) -> str:
        """
        Get base64-encoded image data for Anthropic multimodal API.

        Results are cached via LRU cache for repeated access.
        """
        path = self.get_image_path(page_id)
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            raise PageRenderError(
                f"Page image not found: {path}",
                details={"page_id": page_id, "path": str(path)},
            )

    def get_image_bytes(self, page_id: str) -> bytes:
        """Get raw image bytes for a page."""
        path = self.get_image_path(page_id)
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            raise PageRenderError(
                f"Page image not found: {path}",
                details={"page_id": page_id, "path": str(path)},
            )

    def render_pdf_pages(
        self,
        pdf_path: str,
        document_id: str,
    ) -> List[str]:
        """
        Render all pages of a PDF to images.

        Args:
            pdf_path: Path to source PDF
            document_id: Document identifier for page_id generation

        Returns:
            List of page_ids for rendered pages
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise PageRenderError(
                "PyMuPDF (fitz) is required for PDF page rendering. "
                "Install with: pip install PyMuPDF"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PageRenderError(
                f"PDF not found: {pdf_path}",
                details={"pdf_path": str(pdf_path)},
            )

        doc_dir = self.store_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        page_ids = []
        try:
            doc = fitz.open(str(pdf_path))
            zoom = self.dpi / 72.0  # fitz default is 72 DPI
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)

                # page_number is 1-indexed
                page_number = page_num + 1
                img_path = self._image_path(document_id, page_number)

                pix.save(str(img_path))
                page_id = self.make_page_id(document_id, page_number)
                page_ids.append(page_id)

            doc.close()
            logger.info(f"Rendered {len(page_ids)} pages from {pdf_path.name} → {doc_dir}")

        except Exception as e:
            if not isinstance(e, PageRenderError):
                raise PageRenderError(
                    f"Failed to render PDF pages: {e}",
                    details={"pdf_path": str(pdf_path), "document_id": document_id},
                    cause=e,
                )
            raise

        return page_ids

    def _lazy_render_page(self, document_id: str, page_number: int) -> None:
        """Render a single page on demand from source PDF."""
        # Search for source PDF in source_pdf_dir
        pdf_candidates = list(self.source_pdf_dir.glob(f"{document_id}*.pdf"))
        if not pdf_candidates:
            raise PageRenderError(
                f"Source PDF not found for document '{document_id}' in {self.source_pdf_dir}",
                details={"document_id": document_id, "source_dir": str(self.source_pdf_dir)},
            )

        pdf_path = pdf_candidates[0]
        logger.info(f"Lazy rendering page {page_number} from {pdf_path.name}")

        try:
            import fitz
        except ImportError:
            raise PageRenderError("PyMuPDF (fitz) is required for lazy page rendering")

        doc_dir = self.store_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(str(pdf_path))
            if page_number < 1 or page_number > len(doc):
                doc.close()
                raise PageRenderError(
                    f"Page {page_number} out of range (PDF has {len(doc)} pages)",
                    details={"page_number": page_number, "total_pages": len(doc)},
                )

            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            page = doc[page_number - 1]  # 0-indexed
            pix = page.get_pixmap(matrix=matrix)

            img_path = self._image_path(document_id, page_number)
            pix.save(str(img_path))
            doc.close()

        except Exception as e:
            if not isinstance(e, PageRenderError):
                raise PageRenderError(
                    f"Failed to lazy-render page: {e}",
                    details={"document_id": document_id, "page_number": page_number},
                    cause=e,
                )
            raise

    def list_document_pages(self, document_id: str) -> List[str]:
        """List all available page_ids for a document."""
        doc_dir = self.store_dir / document_id
        if not doc_dir.exists():
            return []

        page_ids = []
        for img_file in sorted(doc_dir.glob(f"page_*.{self.image_format}")):
            # Extract page number from filename
            stem = img_file.stem  # e.g., "page_001"
            num_str = stem.replace("page_", "")
            try:
                page_num = int(num_str)
                page_ids.append(self.make_page_id(document_id, page_num))
            except ValueError:
                logger.warning(f"Skipping malformed page filename: {img_file.name}")
                continue

        return page_ids
