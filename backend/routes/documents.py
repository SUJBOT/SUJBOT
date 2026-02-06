"""
Document REST API

Serves PDF files from the data/ folder with security checks.
Supports document upload with VL indexing pipeline and SSE progress.
All endpoints require JWT authentication.
"""

import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from backend.config import PDF_BASE_DIR
from backend.middleware.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

# Module-level VL components (set by main.py during startup)
_vl_components: Dict[str, Any] = {}

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


def set_vl_components(jina_client: Any, page_store: Any, vector_store: Any) -> None:
    """Set VL components for upload endpoint (called from main.py lifespan)."""
    _vl_components["jina_client"] = jina_client
    _vl_components["page_store"] = page_store
    _vl_components["vector_store"] = vector_store


class DocumentInfo(BaseModel):
    """Document metadata for listing available PDFs."""
    document_id: str
    display_name: str
    filename: str
    size_bytes: int

# Valid document_id patterns
# Direct format: alphanumeric, underscore, hyphen (e.g., "BZ_VR1")
DIRECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
# Legal format: "number/year Sb." (e.g., "157/2025 Sb.")
LEGAL_ID_PATTERN = re.compile(r"^(\d+)/(\d{4})\s*Sb\.$")


def _format_display_name(filename: str) -> str:
    """
    Create human-readable display name from filename.

    Examples:
        "BZ_VR1.pdf" → "BZ VR1"
        "Sb_2016_263_2024-01-01_IZ.pdf" → "263/2016 Sb."
    """
    stem = filename.replace(".pdf", "")

    # Check for legal document format: Sb_YYYY_NNN_*
    legal_match = re.match(r"Sb_(\d{4})_(\d+)_.*", stem)
    if legal_match:
        year, number = legal_match.groups()
        return f"{number}/{year} Sb."

    # Default: replace underscores with spaces
    return stem.replace("_", " ")


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(
    user: Dict = Depends(get_current_user)
) -> List[DocumentInfo]:
    """
    List all available PDF documents.

    Returns:
        List of document metadata with IDs suitable for the PDF viewer.

    Security:
        - Authentication required (JWT)
        - Only returns documents in the allowed data/ directory
    """
    documents = []

    try:
        for pdf_path in PDF_BASE_DIR.glob("*.pdf"):
            filename = pdf_path.name
            doc_id = pdf_path.stem  # Filename without .pdf

            documents.append(DocumentInfo(
                document_id=doc_id,
                display_name=_format_display_name(filename),
                filename=filename,
                size_bytes=pdf_path.stat().st_size
            ))
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )

    # Sort by display name
    documents.sort(key=lambda d: d.display_name)

    logger.info(f"Listed {len(documents)} documents for user {user.get('id', 'unknown')}")
    return documents


def _find_pdf_for_document(document_id: str) -> Optional[Path]:
    """
    Find PDF file for a document_id.

    Handles two formats:
    1. Direct match: "BZ_VR1" → "BZ_VR1.pdf"
    2. Legal format: "157/2025 Sb." → searches for "Sb_{year}_{number}_*.pdf"

    Returns:
        Path to PDF file if found, None otherwise
    """
    # Try direct match first (e.g., "BZ_VR1" → "BZ_VR1.pdf")
    if DIRECT_ID_PATTERN.match(document_id):
        direct_path = PDF_BASE_DIR / f"{document_id}.pdf"
        if direct_path.is_file():
            return direct_path

    # Try legal format (e.g., "157/2025 Sb." → "Sb_2025_157_*.pdf")
    legal_match = LEGAL_ID_PATTERN.match(document_id)
    if legal_match:
        number, year = legal_match.groups()
        # Search for matching PDF with pattern Sb_{year}_{number}_*.pdf
        pattern = f"Sb_{year}_{number}_*.pdf"
        matches = list(PDF_BASE_DIR.glob(pattern))
        if matches:
            # Return first match (should be unique per document)
            return matches[0]

    return None


@router.get("/{document_id:path}/pdf")
async def get_pdf(
    document_id: str,
    user: Dict = Depends(get_current_user)
) -> FileResponse:
    """
    Serve PDF file for a document.

    Args:
        document_id: Document identifier in one of these formats:
            - Direct: "BZ_VR1" → "BZ_VR1.pdf"
            - Legal: "157/2025 Sb." → searches for "Sb_2025_157_*.pdf"

    Returns:
        PDF file stream with inline disposition for browser viewing

    Security:
        - Authentication required (JWT)
        - Pattern-based validation (direct or legal format)
        - Path traversal prevention via resolve() + prefix check
        - Only files from data/ directory allowed
    """
    from urllib.parse import unquote
    document_id = unquote(document_id)

    # Find PDF using pattern matching
    pdf_path = _find_pdf_for_document(document_id)

    if pdf_path is None:
        logger.info(f"PDF not found for document_id: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF not found for document: {document_id}"
        )

    # Resolve to absolute path
    pdf_path = pdf_path.resolve()

    # Security: Verify path is under allowed directory (prevent path traversal)
    if not str(pdf_path).startswith(str(PDF_BASE_DIR.resolve())):
        logger.warning(f"Path traversal attempt blocked: {document_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    pdf_filename = pdf_path.name
    logger.info(f"Serving PDF: {pdf_filename} (doc: {document_id}) to user {user.get('id', 'unknown')}")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_filename,
        headers={
            "Content-Disposition": f"inline; filename={pdf_filename}",
            "Cache-Control": "public, max-age=3600",  # 1 hour cache
        }
    )


def _sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename: keep alphanumeric, underscores, hyphens, dots."""
    # Remove path separators
    name = filename.replace("/", "_").replace("\\", "_")
    # Keep only safe characters
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user: Dict = Depends(get_current_user),
):
    """
    Upload a PDF and index it with VL pipeline, streaming progress via SSE.

    Accepts multipart/form-data with a PDF file.
    Returns SSE stream with progress events during indexing.
    """
    # Validate VL components are available
    if not _vl_components:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VL indexing pipeline not initialized"
        )

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large (max 100 MB)"
        )

    safe_filename = _sanitize_filename(file.filename)
    pdf_path = PDF_BASE_DIR / safe_filename

    # Check for duplicate
    if pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document already exists"
        )

    async def event_generator():
        jina_client = _vl_components["jina_client"]
        page_store = _vl_components["page_store"]
        vector_store = _vl_components["vector_store"]
        document_id = pdf_path.stem

        try:
            # 1. Save file
            pdf_path.write_bytes(content)
            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "uploading",
                    "percent": 100,
                    "message": "File saved",
                }),
            }

            # 2. Render pages
            import fitz
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            zoom = page_store.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            doc_dir = page_store.store_dir / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            page_ids = []
            for page_num in range(total_pages):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)
                page_number = page_num + 1
                img_path = page_store._image_path(document_id, page_number)
                pix.save(str(img_path))
                page_id = page_store.make_page_id(document_id, page_number)
                page_ids.append(page_id)

                percent = int((page_number / total_pages) * 100)
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "stage": "rendering",
                        "percent": percent,
                        "message": f"Rendering page {page_number}/{total_pages}",
                    }),
                }
                await asyncio.sleep(0)  # Yield control for SSE flush

            doc.close()

            # 3. Embed via Jina in batches
            batch_size = jina_client.batch_size
            total_batches = (len(page_ids) + batch_size - 1) // batch_size
            all_embeddings = []

            for batch_idx in range(total_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(page_ids))
                batch_page_ids = page_ids[start:end]

                page_images = []
                for pid in batch_page_ids:
                    page_images.append(page_store.get_image_bytes(pid))

                # Run synchronous Jina embedding in thread pool
                import numpy as np
                batch_embeddings = await asyncio.to_thread(
                    jina_client.embed_pages, page_images
                )
                all_embeddings.append(batch_embeddings)

                percent = int(((batch_idx + 1) / total_batches) * 100)
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "stage": "embedding",
                        "percent": percent,
                        "message": f"Embedding page {batch_idx + 1}/{total_batches}",
                    }),
                }

                # Rate-limit delay between Jina API calls
                if batch_idx + 1 < total_batches:
                    await asyncio.sleep(3)

            embeddings = np.vstack(all_embeddings)

            # 4. Store in PostgreSQL
            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "storing",
                    "percent": 50,
                    "message": "Storing embeddings",
                }),
            }

            pages = []
            for page_id in page_ids:
                doc_id, page_num_val = page_store.page_id_to_components(page_id)
                pages.append({
                    "page_id": page_id,
                    "document_id": doc_id,
                    "page_number": page_num_val,
                    "image_path": page_store.get_image_path(page_id),
                    "metadata": {},
                })

            # nest_asyncio is applied at startup, so sync wrapper works in async context
            vector_store.add_vl_pages(pages, embeddings)

            display_name = _format_display_name(safe_filename)
            yield {
                "event": "complete",
                "data": json.dumps({
                    "document_id": document_id,
                    "pages": total_pages,
                    "display_name": display_name,
                }),
            }

            logger.info(
                f"User {user.get('id', 'unknown')} uploaded and indexed "
                f"{safe_filename} ({total_pages} pages)"
            )

        except Exception as e:
            logger.error(
                "Upload indexing failed for %s (user %s): %s",
                safe_filename,
                user.get("id", "unknown"),
                e,
                exc_info=True,
            )
            # Clean up partial files on failure
            if pdf_path.exists():
                pdf_path.unlink()
            doc_dir = page_store.store_dir / document_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir, ignore_errors=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": str(e),
                    "stage": "indexing",
                }),
            }

    return EventSourceResponse(event_generator())
