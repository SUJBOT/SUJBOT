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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
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

# Strong references for fire-and-forget cleanup tasks (prevents GC before completion)
_background_tasks: set = set()

# Debounced graph rebuild state
_rebuild_timer: Optional[asyncio.TimerHandle] = None
_rebuild_task: Optional[asyncio.Task] = None
_REBUILD_DELAY_SECONDS = 10.0

# Upload state TTL after completion (seconds)
_UPLOAD_STATE_TTL = 60.0


@dataclass
class UploadState:
    """State of an in-progress or recently completed upload."""

    document_id: str
    filename: str
    user_id: int
    category: str
    pdf_path: Path
    task: Optional[asyncio.Task] = None
    events: list = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None


# Active uploads keyed by user_id (one upload per user)
_active_uploads: Dict[int, UploadState] = {}


def set_vl_components(
    jina_client: Any,
    page_store: Any,
    vector_store: Any,
    summary_provider: Any = None,
    entity_extractor: Any = None,
    graph_storage: Any = None,
    community_detector: Any = None,
    community_summarizer: Any = None,
    graph_embedder: Any = None,
) -> None:
    """Set VL components for upload endpoint (called from main.py lifespan)."""
    _vl_components["jina_client"] = jina_client
    _vl_components["page_store"] = page_store
    _vl_components["vector_store"] = vector_store
    _vl_components["summary_provider"] = summary_provider
    _vl_components["entity_extractor"] = entity_extractor
    _vl_components["graph_storage"] = graph_storage
    _vl_components["community_detector"] = community_detector
    _vl_components["community_summarizer"] = community_summarizer
    _vl_components["graph_embedder"] = graph_embedder


def _schedule_graph_rebuild(document_id: str) -> None:
    """Schedule a debounced graph rebuild. Resets timer on each call."""
    global _rebuild_timer, _rebuild_task

    graph_storage = _vl_components.get("graph_storage")
    community_detector = _vl_components.get("community_detector")
    if not graph_storage or not community_detector:
        logger.debug("Graph rebuild skipped: graph_storage or community_detector not configured")
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop for graph rebuild scheduling")
        return

    # Cancel pending timer
    if _rebuild_timer is not None:
        _rebuild_timer.cancel()

    # Cancel in-progress rebuild (new data means stale results)
    if _rebuild_task and not _rebuild_task.done():
        _rebuild_task.cancel()
        logger.info("Cancelled in-progress graph rebuild (new document indexed)")

    def _on_rebuild_done(task: asyncio.Task) -> None:
        """Log exceptions from background graph rebuild tasks."""
        _background_tasks.discard(task)
        if task.cancelled():
            logger.debug("Graph rebuild task was cancelled")
            return
        exc = task.exception()
        if exc:
            logger.error("Background graph rebuild failed: %s", exc, exc_info=exc)

    def _fire():
        global _rebuild_task
        from src.graph.post_processor import rebuild_graph_communities

        _rebuild_task = asyncio.ensure_future(
            rebuild_graph_communities(
                graph_storage=graph_storage,
                community_detector=community_detector,
                community_summarizer=_vl_components.get("community_summarizer"),
                graph_embedder=_vl_components.get("graph_embedder"),
                llm_provider=_vl_components.get("summary_provider"),
                document_id=document_id,
            )
        )
        _background_tasks.add(_rebuild_task)
        _rebuild_task.add_done_callback(_on_rebuild_done)

    _rebuild_timer = loop.call_later(_REBUILD_DELAY_SECONDS, _fire)
    logger.info(
        f"Graph rebuild scheduled in {_REBUILD_DELAY_SECONDS}s (triggered by {document_id})"
    )


class DocumentInfo(BaseModel):
    """Document metadata for listing available PDFs."""

    document_id: str
    display_name: str
    filename: str
    size_bytes: int
    category: str = "documentation"


async def index_document_pipeline(
    pdf_path: Path,
    document_id: str,
    jina_client: Any,
    page_store: Any,
    vector_store: Any,
    summary_provider: Any = None,
    entity_extractor: Any = None,
    graph_storage: Any = None,
):
    """
    Async generator that renders, embeds, and stores a PDF document.

    Yields SSE-formatted dicts with 'event' and 'data' keys for progress/complete/error.
    Shared by upload_document and admin reindex endpoints.
    """
    import fitz
    import numpy as np

    doc = fitz.open(str(pdf_path))
    try:
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
                "data": json.dumps(
                    {
                        "stage": "rendering",
                        "percent": percent,
                        "message": f"Rendering page {page_number}/{total_pages}",
                    }
                ),
            }
            await asyncio.sleep(0)
    finally:
        doc.close()

    # Run embedding and summarization concurrently (different API providers).
    # Entity extraction must run AFTER store — graph.entities has FK to vl_pages.
    # Progress events are funneled through an asyncio.Queue.
    progress_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def embed_task() -> np.ndarray:
        """Embed all pages via Jina API."""
        batch_size = jina_client.batch_size
        total_batches = (len(page_ids) + batch_size - 1) // batch_size
        all_embeds = []

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(page_ids))
            batch_page_ids = page_ids[start:end]

            try:
                page_images = [page_store.get_image_bytes(pid) for pid in batch_page_ids]
            except (FileNotFoundError, OSError) as e:
                raise RuntimeError(
                    f"Failed to read page images for batch {batch_idx + 1}: {e}"
                ) from e

            try:
                batch_embeddings = await asyncio.to_thread(jina_client.embed_pages, page_images)
            except Exception as e:
                raise RuntimeError(
                    f"Embedding API failed on batch {batch_idx + 1}/{total_batches}: {e}"
                ) from e
            all_embeds.append(batch_embeddings)

            percent = int(((batch_idx + 1) / total_batches) * 100)
            await progress_queue.put(
                {
                    "stage": "embedding",
                    "percent": percent,
                    "message": f"Embedding batch {batch_idx + 1}/{total_batches}",
                }
            )

            if batch_idx + 1 < total_batches:
                await asyncio.sleep(3)

        return np.vstack(all_embeds)

    async def summarize_task() -> tuple[dict, str | None]:
        """Summarize pages via LLM. Returns (summaries_dict, model_name)."""
        sums: dict = {}
        try:
            prompt_path = (
                Path(__file__).resolve().parent.parent.parent / "prompts" / "vl_page_summary.txt"
            )
            prompt_text = prompt_path.read_text(encoding="utf-8")
            m_name = summary_provider.get_model_name()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError, AttributeError) as e:
            logger.error(f"Summary setup failed, skipping summarization: {e}")
            await progress_queue.put(
                {
                    "stage": "summarizing",
                    "percent": 100,
                    "message": "Summarization skipped: setup failed",
                }
            )
            return sums, None

        max_consecutive_failures = 3
        consecutive_failures = 0

        for i, page_id in enumerate(page_ids):
            try:
                image_b64 = page_store.get_image_base64(page_id)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{page_store.image_format}",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                response = await asyncio.to_thread(
                    summary_provider.create_message,
                    messages=messages,
                    tools=[],
                    system="",
                    max_tokens=500,
                    temperature=0.0,
                )
                text = response.text.strip() if response.text else None
                if text:
                    sums[page_id] = text
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception as e:
                logger.warning(f"Failed to summarize {page_id}: {e}")
                consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"Aborting summarization after {max_consecutive_failures} "
                    f"consecutive failures (completed {i + 1}/{len(page_ids)})"
                )
                break

            percent = int(((i + 1) / len(page_ids)) * 100)
            await progress_queue.put(
                {
                    "stage": "summarizing",
                    "percent": percent,
                    "message": f"Summarizing page {i + 1}/{len(page_ids)}",
                }
            )

        return sums, m_name

    # Launch embed + summarize concurrently
    embed_future = asyncio.create_task(embed_task())
    summarize_future = asyncio.create_task(summarize_task()) if summary_provider else None

    tasks = [t for t in [embed_future, summarize_future] if t is not None]
    done_event = asyncio.Event()

    async def _wait_all():
        await asyncio.gather(*tasks, return_exceptions=True)
        done_event.set()

    waiter = asyncio.create_task(_wait_all())

    try:
        # Yield progress from queue until embed + summarize complete
        while not done_event.is_set():
            try:
                item = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield {
                    "event": "progress",
                    "data": json.dumps(item),
                }
            except asyncio.TimeoutError:
                continue

        # Drain remaining queue items
        while True:
            try:
                item = progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            yield {
                "event": "progress",
                "data": json.dumps(item),
            }
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected — cancel all running tasks
        for t in tasks:
            t.cancel()
        waiter.cancel()
        logger.info("Indexing pipeline cancelled")
        raise

    # Get results — embedding is mandatory, summarization is optional
    embed_exc = (
        embed_future.exception() if embed_future.done() and not embed_future.cancelled() else None
    )
    if embed_exc:
        raise RuntimeError(f"Embedding failed: {embed_exc}") from embed_exc
    embeddings = embed_future.result()

    summaries: dict = {}
    model_name = None
    if summarize_future:
        sum_exc = (
            summarize_future.exception()
            if summarize_future.done() and not summarize_future.cancelled()
            else None
        )
        if sum_exc:
            logger.error(
                f"Summarization failed, continuing without summaries: {sum_exc}", exc_info=sum_exc
            )
        else:
            summaries, model_name = summarize_future.result()

    # Store in PostgreSQL (needs embeddings + summaries)
    yield {
        "event": "progress",
        "data": json.dumps(
            {
                "stage": "storing",
                "percent": 50,
                "message": "Storing embeddings",
            }
        ),
    }

    pages = []
    for page_id in page_ids:
        doc_id, page_num_val = page_store.page_id_to_components(page_id)
        metadata: dict = {}
        if summaries.get(page_id) and model_name:
            metadata["page_summary"] = summaries[page_id]
            metadata["summary_model"] = model_name
        pages.append(
            {
                "page_id": page_id,
                "document_id": doc_id,
                "page_number": page_num_val,
                "image_path": page_store.get_image_path(page_id),
                "metadata": metadata,
            }
        )

    vector_store.add_vl_pages(pages, embeddings)

    # Entity extraction (sequential, after store — FK on vl_pages)
    if entity_extractor and graph_storage:
        max_consecutive_failures = 3
        consecutive_failures = 0

        for i, page_id in enumerate(page_ids):
            try:
                result = await asyncio.to_thread(
                    entity_extractor.extract_from_page, page_id, page_store
                )
                entities = result.get("entities", [])
                relationships = result.get("relationships", [])

                if entities:
                    await graph_storage.async_add_entities(
                        entities,
                        document_id,
                        source_page_id=page_id,
                    )
                if relationships:
                    await graph_storage.async_add_relationships(
                        relationships,
                        document_id,
                        source_page_id=page_id,
                    )
                consecutive_failures = 0

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Entity extraction failed for {page_id}: {e}", exc_info=True)
                consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"Aborting entity extraction after {max_consecutive_failures} "
                    f"consecutive failures (completed {i + 1}/{len(page_ids)})"
                )
                yield {
                    "event": "warning",
                    "data": json.dumps(
                        {
                            "stage": "graph_extraction",
                            "message": f"Entity extraction aborted after {max_consecutive_failures} consecutive failures",
                        }
                    ),
                }
                break

            percent = int(((i + 1) / len(page_ids)) * 100)
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "stage": "graph_extraction",
                        "percent": percent,
                        "message": f"Extracting entities from page {i + 1}/{len(page_ids)}",
                    }
                ),
            }
            await asyncio.sleep(0)

    yield {
        "event": "complete",
        "data": json.dumps(
            {
                "document_id": document_id,
                "pages": total_pages,
                "display_name": _format_display_name(pdf_path.name),
            }
        ),
    }


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
async def list_documents(user: Dict = Depends(get_current_user)) -> List[DocumentInfo]:
    """
    List all available PDF documents.

    Returns:
        List of document metadata with IDs suitable for the PDF viewer.

    Security:
        - Authentication required (JWT)
        - Only returns documents in the allowed data/ directory
    """
    documents = []

    # Fetch categories from database if available
    categories: Dict[str, str] = {}
    vector_store = _vl_components.get("vector_store")
    if vector_store:
        try:
            await vector_store._ensure_pool()
            async with vector_store.pool.acquire() as conn:
                rows = await conn.fetch("SELECT document_id, category FROM vectors.documents")
                categories = {row["document_id"]: row["category"] for row in rows}
        except Exception as e:
            logger.warning(f"Failed to fetch document categories: {e}", exc_info=True)

    try:
        for pdf_path in PDF_BASE_DIR.glob("*.pdf"):
            filename = pdf_path.name
            doc_id = pdf_path.stem  # Filename without .pdf

            documents.append(
                DocumentInfo(
                    document_id=doc_id,
                    display_name=_format_display_name(filename),
                    filename=filename,
                    size_bytes=pdf_path.stat().st_size,
                    category=categories.get(doc_id, "documentation"),
                )
            )
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list documents"
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
async def get_pdf(document_id: str, user: Dict = Depends(get_current_user)) -> FileResponse:
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
            detail=f"PDF not found for document: {document_id}",
        )

    # Resolve to absolute path
    pdf_path = pdf_path.resolve()

    # Security: Verify path is under allowed directory (prevent path traversal)
    if not str(pdf_path).startswith(str(PDF_BASE_DIR.resolve())):
        logger.warning(f"Path traversal attempt blocked: {document_id}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    pdf_filename = pdf_path.name
    logger.info(
        f"Serving PDF: {pdf_filename} (doc: {document_id}) to user {user.get('id', 'unknown')}"
    )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_filename,
        headers={
            "Content-Disposition": f"inline; filename={pdf_filename}",
            "Cache-Control": "public, max-age=3600",  # 1 hour cache
        },
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


def _cleanup_upload_files(pdf_path: Path, page_store: Any, document_id: str) -> None:
    """Clean up uploaded PDF and rendered page images."""
    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError as e:
            logger.error("Failed to delete uploaded PDF %s: %s", pdf_path, e)
    doc_dir = page_store.store_dir / document_id
    if doc_dir.exists():
        try:
            shutil.rmtree(doc_dir)
        except OSError as e:
            logger.error("Failed to clean up page image directory %s: %s", doc_dir, e)


async def _cleanup_upload_db(vector_store: Any, document_id: str) -> None:
    """Delete all DB entries for a partially indexed document."""
    try:
        await vector_store._ensure_pool()
        async with vector_store.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM graph.entities WHERE document_id = $1", document_id)
                await conn.execute(
                    "DELETE FROM vectors.vl_pages WHERE document_id = $1", document_id
                )
                await conn.execute(
                    "DELETE FROM vectors.documents WHERE document_id = $1", document_id
                )
        logger.info("DB cleanup completed for upload: %s", document_id)
    except Exception as e:
        logger.error("DB cleanup failed for %s: %s", document_id, e, exc_info=True)


def _schedule_upload_state_cleanup(user_id: int, state: UploadState) -> None:
    """Remove finished upload state after TTL so the user can upload again."""

    def _remove():
        current = _active_uploads.get(user_id)
        if current is state:
            del _active_uploads[user_id]
            logger.debug("Cleaned up upload state for user %d", user_id)

    try:
        loop = asyncio.get_running_loop()
        loop.call_later(_UPLOAD_STATE_TTL, _remove)
    except RuntimeError:
        pass


async def _run_pipeline_task(state: UploadState, content: bytes) -> None:
    """Background task: runs the indexing pipeline and updates UploadState.

    Writes the PDF to disk, registers the document category, iterates the
    shared index_document_pipeline generator, and appends every SSE event
    to state.events.  On success it schedules a graph rebuild; on cancel or
    error it cleans up files and DB.
    """
    vector_store = _vl_components["vector_store"]
    jina_client = _vl_components["jina_client"]
    page_store = _vl_components["page_store"]
    summary_provider = _vl_components.get("summary_provider")
    entity_extractor = _vl_components.get("entity_extractor")
    graph_storage = _vl_components.get("graph_storage")
    document_id = state.document_id

    try:
        # 1. Save file to disk
        state.pdf_path.write_bytes(content)
        del content  # free memory
        state.events.append(
            {
                "event": "progress",
                "data": json.dumps(
                    {
                        "stage": "uploading",
                        "percent": 100,
                        "message": "File saved",
                    }
                ),
            }
        )

        # 1b. Register document category
        display_name = _format_display_name(state.pdf_path.name)
        await vector_store._ensure_pool()
        async with vector_store.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO vectors.documents (document_id, category, display_name)
                VALUES ($1, $2, $3)
                ON CONFLICT (document_id) DO UPDATE SET category = $2, display_name = $3
                """,
                document_id,
                state.category,
                display_name,
            )

        # 2-5. Render, embed, summarize, store, extract entities
        async for event in index_document_pipeline(
            state.pdf_path,
            document_id,
            jina_client,
            page_store,
            vector_store,
            summary_provider=summary_provider,
            entity_extractor=entity_extractor,
            graph_storage=graph_storage,
        ):
            state.events.append(event)

        logger.info(
            "User %d uploaded and indexed %s (%s, category=%s)",
            state.user_id,
            state.filename,
            document_id,
            state.category,
        )

        # Schedule debounced graph rebuild (communities + dedup)
        _schedule_graph_rebuild(document_id)

    except asyncio.CancelledError:
        logger.info("Upload cancelled for %s (user %d)", state.filename, state.user_id)
        _cleanup_upload_files(state.pdf_path, page_store, document_id)
        await _cleanup_upload_db(vector_store, document_id)
        state.error = "cancelled"
        state.events.append(
            {
                "event": "error",
                "data": json.dumps({"message": "Upload cancelled", "stage": "cancelled"}),
            }
        )

    except Exception as e:
        logger.error(
            "Upload indexing failed for %s (user %d): %s",
            state.filename,
            state.user_id,
            e,
            exc_info=True,
        )
        _cleanup_upload_files(state.pdf_path, page_store, document_id)
        await _cleanup_upload_db(vector_store, document_id)
        state.error = str(e)
        state.events.append(
            {
                "event": "error",
                "data": json.dumps({"message": str(e), "stage": "indexing"}),
            }
        )

    finally:
        state.done = True
        _schedule_upload_state_cleanup(state.user_id, state)


async def _stream_upload_state(state: UploadState):
    """SSE generator that polls UploadState and yields events.

    On client disconnect (GeneratorExit) this simply stops streaming —
    the pipeline background task keeps running.
    """
    cursor = 0
    try:
        while True:
            # Yield any new events since last poll
            while cursor < len(state.events):
                yield state.events[cursor]
                cursor += 1

            # All events yielded and pipeline finished → done
            if state.done:
                break

            await asyncio.sleep(0.3)
    except (asyncio.CancelledError, GeneratorExit):
        logger.debug(
            "Upload SSE stream disconnected for user %d (pipeline continues)",
            state.user_id,
        )


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    category: str = Form("documentation"),
    user: Dict = Depends(get_current_user),
):
    """
    Upload a PDF and index it as a background task, streaming progress via SSE.

    The indexing pipeline runs independently of the SSE connection — a page
    refresh reconnects to the same in-progress upload via GET /upload-status.
    Only one upload per user at a time (409 if already active).
    """
    # Validate category
    if category not in ("documentation", "legislation"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category must be 'documentation' or 'legislation'",
        )
    # Validate VL components are available
    if not _vl_components:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VL indexing pipeline not initialized",
        )

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported"
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large (max 100 MB)",
        )

    safe_filename = _sanitize_filename(file.filename)
    pdf_path = PDF_BASE_DIR / safe_filename

    # Check for duplicate file on disk
    if pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document already exists")

    # Reject if upload already active for this user
    user_id = user["id"]
    existing = _active_uploads.get(user_id)
    if existing and not existing.done:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An upload is already in progress",
        )

    # Create upload state and start background task
    state = UploadState(
        document_id=pdf_path.stem,
        filename=safe_filename,
        user_id=user_id,
        category=category,
        pdf_path=pdf_path,
    )
    task = asyncio.create_task(_run_pipeline_task(state, content))
    state.task = task
    _active_uploads[user_id] = state

    # Prevent GC of the background task
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return EventSourceResponse(_stream_upload_state(state))


@router.get("/upload-status")
async def upload_status(user: Dict = Depends(get_current_user)):
    """Return SSE stream for an active upload, or 204 if none."""
    state = _active_uploads.get(user["id"])
    if not state:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return EventSourceResponse(_stream_upload_state(state))


@router.post("/upload-cancel")
async def cancel_upload(user: Dict = Depends(get_current_user)):
    """Cancel an active upload and clean up."""
    user_id = user["id"]
    state = _active_uploads.get(user_id)

    if not state or state.done:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active upload to cancel",
        )

    document_id = state.document_id

    # Remove from active uploads immediately (allows re-upload)
    del _active_uploads[user_id]

    # Cancel the background task (triggers cleanup in _run_pipeline_task)
    if state.task and not state.task.done():
        state.task.cancel()

    return {"status": "cancelled", "document_id": document_id}
