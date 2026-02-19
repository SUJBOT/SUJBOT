# Image Upload & Indexing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to upload images (PNG, JPG, TIFF, BMP, WebP) that get combined into a single PDF document and indexed through the existing VL pipeline.

**Architecture:** Images are converted to a multi-page PDF via PyMuPDF (`fitz`), then the existing `_run_pipeline_task()` handles everything else unchanged. Frontend groups image files into a batch, sends to a new `/documents/upload-images` endpoint that accepts multiple files.

**Tech Stack:** PyMuPDF (fitz), FastAPI, React, TypeScript

---

### Task 1: Add `images_to_pdf()` to DocumentConverter

**Files:**
- Modify: `src/vl/document_converter.py:23` (SUPPORTED_EXTENSIONS)
- Modify: `src/vl/document_converter.py:41` (DocumentConverter class — add method)
- Test: `tests/vl/test_document_converter.py` (create if needed)

**Step 1: Write the failing test**

Create `tests/vl/test_document_converter.py` (or add to existing). The test creates tiny PNG images in memory and calls `images_to_pdf()`:

```python
import fitz
import pytest
from src.vl.document_converter import DocumentConverter, IMAGE_EXTENSIONS

def _make_png(width=100, height=80, color=(255, 0, 0)):
    """Create a minimal PNG image in memory using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(fitz.Rect(0, 0, width, height), color=color, fill=color)
    # Render page to PNG pixmap
    pix = page.get_pixmap()
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


class TestImagesToPdf:
    def test_single_image(self):
        png = _make_png()
        result = DocumentConverter.images_to_pdf([png], ["test.png"])
        assert isinstance(result, bytes)
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) == 1
        doc.close()

    def test_multiple_images_become_pages(self):
        images = [_make_png(color=(255, 0, 0)), _make_png(color=(0, 255, 0)), _make_png(color=(0, 0, 255))]
        result = DocumentConverter.images_to_pdf(images, ["a.png", "b.png", "c.png"])
        doc = fitz.open(stream=result, filetype="pdf")
        assert len(doc) == 3
        doc.close()

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No images"):
            DocumentConverter.images_to_pdf([], [])

    def test_corrupt_image_raises_conversion_error(self):
        from src.exceptions import ConversionError
        with pytest.raises(ConversionError):
            DocumentConverter.images_to_pdf([b"not an image"], ["bad.png"])

    def test_image_extensions_constant(self):
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".tif" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/vl/test_document_converter.py -v`
Expected: FAIL — `IMAGE_EXTENSIONS` not defined, `images_to_pdf` not defined.

**Step 3: Implement**

In `src/vl/document_converter.py`:

1. Add constant after `SUPPORTED_EXTENSIONS` (line 23):
```python
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
```

2. Add static method to `DocumentConverter` class (after `check_dependencies`, around line 126):
```python
@staticmethod
def images_to_pdf(image_buffers: list[bytes], filenames: list[str]) -> bytes:
    """Combine multiple images into a single PDF (one image per page).

    Each image is inserted as a full page preserving its aspect ratio.

    Args:
        image_buffers: Raw image bytes (PNG, JPG, TIFF, BMP, WebP)
        filenames: Corresponding filenames (for error messages)

    Returns:
        PDF bytes

    Raises:
        ValueError: If image_buffers is empty
        ConversionError: If any image cannot be processed
    """
    if not image_buffers:
        raise ValueError("No images provided")

    doc = fitz.open()
    try:
        for i, (img_bytes, fname) in enumerate(zip(image_buffers, filenames)):
            try:
                img_doc = fitz.open(stream=img_bytes, filetype="png")  # fitz auto-detects format
                if len(img_doc) == 0:
                    raise ConversionError(
                        f"Image has no pages: {fname}",
                        details={"filename": fname, "index": i},
                    )
                # Get image dimensions from first page
                img_rect = img_doc[0].rect
                # Create page matching image aspect ratio
                page = doc.new_page(width=img_rect.width, height=img_rect.height)
                page.insert_image(page.rect, stream=img_bytes)
                img_doc.close()
            except ConversionError:
                raise
            except Exception as e:
                raise ConversionError(
                    f"Failed to process image {fname}: {e}",
                    details={"filename": fname, "index": i},
                    cause=e,
                )
        return doc.tobytes()
    finally:
        doc.close()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/vl/test_document_converter.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/vl/document_converter.py tests/vl/test_document_converter.py
git commit -m "feat: add images_to_pdf() conversion method for image upload support"
```

---

### Task 2: Add backend endpoint `/documents/upload-images`

**Files:**
- Modify: `backend/routes/documents.py` (add new endpoint after existing `/upload`)
- Test: `tests/backend/test_upload_images.py` (create)

**Step 1: Write the failing test**

Create `tests/backend/test_upload_images.py`:

```python
"""Tests for the image batch upload endpoint."""
import io
import pytest
import fitz
from unittest.mock import AsyncMock, MagicMock, patch

def _make_png(width=100, height=80):
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(fitz.Rect(0, 0, width, height), color=(255, 0, 0), fill=(255, 0, 0))
    pix = page.get_pixmap()
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes

@pytest.fixture
def png_bytes():
    return _make_png()

class TestUploadImagesValidation:
    """Test validation logic without actually running the pipeline."""

    def test_no_files_returns_400(self, client, auth_headers):
        """Endpoint requires at least one image file."""
        response = client.post(
            "/documents/upload-images",
            headers=auth_headers,
            data={"category": "documentation", "access_level": "public"},
        )
        assert response.status_code == 400

    def test_non_image_file_returns_400(self, client, auth_headers):
        """Rejects non-image files."""
        response = client.post(
            "/documents/upload-images",
            headers=auth_headers,
            files=[("files", ("doc.pdf", b"%PDF-1.4 fake", "application/pdf"))],
            data={"category": "documentation", "access_level": "public"},
        )
        assert response.status_code == 400

    def test_invalid_category_returns_400(self, client, auth_headers, png_bytes):
        response = client.post(
            "/documents/upload-images",
            headers=auth_headers,
            files=[("files", ("img.png", png_bytes, "image/png"))],
            data={"category": "invalid", "access_level": "public"},
        )
        assert response.status_code == 400
```

Note: These tests depend on the project's test fixtures (`client`, `auth_headers`). If these don't exist yet, create minimal stubs or use FastAPI TestClient with mocked auth.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/test_upload_images.py -v`
Expected: FAIL — endpoint not found (404).

**Step 3: Implement the endpoint**

Add to `backend/routes/documents.py` after the existing `upload_document` endpoint (~line 1008):

```python
@router.post("/upload-images")
async def upload_images(
    request: Request,
    files: List[UploadFile] = File(...),
    category: str = Form("documentation"),
    access_level: str = Form("public"),
    user: Dict = Depends(get_current_user),
):
    """
    Upload multiple image files as a single document.

    Images are combined into a PDF (one page per image) and indexed
    through the standard VL pipeline. The document name is auto-generated
    as "Images YYYY-MM-DD HH:MM".
    """
    from datetime import datetime
    from src.vl.document_converter import IMAGE_EXTENSIONS, DocumentConverter

    # Validate category
    if category not in ("documentation", "legislation"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Category must be 'documentation' or 'legislation'",
        )
    # Validate access level
    if access_level not in ("public", "secret"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Access level must be 'public' or 'secret'",
        )
    # Must have at least one file
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image file is required",
        )
    # Validate VL components
    if not _get_vl_components_from_deps():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VL indexing pipeline not initialized",
        )

    # Validate all files are images and read content
    image_buffers: list[bytes] = []
    filenames: list[str] = []
    total_size = 0

    for f in files:
        if not f.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All files must have filenames",
            )
        ext = Path(f.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format: {ext}. Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}",
            )
        content = await f.read()
        total_size += len(content)
        if total_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Total file size too large (max 100 MB)",
            )
        image_buffers.append(content)
        filenames.append(f.filename)

    # Sort by filename for deterministic page order
    paired = sorted(zip(filenames, image_buffers), key=lambda x: x[0])
    filenames, image_buffers = [list(t) for t in zip(*paired)]

    # Generate document name and path
    now = datetime.now()
    doc_name = f"Images_{now.strftime('%Y-%m-%d_%H-%M')}"
    safe_stem = _sanitize_filename(doc_name)
    pdf_path = PDF_BASE_DIR / f"{safe_stem}.pdf"

    # Ensure unique path (append counter if needed)
    counter = 1
    while pdf_path.exists():
        safe_stem = _sanitize_filename(f"{doc_name}_{counter}")
        pdf_path = PDF_BASE_DIR / f"{safe_stem}.pdf"
        counter += 1

    # Reject if upload already active for this user
    user_id = user["id"]
    existing = _active_uploads.get(user_id)
    if existing and not existing.done:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An upload is already in progress",
        )

    # Convert images to PDF
    try:
        pdf_bytes = DocumentConverter.images_to_pdf(image_buffers, filenames)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process images: {sanitize_error(e)}",
        )
    del image_buffers  # free memory

    # Create upload state and start background task
    state = UploadState(
        document_id=pdf_path.stem,
        filename=f"{safe_stem}.pdf",
        user_id=user_id,
        category=category,
        pdf_path=pdf_path,
        access_level=access_level,
        file_ext=".pdf",  # Already converted to PDF
    )
    task = asyncio.create_task(_run_pipeline_task(state, pdf_bytes))
    state.task = task
    _active_uploads[user_id] = state

    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return EventSourceResponse(_stream_upload_state(state))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/test_upload_images.py -v`
Expected: PASS (or adjust test fixtures as needed for the test environment).

**Step 5: Commit**

```bash
git add backend/routes/documents.py tests/backend/test_upload_images.py
git commit -m "feat: add /documents/upload-images endpoint for batch image upload"
```

---

### Task 3: Add `uploadImages()` to frontend API service

**Files:**
- Modify: `frontend/src/services/api.ts:354` (add method after `uploadDocument`)

**Step 1: Add the method**

Add after `uploadDocument()` method (around line 404):

```typescript
/**
 * Upload multiple images as a single document, streaming indexing progress via SSE
 */
async *uploadImages(
  files: File[],
  signal?: AbortSignal,
  category?: string,
  accessLevel?: string
): AsyncGenerator<SSEEvent, void, unknown> {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  if (category) formData.append('category', category);
  if (accessLevel) formData.append('access_level', accessLevel);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/documents/upload-images`, {
      method: 'POST',
      credentials: 'include',
      body: formData,
      signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') return;
    yield {
      event: 'error',
      data: { error: `Upload failed: ${(error as Error).message}`, type: 'NetworkError' },
    };
    return;
  }

  if (!response.ok) {
    let detail = `Upload failed (${response.status})`;
    try {
      const errorData = await response.json();
      detail = errorData.detail || detail;
    } catch (e) {
      console.warn('Failed to parse upload error response:', e);
    }
    yield {
      event: 'error',
      data: { error: detail, type: 'HTTPError', status: response.status },
    };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    yield { event: 'error', data: { error: 'No response body', type: 'NoResponseBody' } };
    return;
  }

  yield* parseSSEStream(reader, { timeoutMs: 10 * 60 * 1000, abortSignal: signal });
}
```

**Step 2: Commit**

```bash
git add frontend/src/services/api.ts
git commit -m "feat: add uploadImages() API method for batch image upload"
```

---

### Task 4: Update UploadModal to support image files

**Files:**
- Modify: `frontend/src/components/upload/UploadModal.tsx`
- Modify: `frontend/src/i18n/locales/en.json`
- Modify: `frontend/src/i18n/locales/cs.json`

**Step 1: Update i18n strings**

In `en.json`, in the `documentBrowser` section, update/add:
```json
"unsupportedFormat": "Unsupported format. Supported: PDF, DOCX, TXT, Markdown, HTML, LaTeX, images (PNG, JPG, TIFF, BMP, WebP)",
"dropzoneFormats": "PDF, DOCX, TXT, MD, HTML, LaTeX, PNG, JPG, TIFF, BMP, WebP (max 100 MB)",
"imagesBatchName": "Images {{datetime}}",
"convertingImages": "Converting images to document..."
```

In `cs.json`, in the `documentBrowser` section, update/add:
```json
"unsupportedFormat": "Nepodporovaný formát. Podporováno: PDF, DOCX, TXT, Markdown, HTML, LaTeX, obrázky (PNG, JPG, TIFF, BMP, WebP)",
"dropzoneFormats": "PDF, DOCX, TXT, MD, HTML, LaTeX, PNG, JPG, TIFF, BMP, WebP (max 100 MB)",
"imagesBatchName": "Obrázky {{datetime}}",
"convertingImages": "Převod obrázků na dokument..."
```

**Step 2: Update UploadModal.tsx**

Key changes to `frontend/src/components/upload/UploadModal.tsx`:

1. **Add image extensions to constants** (line 36):
```typescript
const SUPPORTED_EXTENSIONS = [
  '.pdf', '.docx', '.txt', '.md', '.html', '.htm', '.tex', '.latex',
  '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp',
];
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp']);
```

2. **Update `<input>` accept attribute** (line 378):
```typescript
accept=".pdf,.docx,.txt,.md,.html,.htm,.tex,.latex,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
```

3. **Add helper to detect image files**:
```typescript
const isImageFile = (file: File): boolean => {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return IMAGE_EXTENSIONS.has(ext);
};
```

4. **Modify `handleUpload`** to detect image batches and route them to `uploadImages()`:

In the upload loop, before processing each file, check if there are image entries. Group all images and send them via `apiService.uploadImages()` as a single batch, while non-image files continue through `apiService.uploadDocument()` individually.

The logic:
- Partition `files` into `imageEntries` and `docEntries`
- If `imageEntries.length > 0`: upload all images as one batch via `uploadImages()`
- Then upload each `docEntry` individually via existing `uploadDocument()`
- Progress tracking: image batch counts as 1 "file" in the progress bar

5. **Use `ImageIcon` from lucide-react** for image file entries in the file list (instead of `FileText`):
```typescript
import { Image as ImageIcon } from 'lucide-react';
// In file list row:
{isImageFile(entry.file) ? (
  <ImageIcon size={16} className="text-accent-400 dark:text-accent-500 flex-shrink-0" />
) : (
  <FileText size={16} className="text-accent-400 dark:text-accent-500 flex-shrink-0" />
)}
```

**Step 3: Verify frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds with no TypeScript errors.

**Step 4: Commit**

```bash
git add frontend/src/components/upload/UploadModal.tsx frontend/src/services/api.ts frontend/src/i18n/locales/en.json frontend/src/i18n/locales/cs.json
git commit -m "feat: update upload modal to support image file uploads (PNG, JPG, TIFF, BMP, WebP)"
```

---

### Task 5: Manual integration test

**Step 1: Build and deploy**

```bash
# Build frontend
cd frontend && npm run build && cd ..
docker build -t sujbot-frontend --target production --build-arg VITE_API_BASE_URL="" -f docker/frontend/Dockerfile frontend/

# Build backend
docker build -t sujbot-backend -f docker/backend/Dockerfile .

# Recreate containers (follow CLAUDE.md deploy instructions)
```

**Step 2: Test in browser**

1. Open upload modal
2. Drop 3 PNG images → verify they appear in file list with image icons
3. Set category and access level
4. Click upload → verify progress shows "Converting images to document..." then normal pipeline stages
5. Verify document appears in document list with auto-generated name
6. Ask the agent a question about the image content → verify it can find and reference the images

**Step 3: Test edge cases**

- Upload mix of images + PDF → images batch uploaded separately, PDF uploaded individually
- Upload single image → works as 1-page document
- Upload corrupt image → error message shown
- Drop unsupported file type (.exe) → skipped with warning
