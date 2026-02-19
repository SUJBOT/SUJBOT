# Image Upload & Indexing Design

**Date:** 2026-02-19
**Status:** Approved

## Goal

Allow users to upload images (PNG, JPG, TIFF, BMP, WebP) via the existing upload modal and have them indexed into the VL pipeline for semantic search, summarization, and entity extraction.

## Decisions

- **Multiple images → one document**: Images uploaded together are grouped as pages of a single document, ordered by filename.
- **Auto-generated name**: `"Images YYYY-MM-DD HH:MM"` (no user input required).
- **Full pipeline**: Embed + summarize + entity extract — same as PDF pages.

## Approach: Image-to-PDF Conversion

Convert images to a single PDF first, then run the existing indexing pipeline unchanged.

### Why This Approach

- Zero changes to the indexing pipeline (`render → embed → summarize → extract → store`).
- PyMuPDF already supports inserting images as PDF pages natively.
- Quality loss from image → PDF → re-rendered PNG at 150 DPI is negligible for typical uploads (scanned pages, photos of documents, diagrams).

### Changes

**Frontend (`UploadModal.tsx`):**
1. Add image extensions to `SUPPORTED_EXTENSIONS`: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp`.
2. Detect when selected files contain images. Group all images into a single batch upload.
3. Send images as multi-file FormData to new batch endpoint.
4. Non-image files continue through existing per-file upload flow unchanged.
5. Auto-generate document name displayed in upload progress.

**Backend (`document_converter.py`):**
1. Add `images_to_pdf(image_buffers, filenames) -> bytes` class method.
2. Uses PyMuPDF to insert each image as a full page (preserving aspect ratio).
3. Add image extensions to `SUPPORTED_EXTENSIONS`.

**Backend (`documents.py` route):**
1. New endpoint `POST /documents/upload-images` accepting multiple files + category + access_level.
2. Auto-generates document name and document_id.
3. Calls `DocumentConverter.images_to_pdf()`, saves PDF, then feeds into existing `_run_pipeline_task()`.

**i18n:** Update `cs.json` and `en.json` for new UI strings (image upload labels, progress messages).
