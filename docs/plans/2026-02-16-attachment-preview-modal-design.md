# Attachment Preview Modal Design

**Date:** 2026-02-16
**Status:** Approved

## Problem

Attachments in sent messages and input bar show only metadata chips (filename, size). Users cannot preview or re-view attachment content after sending. Base64 data is discarded after send.

## Solution

Persistent filesystem storage for attachments + GET endpoint + frontend preview modal.

## Backend

### 1. Attachment Storage

- **Path:** `data/attachments/{conversation_id}/{attachment_id}.{ext}`
- On message send, decode base64 and write to disk
- Add `attachment_id` (UUID) to existing `messages.metadata.attachments[]` entries
- Flat per-conversation structure, UUID filenames prevent collisions

### 2. GET Endpoint

- `GET /api/attachments/{conversation_id}/{attachment_id}`
- Auth middleware verifies conversation ownership
- Returns binary file with correct Content-Type header
- Streams file directly (no base64 re-encoding)

### 3. Cleanup

- When conversation is deleted, remove `data/attachments/{conversation_id}/` directory

## Frontend

### 4. AttachmentPreviewModal Component

- Opens on click of attachment chip (both in sent messages and input bar)
- **Images:** `<img>` with `object-fit: contain`, fills modal area
- **PDF:** Inline embed or download link
- **Documents (DOCX, TXT, MD, etc.):** Filename + icon + download button
- Backdrop: `bg-black/80`, `backdrop-blur-sm`
- Close: Escape key, click outside, X button
- Arrow navigation for multiple attachments

### 5. Data Flow

- **Pre-send (input bar):** base64 in React state -> `URL.createObjectURL(blob)` for instant preview
- **Post-send (message):** `attachment_id` in metadata -> fetch from `/api/attachments/{conv_id}/{att_id}`

### 6. i18n

Add keys: `attachments.preview`, `attachments.download`, `attachments.close`

## Files to Modify

### Backend
- `backend/main.py` — save attachments to disk in chat endpoint, add GET route
- `backend/routes/conversations.py` — cleanup attachments on conversation delete
- `backend/models.py` — no changes needed (existing AttachmentData model sufficient)

### Frontend
- New: `frontend/src/components/chat/AttachmentPreviewModal.tsx`
- `frontend/src/components/chat/ChatMessage.tsx` — make chips clickable
- `frontend/src/components/chat/ChatInput.tsx` — make chips clickable
- `frontend/src/types/index.ts` — add `attachmentId` to `MessageAttachmentMeta`
- `frontend/src/services/api.ts` — add `getAttachmentUrl()` helper
- `frontend/src/i18n/locales/en.json` — add preview keys
- `frontend/src/i18n/locales/cs.json` — add preview keys
