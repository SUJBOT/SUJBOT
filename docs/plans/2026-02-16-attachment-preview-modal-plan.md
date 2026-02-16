# Attachment Preview Modal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add persistent attachment storage and a preview modal that opens when clicking attachment chips in sent messages or input bar.

**Architecture:** Backend saves attachment files to `data/attachments/{conversation_id}/` on message send, exposes GET endpoint for retrieval. Frontend adds `AttachmentPreviewModal` component shared by both ChatMessage and ChatInput chips. Pre-send attachments use in-memory base64; post-send use backend URL.

**Tech Stack:** FastAPI (backend), React + TypeScript (frontend), Tailwind CSS, lucide-react icons, react-i18next

---

### Task 1: Backend — Save attachments to filesystem

**Files:**
- Modify: `backend/main.py:855-866` (attachment metadata save)

**Step 1: Add attachment saving logic**

In `backend/main.py`, add a helper function and modify the attachment metadata block (lines 855-866) to also save files to disk and include `attachment_id` in metadata.

Add this import at the top of `backend/main.py` (near other imports):

```python
import shutil
```

Add this helper function (before the chat endpoint):

```python
ATTACHMENTS_DIR = Path("data/attachments")


def _save_attachment_files(
    conversation_id: str,
    attachments: list,
) -> list[dict]:
    """Save attachment files to disk and return metadata with attachment_ids."""
    import base64

    attachment_dir = ATTACHMENTS_DIR / conversation_id
    attachment_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []
    for att in attachments:
        att_id = uuid.uuid4().hex[:16]
        # Determine extension from filename
        ext = Path(att.filename).suffix or ".bin"
        file_path = attachment_dir / f"{att_id}{ext}"

        # Decode and save
        file_data = base64.b64decode(att.base64_data)
        file_path.write_bytes(file_data)

        metadata_list.append({
            "attachment_id": att_id,
            "filename": att.filename,
            "mime_type": att.mime_type,
            "size_bytes": len(file_data),
        })
        logger.debug(f"Saved attachment {att.filename} as {file_path}")

    return metadata_list
```

Then modify lines 855-866 to use this function:

```python
# Add attachment metadata (filenames/types only, NOT base64 data)
if request.attachments:
    if user_metadata is None:
        user_metadata = {}
    user_metadata["attachments"] = _save_attachment_files(
        request.conversation_id, request.attachments
    )
```

**Step 2: Run existing tests to verify no breakage**

Run: `uv run pytest tests/ -v -x --ignore=tests/production`
Expected: All existing tests still pass

**Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: save chat attachments to filesystem with UUID identifiers"
```

---

### Task 2: Backend — GET endpoint for attachment retrieval

**Files:**
- Modify: `backend/main.py` (add new route)

**Step 1: Add attachment retrieval endpoint**

Add this route in `backend/main.py` (after the existing `/documents` routes, or near the chat endpoint):

```python
from fastapi.responses import FileResponse


@app.get("/api/attachments/{conversation_id}/{attachment_id}")
async def get_attachment(
    conversation_id: str,
    attachment_id: str,
    user: dict = Depends(get_current_user),
):
    """Retrieve a saved attachment file. Requires conversation ownership."""
    # Verify ownership
    owns = await postgres_adapter.verify_conversation_ownership(
        conversation_id, user["id"]
    )
    if not owns:
        raise HTTPException(status_code=403, detail="Access denied")

    # Find attachment file (attachment_id + any extension)
    att_dir = ATTACHMENTS_DIR / conversation_id
    if not att_dir.exists():
        raise HTTPException(status_code=404, detail="Attachment not found")

    # Glob for the file with any extension
    matches = list(att_dir.glob(f"{attachment_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Attachment not found")

    file_path = matches[0]

    # Determine content type from extension
    import mimetypes
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    return FileResponse(
        path=str(file_path),
        media_type=content_type,
        filename=file_path.name,
    )
```

Note: This uses `/api/` prefix so nginx auto-proxies it (see `location /api/` block in nginx config). No nginx changes needed.

**Step 2: Run tests**

Run: `uv run pytest tests/ -v -x --ignore=tests/production`
Expected: PASS

**Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: add GET /api/attachments endpoint for attachment retrieval"
```

---

### Task 3: Backend — Cleanup attachments on conversation delete

**Files:**
- Modify: `backend/routes/conversations.py:274-311` (delete_conversation endpoint)

**Step 1: Add attachment cleanup to delete_conversation**

In `backend/routes/conversations.py`, add import at top:

```python
import shutil
from pathlib import Path
```

Add constant:

```python
ATTACHMENTS_DIR = Path("data/attachments")
```

Modify `delete_conversation` (line ~294-302) to also cleanup attachments after successful DB deletion:

```python
    try:
        # Delete conversation (includes ownership check)
        deleted = await adapter.delete_conversation(conversation_id, user["id"])

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found or you do not have access to it"
            )

        # Cleanup attachment files
        att_dir = ATTACHMENTS_DIR / conversation_id
        if att_dir.exists():
            shutil.rmtree(att_dir, ignore_errors=True)
            logger.info(f"Cleaned up attachments for conversation {conversation_id}")

        return None  # 204 No Content
```

**Step 2: Run tests**

Run: `uv run pytest tests/ -v -x --ignore=tests/production`
Expected: PASS

**Step 3: Commit**

```bash
git add backend/routes/conversations.py
git commit -m "feat: cleanup attachment files on conversation delete"
```

---

### Task 4: Frontend — Update types and API service

**Files:**
- Modify: `frontend/src/types/index.ts:71-75`
- Modify: `frontend/src/services/api.ts:616-621`

**Step 1: Add `attachmentId` to MessageAttachmentMeta**

In `frontend/src/types/index.ts`, update the `MessageAttachmentMeta` interface (lines 71-75):

```typescript
export interface MessageAttachmentMeta {
  attachmentId?: string;  // UUID for backend retrieval (absent for pre-send)
  filename: string;
  mimeType: string;
  sizeBytes: number;
}
```

**Step 2: Update API service to map `attachment_id`**

In `frontend/src/services/api.ts`, update the attachment metadata mapping (lines 616-621):

```typescript
      if (msg.role === 'user' && msg.metadata?.attachments && Array.isArray(msg.metadata.attachments)) {
        attachments = msg.metadata.attachments.map((att: any) => ({
          attachmentId: att.attachment_id,
          filename: att.filename,
          mimeType: att.mime_type,
          sizeBytes: att.size_bytes,
        }));
      }
```

**Step 3: Add `getAttachmentUrl` helper to API service**

Add this method to the `ApiService` class in `frontend/src/services/api.ts`:

```typescript
  /**
   * Get URL for a stored attachment file.
   */
  getAttachmentUrl(conversationId: string, attachmentId: string): string {
    return `${API_BASE_URL}/api/attachments/${encodeURIComponent(conversationId)}/${encodeURIComponent(attachmentId)}`;
  }
```

**Step 4: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/services/api.ts
git commit -m "feat: add attachmentId to types and API service helper"
```

---

### Task 5: Frontend — Update useChat to preserve attachmentId in sent messages

**Files:**
- Modify: `frontend/src/hooks/useChat.ts:341-345`

**Step 1: Update userMessage creation to include attachmentId**

In `useChat.ts`, modify the attachment mapping in userMessage creation (lines 341-345). The `attachmentId` won't exist pre-send (we only have it after backend responds), but the backend saves it in metadata. When conversation history is loaded from API (Task 4), the `attachmentId` is already mapped. For the live session before refresh, we need the backend to return attachment IDs.

Actually — the simplest approach: the backend already saves the attachment and the metadata gets stored in the DB. When the page is refreshed and messages are loaded via `getConversationMessages`, the `attachment_id` is already in the metadata. For the current session (before refresh), we won't have `attachmentId` on the freshly-sent message — but we can work around this by using the base64 data that's still in memory during the current session.

No changes needed here. The `attachmentId` flows naturally through the API service when messages are loaded from history.

**Step 1 (revised): No code changes needed**

Skip this task — the data flow already handles this correctly:
- Pre-send: base64 available in React state
- Current session post-send: attachment metadata stored without attachmentId (preview via base64 if still in memory, or unavailable until refresh)
- After refresh: messages loaded from API include `attachment_id` from DB metadata

**Step 2: Commit** (skip — no changes)

---

### Task 6: Frontend — Create AttachmentPreviewModal component

**Files:**
- Create: `frontend/src/components/chat/AttachmentPreviewModal.tsx`

**Step 1: Create the modal component**

Create `frontend/src/components/chat/AttachmentPreviewModal.tsx`:

```tsx
import { useState, useEffect, useCallback } from 'react';
import { X, Download, ChevronLeft, ChevronRight, FileText, File } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useFadeIn } from '../../design-system/animations/hooks/useFadeIn';
import type { MessageAttachmentMeta } from '../../types';
import { apiService } from '../../services/api';

export interface PreviewAttachment {
  /** Attachment metadata */
  meta: MessageAttachmentMeta;
  /** Base64 data (available pre-send or from input bar) */
  base64Data?: string;
  /** Conversation ID (for backend URL) */
  conversationId?: string;
}

interface AttachmentPreviewModalProps {
  isOpen: boolean;
  attachments: PreviewAttachment[];
  initialIndex: number;
  onClose: () => void;
}

export function AttachmentPreviewModal({
  isOpen,
  attachments,
  initialIndex,
  onClose,
}: AttachmentPreviewModalProps) {
  const { t } = useTranslation();
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const { style: fadeStyle } = useFadeIn({ duration: 'fast' });

  // Reset index when modal opens with new data
  useEffect(() => {
    setCurrentIndex(initialIndex);
  }, [initialIndex, isOpen]);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') navigatePrev();
      if (e.key === 'ArrowRight') navigateNext();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, currentIndex, attachments.length]);

  const navigatePrev = useCallback(() => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : attachments.length - 1));
  }, [attachments.length]);

  const navigateNext = useCallback(() => {
    setCurrentIndex((prev) => (prev < attachments.length - 1 ? prev + 1 : 0));
  }, [attachments.length]);

  if (!isOpen || attachments.length === 0) return null;

  const current = attachments[currentIndex];
  if (!current) return null;

  const isImage = current.meta.mimeType.startsWith('image/');
  const isPdf = current.meta.mimeType === 'application/pdf';

  // Build URL for display
  const getUrl = (): string | null => {
    if (current.base64Data) {
      // Pre-send: convert base64 to object URL
      return `data:${current.meta.mimeType};base64,${current.base64Data}`;
    }
    if (current.meta.attachmentId && current.conversationId) {
      return apiService.getAttachmentUrl(current.conversationId, current.meta.attachmentId);
    }
    return null;
  };

  const url = getUrl();

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  const handleDownload = () => {
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = current.meta.filename;
    a.click();
  };

  return (
    <div
      className={cn(
        'fixed inset-0 z-50',
        'flex items-center justify-center',
        'bg-black/80 backdrop-blur-sm'
      )}
      style={fadeStyle}
      onClick={handleBackdropClick}
    >
      {/* Modal container */}
      <div className={cn(
        'relative flex flex-col',
        'max-w-[90vw] max-h-[90vh]',
        'bg-white dark:bg-accent-900',
        'rounded-2xl shadow-2xl',
        'overflow-hidden'
      )}>
        {/* Header */}
        <div className={cn(
          'flex items-center justify-between',
          'px-4 py-3',
          'border-b border-accent-200 dark:border-accent-800'
        )}>
          <div className="flex items-center gap-2 min-w-0">
            {isImage ? (
              <span className="text-blue-500"><FileText size={16} /></span>
            ) : (
              <span className="text-blue-500"><File size={16} /></span>
            )}
            <span className="text-sm font-medium truncate text-accent-900 dark:text-accent-100">
              {current.meta.filename}
            </span>
            <span className="text-xs text-accent-500 dark:text-accent-400 flex-shrink-0">
              ({(current.meta.sizeBytes / 1024).toFixed(0)} KB)
            </span>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            {url && (
              <button
                onClick={handleDownload}
                className={cn(
                  'p-2 rounded-lg',
                  'text-accent-500 hover:text-accent-700',
                  'dark:text-accent-400 dark:hover:text-accent-200',
                  'hover:bg-accent-100 dark:hover:bg-accent-800',
                  'transition-colors'
                )}
                title={t('attachments.download')}
              >
                <Download size={16} />
              </button>
            )}
            <button
              onClick={onClose}
              className={cn(
                'p-2 rounded-lg',
                'text-accent-500 hover:text-accent-700',
                'dark:text-accent-400 dark:hover:text-accent-200',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'transition-colors'
              )}
              title={t('attachments.close')}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className={cn(
          'flex-1 flex items-center justify-center',
          'p-4 min-h-[300px]',
          'overflow-auto'
        )}>
          {url ? (
            isImage ? (
              <img
                src={url}
                alt={current.meta.filename}
                className="max-w-full max-h-[75vh] object-contain rounded"
              />
            ) : isPdf ? (
              <iframe
                src={url}
                title={current.meta.filename}
                className="w-full h-[75vh] rounded border border-accent-200 dark:border-accent-700"
              />
            ) : (
              // Non-previewable file — show icon + download
              <div className="text-center space-y-4">
                <File size={48} className="mx-auto text-accent-400 dark:text-accent-500" />
                <p className="text-sm text-accent-600 dark:text-accent-400">
                  {current.meta.filename}
                </p>
                <button
                  onClick={handleDownload}
                  className={cn(
                    'px-4 py-2 rounded-lg text-sm font-medium',
                    'bg-accent-900 dark:bg-accent-100',
                    'text-white dark:text-accent-900',
                    'hover:bg-accent-800 dark:hover:bg-accent-200',
                    'transition-colors'
                  )}
                >
                  {t('attachments.download')}
                </button>
              </div>
            )
          ) : (
            // No URL available (pre-refresh, no attachmentId yet)
            <div className="text-center space-y-2">
              <File size={48} className="mx-auto text-accent-300 dark:text-accent-600" />
              <p className="text-sm text-accent-500 dark:text-accent-400">
                {current.meta.filename}
              </p>
              <p className="text-xs text-accent-400 dark:text-accent-500">
                {t('attachments.previewUnavailable')}
              </p>
            </div>
          )}
        </div>

        {/* Navigation arrows (if multiple attachments) */}
        {attachments.length > 1 && (
          <>
            <button
              onClick={navigatePrev}
              className={cn(
                'absolute left-2 top-1/2 -translate-y-1/2',
                'p-2 rounded-full',
                'bg-black/30 hover:bg-black/50',
                'text-white',
                'transition-colors'
              )}
            >
              <ChevronLeft size={20} />
            </button>
            <button
              onClick={navigateNext}
              className={cn(
                'absolute right-2 top-1/2 -translate-y-1/2',
                'p-2 rounded-full',
                'bg-black/30 hover:bg-black/50',
                'text-white',
                'transition-colors'
              )}
            >
              <ChevronRight size={20} />
            </button>
            {/* Page indicator */}
            <div className={cn(
              'absolute bottom-2 left-1/2 -translate-x-1/2',
              'px-3 py-1 rounded-full',
              'bg-black/40 text-white text-xs'
            )}>
              {currentIndex + 1} / {attachments.length}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
```

**Step 2: Verify it compiles**

Run: `cd /home/prusemic/SUJBOT/frontend && npx tsc --noEmit`
Expected: No errors related to AttachmentPreviewModal

**Step 3: Commit**

```bash
git add frontend/src/components/chat/AttachmentPreviewModal.tsx
git commit -m "feat: add AttachmentPreviewModal component"
```

---

### Task 7: Frontend — Make ChatMessage attachment chips clickable

**Files:**
- Modify: `frontend/src/components/chat/ChatMessage.tsx:149-178`

**Step 1: Add state and modal to ChatMessage**

In `ChatMessage.tsx`, the component needs to accept `conversationId` as a prop and manage preview modal state. Modify the component:

Add import:
```tsx
import { AttachmentPreviewModal, type PreviewAttachment } from './AttachmentPreviewModal';
```

Add `conversationId` to `ChatMessageProps`:
```typescript
interface ChatMessageProps {
  message: Message;
  conversationId?: string;  // For attachment preview URLs
  animationDelay?: number;
  onEdit: (messageId: string, newContent: string) => void;
  onRegenerate: (messageId: string) => void;
  disabled?: boolean;
  responseDurationMs?: number;
}
```

Add state inside the component:
```typescript
const [previewIndex, setPreviewIndex] = useState<number | null>(null);
```

Modify the attachment chips (lines 149-178) to be clickable:

```tsx
{isUser && message.attachments && message.attachments.length > 0 && (
  <div className={cn('flex flex-wrap gap-1.5', 'justify-end')}>
    {message.attachments.map((att, idx) => (
      <button
        key={idx}
        type="button"
        onClick={() => setPreviewIndex(idx)}
        className={cn(
          'inline-flex items-center gap-1.5 px-2.5 py-1',
          'bg-blue-50 dark:bg-blue-900/30',
          'text-blue-700 dark:text-blue-300',
          'rounded-lg',
          'border border-blue-200 dark:border-blue-800',
          'text-xs',
          'cursor-pointer hover:bg-blue-100 dark:hover:bg-blue-900/50',
          'transition-colors'
        )}
      >
        {att.mimeType.startsWith('image/') ? (
          <Image size={12} />
        ) : att.mimeType === 'application/pdf' ? (
          <FileText size={12} />
        ) : (
          <File size={12} />
        )}
        <span className="truncate max-w-[150px]" title={att.filename}>
          {att.filename}
        </span>
        <span className="text-blue-400 dark:text-blue-500">
          ({(att.sizeBytes / 1024).toFixed(0)} KB)
        </span>
      </button>
    ))}
  </div>
)}
```

Add the modal at the end of the component (before the final closing `</div>`s):

```tsx
{/* Attachment preview modal */}
{previewIndex !== null && message.attachments && (
  <AttachmentPreviewModal
    isOpen={previewIndex !== null}
    attachments={message.attachments.map(att => ({
      meta: att,
      conversationId,
    }))}
    initialIndex={previewIndex}
    onClose={() => setPreviewIndex(null)}
  />
)}
```

**Step 2: Update ChatMessage usage in ChatContainer to pass conversationId**

Check where `ChatMessage` is rendered and pass `conversationId`. In `ChatContainer.tsx`, add the prop:

```tsx
<ChatMessage
  ...existing props...
  conversationId={currentConversation?.id}
/>
```

**Step 3: Verify compilation**

Run: `cd /home/prusemic/SUJBOT/frontend && npx tsc --noEmit`
Expected: No errors

**Step 4: Commit**

```bash
git add frontend/src/components/chat/ChatMessage.tsx frontend/src/components/chat/ChatContainer.tsx
git commit -m "feat: make message attachment chips clickable with preview modal"
```

---

### Task 8: Frontend — Make ChatInput attachment chips clickable

**Files:**
- Modify: `frontend/src/components/chat/ChatInput.tsx:214-257`

**Step 1: Add preview state and modal to ChatInput**

Add import:
```tsx
import { AttachmentPreviewModal, type PreviewAttachment } from './AttachmentPreviewModal';
```

Add state:
```typescript
const [previewIndex, setPreviewIndex] = useState<number | null>(null);
```

Modify attachment chips (lines 217-255) — wrap the chip `div` in a clickable button (keeping the X remove button separate):

```tsx
{attachments.map(att => (
  <div
    key={att.id}
    className={cn(
      'inline-flex items-center gap-1.5 px-2.5 py-1',
      'bg-blue-50 dark:bg-blue-900/30',
      'text-blue-700 dark:text-blue-300',
      'rounded-lg',
      'border border-blue-200 dark:border-blue-800',
      'text-xs'
    )}
  >
    <button
      type="button"
      onClick={() => setPreviewIndex(attachments.indexOf(att))}
      className={cn(
        'inline-flex items-center gap-1.5',
        'cursor-pointer hover:text-blue-900 dark:hover:text-blue-100',
        'transition-colors'
      )}
    >
      {att.mimeType.startsWith('image/') ? (
        <Image size={12} />
      ) : (
        <File size={12} />
      )}
      <span className="truncate max-w-[150px]" title={att.filename}>
        {att.filename}
      </span>
      <span className="text-blue-400 dark:text-blue-500">
        ({(att.sizeBytes / 1024).toFixed(0)} KB)
      </span>
    </button>
    <button
      type="button"
      onClick={() => removeAttachment(att.id)}
      className={cn(
        'p-0.5 rounded-full -mr-0.5',
        'text-blue-400 dark:text-blue-500',
        'hover:bg-blue-200 dark:hover:bg-blue-800',
        'hover:text-blue-600 dark:hover:text-blue-300',
        'transition-colors duration-150'
      )}
      title={t('attachments.remove')}
    >
      <X size={10} />
    </button>
  </div>
))}
```

Add the modal at the end (before the closing `</form>`):

```tsx
{/* Attachment preview modal (pre-send) */}
{previewIndex !== null && (
  <AttachmentPreviewModal
    isOpen={previewIndex !== null}
    attachments={attachments.map(att => ({
      meta: {
        filename: att.filename,
        mimeType: att.mimeType,
        sizeBytes: att.sizeBytes,
      },
      base64Data: att.base64Data,
    }))}
    initialIndex={previewIndex}
    onClose={() => setPreviewIndex(null)}
  />
)}
```

**Step 2: Verify compilation**

Run: `cd /home/prusemic/SUJBOT/frontend && npx tsc --noEmit`
Expected: No errors

**Step 3: Commit**

```bash
git add frontend/src/components/chat/ChatInput.tsx
git commit -m "feat: make input bar attachment chips clickable with preview modal"
```

---

### Task 9: Frontend — Add i18n keys

**Files:**
- Modify: `frontend/src/i18n/locales/en.json`
- Modify: `frontend/src/i18n/locales/cs.json`

**Step 1: Add English keys**

In `en.json`, add to the `attachments` section:

```json
"download": "Download",
"close": "Close",
"previewUnavailable": "Preview available after page refresh"
```

**Step 2: Add Czech keys**

In `cs.json`, add to the `attachments` section:

```json
"download": "Stáhnout",
"close": "Zavřít",
"previewUnavailable": "Náhled bude dostupný po obnovení stránky"
```

**Step 3: Commit**

```bash
git add frontend/src/i18n/locales/en.json frontend/src/i18n/locales/cs.json
git commit -m "feat: add i18n keys for attachment preview modal"
```

---

### Task 10: Manual smoke test

**Step 1: Start the dev stack**

Run: `docker compose up -d`

**Step 2: Test pre-send preview**

1. Log in to the web UI
2. Attach an image file
3. Click the attachment chip in the input bar
4. Verify the preview modal opens with the image displayed
5. Close the modal (Escape, click outside, or X button)

**Step 3: Test post-send preview**

1. Send a message with an image attachment
2. Refresh the page (so attachment loads from API with `attachment_id`)
3. Click the attachment chip in the sent message
4. Verify the preview modal opens with the image fetched from backend

**Step 4: Test multiple attachments**

1. Attach 2-3 files (mix of images and PDFs)
2. Click on one — verify arrow navigation between attachments
3. Verify keyboard navigation (left/right arrows, Escape)

**Step 5: Test conversation delete cleanup**

1. Send a message with attachments
2. Delete the conversation
3. Verify `data/attachments/{conversation_id}/` directory is removed
