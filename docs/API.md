# Backend API Reference

The backend is a FastAPI application (`backend/main.py`) serving at port 8000. All endpoints except `/health` and `/auth/login` require JWT authentication via httpOnly cookies.

## Authentication

### `POST /auth/login`

Login with email and password. Returns JWT token as httpOnly cookie.

**Request body:**
```json
{ "email": "user@example.com", "password": "password123" }
```

**Response:** `200 OK` with `Set-Cookie: access_token=<jwt>`

### `POST /auth/logout`

Clear the JWT cookie.

**Response:** `200 OK`

### `GET /auth/me`

Get current authenticated user profile.

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_admin": false,
  "created_at": "2025-01-01T00:00:00",
  "last_login_at": "2025-01-15T10:30:00"
}
```

### `POST /auth/change-password`

Change the current user's password.

**Request body:**
```json
{ "current_password": "old", "new_password": "new" }
```

### `POST /auth/register`

Create a new user. Admin only.

**Request body:**
```json
{ "email": "new@example.com", "password": "StrongPass123!" }
```

## Chat

### `POST /chat/stream`

Main chat endpoint. Streams response via Server-Sent Events (SSE).

**Request body:**
```json
{
  "message": "What does regulation X require?",
  "conversation_id": "abc123"  // optional, creates new if omitted
}
```

**SSE event types:**

| Event | Data | Description |
|-------|------|-------------|
| `tool_health` | `{"tools": [...], "count": 8}` | Available tools (first event in every stream) |
| `title_update` | `{"title": "Generated Title"}` | Auto-generated conversation title (first message only) |
| `text_delta` | `{"content": "partial text..."}` | Streaming text chunks |
| `tool_call` | `{"tool_name": "search", "tool_input": {...}, "call_id": "..."}` | Sent before tool execution |
| `tool_calls_summary` | `{"tool_calls": [...], "count": 2}` | Summary after all tools complete |
| `cost_summary` | `{"total_cost": 0.105, "total_input_tokens": ..., "total_output_tokens": ...}` | Cost breakdown in USD |
| `message_saved` | `{"message_id": 123}` | Confirmation of DB save |
| `done` | `{}` | Stream complete |
| `error` | `{"error": "message", "type": "ErrorType"}` | Error occurred |

### `POST /chat/clarify`

Resume an interrupted workflow with user clarification (after a `clarification_needed` event). Currently disabled (`config.json` → `multi_agent.clarification.enabled: false`).

**Request body:**
```json
{
  "thread_id": "abc123",
  "response": "I need information about GDPR Article 17 specifically."
}
```

**SSE event types:** Same as `/chat/stream` (text_delta, cost_summary, done, error).

## Conversations

### `GET /conversations`

List all conversations for the authenticated user (limit: 50).

**Response:**
```json
[
  {
    "id": "abc123",
    "title": "Nuclear safety regulations",
    "created_at": "2025-01-01T00:00:00",
    "updated_at": "2025-01-01T12:00:00"
  }
]
```

### `POST /conversations`

Create a new conversation.

**Response:**
```json
{ "id": "new-conversation-id" }
```

### `GET /conversations/{conversation_id}/messages`

Get message history for a conversation (limit: 100).

### `POST /conversations/{conversation_id}/messages`

Append a message to a conversation.

### `DELETE /conversations/{conversation_id}`

Delete a conversation and all its messages.

### `PATCH /conversations/{conversation_id}/title`

Update conversation title.

**Request body:**
```json
{ "title": "New Title" }
```

### `DELETE /conversations/{conversation_id}/messages/after/{keep_count}`

Truncate messages after `keep_count`. Used for edit/regenerate functionality.

## Documents

### `GET /documents/`

List all available PDF documents with metadata.

### `GET /documents/{document_id}/pdf`

Serve a PDF file (inline Content-Disposition for in-browser viewing).

### `POST /documents/upload`

Upload a PDF and index it via the VL pipeline. Streams progress via SSE.

**Request:** `multipart/form-data` with PDF file

**SSE event types:**

| Event | Data | Description |
|-------|------|-------------|
| `progress` | `{"stage": "rendering", "percent": 50, "message": "..."}` | Pipeline progress |
| `complete` | `{"document_id": "DOC_1", "pages": 105, "display_name": "Doc Title"}` | Indexing finished |
| `warning` | `{"stage": "graph_extraction", "message": "..."}` | Non-fatal issue |
| `error` | `{"message": "Error text", "stage": "indexing"}` | Fatal error |

**Progress stages:** `uploading` → `rendering` → `embedding` → `summarizing` → `storing` → `graph_extraction`

## Citations

### `GET /citations/{chunk_id}`

Resolve a chunk/page ID to citation metadata (document title, page number, section).

### `POST /citations/batch`

Batch citation lookup (max 50 chunk IDs).

**Request body:**
```json
{ "chunk_ids": ["doc_L3_c1_sec_1", "doc_vl_page_5"] }
```

## Settings

### `GET /settings/agent-variant`

Get the current user's agent variant preference (`remote` or `local`).

### `POST /settings/agent-variant`

Update agent variant preference.

**Request body:**
```json
{ "variant": "local" }
```

### `GET /settings/spending`

Get user spending information (total cost, limit, remaining).

## Feedback

### `POST /feedback`

Submit feedback for an assistant message (thumbs up/down with optional comment).

**Request body:**
```json
{
  "message_id": 123,
  "score": 1,
  "comment": "Helpful answer",
  "run_id": "optional-langsmith-run-id"
}
```

- `score`: `1` (positive) or `-1` (negative)
- `run_id`: Optional LangSmith trace ID for linking feedback to a specific run

### `GET /feedback/{message_id}`

Check if the current user has already submitted feedback for a message.

## Admin

All admin endpoints require admin privileges (`is_admin: true`).

### `POST /admin/login`

Admin-specific login (verifies admin flag).

### Users

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/users` | List all users (paginated, limit: 50) |
| `GET` | `/admin/users/{id}` | Get user details |
| `POST` | `/admin/users` | Create new user |
| `PUT` | `/admin/users/{id}` | Update user |
| `DELETE` | `/admin/users/{id}` | Delete user |
| `POST` | `/admin/users/{id}/spending/reset` | Reset spending counter |

### Admin Conversation Viewing

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/users/{user_id}/conversations` | List user conversations (read-only) |
| `GET` | `/admin/users/{user_id}/conversations/{conversation_id}/messages` | View conversation messages |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/health` | Detailed health check (PostgreSQL, backend) |
| `GET` | `/admin/stats` | System statistics (users, conversations, documents) |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/documents` | List all documents with metadata |
| `DELETE` | `/admin/documents/{document_id}` | Delete document (PDF, vectors, page images, graph) |
| `POST` | `/admin/documents/{document_id}/reindex` | Reindex document (SSE stream) |

## Error Codes

| Code | Meaning | Common Cause |
|------|---------|-------------|
| 401 | Unauthorized | Missing or expired JWT cookie |
| 402 | Payment Required | User spending limit exceeded |
| 403 | Forbidden | Non-admin accessing admin endpoints |
| 404 | Not Found | Invalid conversation/document/user ID |
| 422 | Validation Error | Invalid request body (FastAPI auto-validation) |
| 500 | Internal Server Error | Unhandled exception in backend |

## Security

- **Authentication**: JWT tokens in httpOnly cookies (immune to XSS)
- **Password hashing**: Argon2 (OWASP-recommended)
- **Rate limiting**: Token bucket algorithm per IP
- **Headers**: CSP, HSTS, X-Frame-Options, X-Content-Type-Options
- **CORS**: Explicit allow-list configuration
- **SQL protection**: Parameterized queries via asyncpg
- **Conversation isolation**: Users can only access their own conversations
