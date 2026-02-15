# Parallel Chat Streaming

**Date:** 2026-02-15
**Status:** Approved

## Problem

The frontend uses a single global `isStreaming` boolean. When any conversation is generating, ALL conversations show the stop button and block input. Users cannot:
- Switch to another conversation and send a message while one is generating
- Run multiple conversations in parallel

The backend already supports concurrent requests (each request is fully isolated).

## Solution: Per-Conversation Streaming Map

Replace global streaming state with per-conversation tracking.

### Data Structures

```typescript
// React state — triggers re-renders for UI updates
const [streamingConversationIds, setStreamingConversationIds] = useState<Set<string>>(new Set());

// Mutable ref — no re-renders, holds streaming internals
const streamingRefsMap = useRef<Map<string, StreamingState>>(new Map());

interface StreamingState {
  currentMessage: Message | null;
  currentToolCalls: Map<string, ToolCall>;
  abortController: AbortController;
}
```

### Derived Values

```typescript
// Per-conversation streaming check (used by components)
const isCurrentConversationStreaming = currentConversationId
  ? streamingConversationIds.has(currentConversationId)
  : false;
```

### Changes by File

**`useChat.ts`** (main refactor):
1. `isStreaming` boolean → `streamingConversationIds` Set state
2. Global refs → `streamingRefsMap` ref (`Map<string, StreamingState>`)
3. `sendMessage`: guard per-conversation (not globally)
4. `cancelStreaming`: accept optional `conversationId`, cancel specific stream
5. `editMessage` / `regenerateMessage`: guard per-conversation
6. `beforeunload`: abort ALL active controllers
7. `deleteConversation`: abort stream if conversation was streaming
8. Return `isStreaming` as derived value for current conversation (API unchanged)
9. All SSE event handlers: read/write from `streamingRefsMap.get(conversationId)` instead of global refs

**`ChatInput.tsx`** — no changes (receives `isStreaming` per-conversation from parent)

**`ChatContainer.tsx`** — no changes (receives `isStreaming` per-conversation from parent)

**`App.tsx`** — no changes (receives `isStreaming` from hook, now derived per-conversation)

**Backend** — zero changes

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| Switch to streaming conversation | Live streaming text visible (state updates target by conversation ID) |
| Switch to idle conversation | Send button shown, can type and send |
| Delete streaming conversation | Abort its stream, remove from state |
| Page refresh during multiple streams | `beforeunload` aborts all, backend saves partial responses |
| Same conversation double-send | Blocked (per-conversation guard) |
| Clarification during parallel streams | Works on current conversation only |

### Key Constraint

The `sendMessage` function captures `conversationId` at call time and uses it throughout the SSE loop. This naturally isolates concurrent streams — each closure operates on its own conversation ID.
