# Parallel Chat Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to generate responses in multiple conversations simultaneously by making streaming state per-conversation instead of global.

**Architecture:** Replace the single global `isStreaming` boolean + global refs with a `Set<string>` of streaming conversation IDs (React state for UI) and a `Map<string, StreamingState>` ref (mutable data for each active stream). The return API stays identical — `isStreaming` becomes a derived boolean for the current conversation.

**Tech Stack:** React (hooks, refs, state), TypeScript, Vite dev server for verification

**No frontend test framework exists.** Verification is via Vite compilation (type errors = build failure) and manual browser testing.

---

### Task 1: Add StreamingState type and new data structures

**Files:**
- Modify: `frontend/src/hooks/useChat.ts:35-48` (state declarations)

**Step 1: Add StreamingState interface and replace state/refs**

At the top of `useChat.ts` (after the `SpendingLimitError` interface), add:

```typescript
/** Per-conversation streaming state (held in a ref Map, not React state) */
interface StreamingState {
  currentMessage: Message | null;
  currentToolCalls: Map<string, ToolCall>;
  abortController: AbortController;
}
```

Inside `useChat()`, replace:

```typescript
// OLD — remove these:
const [isStreaming, setIsStreaming] = useState(false);
const currentMessageRef = useRef<Message | null>(null);
const currentToolCallsRef = useRef<Map<string, ToolCall>>(new Map());
const abortControllerRef = useRef<AbortController | null>(null);
```

With:

```typescript
// NEW — per-conversation streaming
const [streamingConversationIds, setStreamingConversationIds] = useState<Set<string>>(new Set());
const streamingRefsMap = useRef<Map<string, StreamingState>>(new Map());

// Derived: is the CURRENT conversation streaming? (backwards-compatible API)
const isStreaming = currentConversationId
  ? streamingConversationIds.has(currentConversationId)
  : false;
```

**Step 2: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -30`

Expected: Type errors in `sendMessage`, `cancelStreaming`, etc. (they still reference old refs). This is expected — we fix them in subsequent tasks.

---

### Task 2: Refactor sendMessage to use per-conversation streaming

**Files:**
- Modify: `frontend/src/hooks/useChat.ts` (sendMessage function, ~lines 293-822)

**Step 1: Update sendMessage guard and setup**

Replace the guard at the top of `sendMessage`:

```typescript
// OLD:
if (isStreaming || (!content.trim() && (!attachments || attachments.length === 0))) {
  return;
}
```

With:

```typescript
// NEW: Block duplicate sends to SAME conversation, allow others
if (!content.trim() && (!attachments || attachments.length === 0)) {
  return;
}
```

After determining `conversation` (around line 320), add a per-conversation guard:

```typescript
const convId = conversation.id;

// Block if THIS conversation is already streaming
if (streamingConversationIds.has(convId)) {
  return;
}
```

Replace the streaming state initialization (around lines 395-402):

```typescript
// OLD:
setIsStreaming(true);
if (abortControllerRef.current) {
  abortControllerRef.current.abort();
}
abortControllerRef.current = new AbortController();
```

With:

```typescript
// NEW: Per-conversation streaming state
const abortController = new AbortController();
const streamState: StreamingState = {
  currentMessage: currentMessage,  // the message we just created above
  currentToolCalls: new Map(),
  abortController,
};
streamingRefsMap.current.set(convId, streamState);
setStreamingConversationIds(prev => new Set(prev).add(convId));
```

Where `currentMessage` is the assistant message object that was previously assigned to `currentMessageRef.current` (around line 367). Rename the local variable from the ref assignment to a `const`:

```typescript
// OLD:
currentMessageRef.current = { id: ..., role: 'assistant', ... };
currentToolCallsRef.current = new Map();
```

```typescript
// NEW:
const currentMessage: Message = { id: ..., role: 'assistant', ... };
// (currentToolCalls is inside streamState)
```

**Step 2: Update all SSE event handlers to use streamState**

Throughout the `for await` loop, replace all occurrences of:
- `currentMessageRef.current` → `streamState.currentMessage`
- `currentToolCallsRef.current` → `streamState.currentToolCalls`

This is a mechanical find-and-replace within the `sendMessage` function body.

Also replace the abort signal reference:

```typescript
// OLD:
abortControllerRef.current.signal
```

```typescript
// NEW:
abortController.signal
```

**Step 3: Update the finally block**

Replace:

```typescript
// OLD:
finally {
  setIsStreaming(false);
  currentMessageRef.current = null;
  currentToolCallsRef.current = new Map();
  abortControllerRef.current = null;
  apiService.invalidateSpendingCache();
  setSpendingRefreshTrigger((prev) => prev + 1);
}
```

With:

```typescript
// NEW:
finally {
  streamingRefsMap.current.delete(convId);
  setStreamingConversationIds(prev => {
    const next = new Set(prev);
    next.delete(convId);
    return next;
  });
  apiService.invalidateSpendingCache();
  setSpendingRefreshTrigger((prev) => prev + 1);
}
```

**Step 4: Update sendMessage dependency array**

Remove `isStreaming` from the dependency array (it's no longer used as a guard):

```typescript
// OLD:
[isStreaming, createConversation, currentConversationId, conversations]
// NEW:
[streamingConversationIds, createConversation, currentConversationId, conversations]
```

**Step 5: Verify compilation**

Run: `cd frontend && npx tsc --noEmit 2>&1 | head -30`

Expected: Remaining errors in `cancelStreaming`, `editMessage`, `regenerateMessage`, `submitClarification`, `beforeunload`.

---

### Task 3: Refactor cancelStreaming

**Files:**
- Modify: `frontend/src/hooks/useChat.ts` (cancelStreaming function, ~lines 1212-1256)

**Step 1: Rewrite cancelStreaming**

Replace the entire `cancelStreaming` function:

```typescript
const cancelStreaming = useCallback(() => {
  const convId = currentConversationId;
  if (!convId) return;

  const streamState = streamingRefsMap.current.get(convId);
  if (!streamState) {
    console.log('🛑 useChat: cancelStreaming called but no active stream for', convId);
    return;
  }

  console.log('🛑 useChat: User cancelled streaming for', convId);
  streamState.abortController.abort();

  // Clear agent progress in the message state
  if (streamState.currentMessage) {
    const messageId = streamState.currentMessage.id;
    setConversations(prev =>
      prev.map(conv => {
        if (conv.id !== convId) return conv;
        return {
          ...conv,
          messages: conv.messages.map(msg =>
            msg.id === messageId
              ? {
                  ...msg,
                  agentProgress: msg.agentProgress
                    ? { ...msg.agentProgress, isStreaming: false, currentAgent: null }
                    : undefined,
                }
              : msg
          ),
        };
      })
    );
  }

  // Cleanup
  streamingRefsMap.current.delete(convId);
  setStreamingConversationIds(prev => {
    const next = new Set(prev);
    next.delete(convId);
    return next;
  });
}, [currentConversationId]);
```

---

### Task 4: Refactor editMessage and regenerateMessage guards

**Files:**
- Modify: `frontend/src/hooks/useChat.ts` (editMessage ~line 827, regenerateMessage ~line 893)

**Step 1: Update editMessage guard**

Replace:

```typescript
if (isStreaming || !newContent.trim()) return;
```

With:

```typescript
if (!newContent.trim()) return;
// Block if current conversation is already streaming
if (currentConversationId && streamingConversationIds.has(currentConversationId)) return;
```

Update dependency array: replace `isStreaming` with `streamingConversationIds`.

**Step 2: Update regenerateMessage guard**

Replace:

```typescript
if (isStreaming) {
  return;
}
```

With:

```typescript
if (currentConversationId && streamingConversationIds.has(currentConversationId)) {
  return;
}
```

Update dependency array: replace `isStreaming` with `streamingConversationIds`.

---

### Task 5: Refactor cleanup handlers (beforeunload, deleteConversation, submitClarification)

**Files:**
- Modify: `frontend/src/hooks/useChat.ts`

**Step 1: Update beforeunload handler**

Replace:

```typescript
useEffect(() => {
  const handleBeforeUnload = (e: BeforeUnloadEvent) => {
    if (abortControllerRef.current) {
      e.preventDefault();
      e.returnValue = '';
      console.log('🔄 useChat: Aborting stream due to page unload');
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  window.addEventListener('beforeunload', handleBeforeUnload);
  return () => {
    window.removeEventListener('beforeunload', handleBeforeUnload);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };
}, []);
```

With:

```typescript
useEffect(() => {
  const handleBeforeUnload = (e: BeforeUnloadEvent) => {
    if (streamingRefsMap.current.size > 0) {
      e.preventDefault();
      e.returnValue = '';
      console.log('🔄 useChat: Aborting all streams due to page unload');
      for (const [, state] of streamingRefsMap.current) {
        state.abortController.abort();
      }
      streamingRefsMap.current.clear();
    }
  };

  window.addEventListener('beforeunload', handleBeforeUnload);
  return () => {
    window.removeEventListener('beforeunload', handleBeforeUnload);
    for (const [, state] of streamingRefsMap.current) {
      state.abortController.abort();
    }
    streamingRefsMap.current.clear();
  };
}, []);
```

**Step 2: Update deleteConversation**

In `deleteConversation`, add stream cleanup before removing from state:

```typescript
const deleteConversation = useCallback(async (id: string) => {
  // Abort stream if this conversation is streaming
  const streamState = streamingRefsMap.current.get(id);
  if (streamState) {
    streamState.abortController.abort();
    streamingRefsMap.current.delete(id);
    setStreamingConversationIds(prev => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  try {
    // ... rest unchanged
```

**Step 3: Update submitClarification**

In `submitClarification`, replace streaming state management:

Replace `setIsStreaming(true)` and refs setup with per-conversation equivalents (same pattern as sendMessage). Replace the finally block similarly.

```typescript
// Setup (around line 1004-1006):
const convId = conversation.id;
const abortController = new AbortController();
const streamState: StreamingState = {
  currentMessage: currentMessage,  // the message created above
  currentToolCalls: new Map(),
  abortController,
};
streamingRefsMap.current.set(convId, streamState);
setStreamingConversationIds(prev => new Set(prev).add(convId));
```

Replace `currentMessageRef.current` → `streamState.currentMessage` in the event handlers.

```typescript
// Finally block:
finally {
  streamingRefsMap.current.delete(convId);
  setStreamingConversationIds(prev => {
    const next = new Set(prev);
    next.delete(convId);
    return next;
  });
}
```

**Step 4: Update cancelClarification**

Replace `setIsStreaming(false)` with no-op (the stream cleanup happens via abort).

Actually, `cancelClarification` just resets clarification state — it should also abort the stream:

```typescript
const cancelClarification = useCallback(() => {
  console.log('🚫 FRONTEND: Clarification cancelled');
  setClarificationData(null);
  setAwaitingClarification(false);
  // Note: isStreaming for current conversation is derived, no explicit reset needed
  // The stream will complete or be aborted independently
}, []);
```

---

### Task 6: Verify compilation and manual testing

**Step 1: Verify TypeScript compilation**

Run: `cd frontend && npx tsc --noEmit`

Expected: No errors.

**Step 2: Start dev server**

Run: `cd frontend && npm run dev`

**Step 3: Manual test cases**

1. **Basic send/receive**: Send a message, verify streaming works normally
2. **Stop button**: While streaming, verify stop button appears only in the streaming conversation
3. **Switch during generation**: Start a message, switch to another conversation — verify:
   - The other conversation shows send button (not stop)
   - You can type and send a message there
   - Switching back shows live streaming text
4. **Parallel streams**: Start generation in conv A, switch to conv B, send a message in B — verify both stream independently
5. **Cancel one stream**: While both are streaming, cancel one — verify the other continues
6. **Delete streaming conversation**: Delete a conversation that's currently streaming — verify no errors

**Step 4: Commit**

```bash
git add frontend/src/hooks/useChat.ts
git commit -m "feat: per-conversation streaming state for parallel generation

Replace global isStreaming boolean with per-conversation Set<string>.
Each conversation tracks its own AbortController, currentMessage,
and toolCalls independently. Users can now generate responses in
multiple conversations simultaneously."
```
