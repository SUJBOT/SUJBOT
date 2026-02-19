/**
 * useChat hook - Manages chat state and SSE streaming
 *
 * Supports per-conversation parallel streaming: multiple conversations
 * can stream responses simultaneously, each with isolated state.
 */

import { useState, useCallback, useRef, useEffect, type SetStateAction, type Dispatch } from 'react';
import { apiService } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import type { Message, Conversation, ToolCall, ClarificationData, Attachment } from '../types';

/**
 * Helper to update a single conversation in the conversations array by ID.
 * Avoids repeating `prev.map(c => c.id === id ? { ...c, ...updates } : c)` everywhere.
 */
function updateConversationById(
  setter: Dispatch<SetStateAction<Conversation[]>>,
  id: string,
  updates: Partial<Conversation>,
) {
  setter(prev => prev.map(c => c.id === id ? { ...c, ...updates } : c));
}

// UUID validation regex for conversation IDs
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Validate and extract conversation ID from URL parameter
 */
function getValidConversationIdFromUrl(): string | null {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  const urlConversationId = params.get('c');
  // Only accept valid UUID format to prevent invalid state
  if (urlConversationId && UUID_REGEX.test(urlConversationId)) {
    return urlConversationId;
  }
  return null;
}

// Spending limit error data from 402 response
export interface SpendingLimitError {
  message_cs: string;
  message_en: string;
  total_spent_czk: number;
  spending_limit_czk: number;
}

// Per-conversation streaming state (mutable, stored in ref map)
interface StreamingState {
  currentMessage: Message | null;
  currentToolCalls: Map<string, ToolCall>;
  abortController: AbortController;
}

export function useChat() {
  const { isAuthenticated } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  // Current conversation ID - initialized from URL query param for refresh persistence
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(
    getValidConversationIdFromUrl
  );
  // Set of conversation IDs currently streaming (React state for UI re-renders)
  const [streamingConversationIds, setStreamingConversationIds] = useState<Set<string>>(new Set());
  const [clarificationData, setClarificationData] = useState<ClarificationData | null>(null);
  const [awaitingClarification, setAwaitingClarification] = useState(false);
  // Spending tracking
  const [spendingLimitError, setSpendingLimitError] = useState<SpendingLimitError | null>(null);
  const [spendingRefreshTrigger, setSpendingRefreshTrigger] = useState(0);

  // Derived: is the CURRENT conversation streaming? (backward-compatible API)
  const isStreaming = currentConversationId
    ? streamingConversationIds.has(currentConversationId)
    : false;

  // Sync URL with current conversation ID (for refresh persistence and shareable links)
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const url = new URL(window.location.href);
    if (currentConversationId) {
      url.searchParams.set('c', currentConversationId);
    } else {
      url.searchParams.delete('c');
    }

    // Update URL without page reload (replaceState to avoid history spam)
    window.history.replaceState({}, '', url.toString());
  }, [currentConversationId]);

  // Handle browser back/forward navigation
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handlePopState = () => {
      // Use same validation as initial load
      setCurrentConversationId(getValidConversationIdFromUrl());
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Load conversations from server when user is authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      // Clear conversations on logout
      setConversations([]);
      setCurrentConversationId(null);
      return;
    }

    const loadConversations = async () => {
      try {
        const serverConversations = await apiService.getConversations();
        setConversations(serverConversations);
      } catch (error) {
        console.error('Failed to load conversations from server:', error);
        // Don't block UI - user can still create new conversations
      }
    };

    loadConversations();
  }, [isAuthenticated]); // Re-run when auth state changes (login/logout)

  // Validate that current conversation still exists after loading conversations
  useEffect(() => {
    if (currentConversationId && conversations.length > 0) {
      const conversationExists = conversations.some(c => c.id === currentConversationId);
      if (!conversationExists) {
        // Current conversation was deleted, clear it
        console.log('Current conversation no longer exists, clearing currentConversationId');
        setCurrentConversationId(null);
      }
    }
  }, [conversations, currentConversationId]);

  // Per-conversation streaming state map (mutable ref, no re-renders)
  const streamingRefsMap = useRef<Map<string, StreamingState>>(new Map());

  // Warn user and cancel streaming on page unload (refresh, close, navigation)
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      // Only show warning if any conversation is actively streaming
      if (streamingRefsMap.current.size > 0) {
        // Standard way to show browser's "Leave page?" dialog
        e.preventDefault();
        // For older browsers (Chrome < 119)
        e.returnValue = '';

        console.log('🔄 useChat: Aborting all streams due to page unload');
        for (const [id, state] of streamingRefsMap.current) {
          console.log(`  Aborting stream for conversation ${id}`);
          state.abortController.abort();
        }
        streamingRefsMap.current.clear();
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      // Also cleanup on hook unmount
      for (const [, state] of streamingRefsMap.current) {
        state.abortController.abort();
      }
      streamingRefsMap.current.clear();
    };
  }, []);

  // Ref pattern to avoid stale closures in async callbacks
  //
  // Problem: In regenerateMessage and editMessage, we read conversation state from an async callback
  // that may execute after state updates. Using `conversations` directly would capture
  // the value from when the callback was created (stale closure).
  //
  // Solution: Keep a ref that always points to latest conversations array. The ref
  // object never changes identity, so it's safe to read in async contexts.
  //
  // When to use: Async callbacks that need latest state (regenerate, edit)
  // When NOT to use: Synchronous operations - use functional setState instead
  const conversationsRef = useRef<Conversation[]>(conversations);

  // Keep ref synchronized with state
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  /**
   * Clean invalid/incomplete messages from conversation
   * Prevents corrupted data in database
   */
  const cleanMessages = useCallback((messages: Message[]): Message[] => {
    return messages.filter(msg =>
      msg &&
      msg.role &&
      msg.id &&
      msg.timestamp &&
      msg.content !== undefined
    );
  }, []);

  /**
   * Get current conversation
   */
  const currentConversation = conversations.find((c) => c.id === currentConversationId);

  /**
   * Create a new conversation
   * Uses pushState for browser history (back/forward navigation)
   */
  const createConversation = useCallback(async () => {
    try {
      // Create conversation on server
      const newConversation = await apiService.createConversation('New Conversation');

      // Update local state
      setConversations((prev) => [...prev, newConversation]);
      setCurrentConversationId(newConversation.id);

      // Add to browser history for back/forward navigation
      const url = new URL(window.location.href);
      url.searchParams.set('c', newConversation.id);
      window.history.pushState({}, '', url.toString());

      return newConversation;
    } catch (error) {
      console.error('Failed to create conversation:', error);
      throw error;
    }
  }, []);

  /**
   * Select a conversation
   * Uses pushState for browser history (back/forward navigation)
   */
  const selectConversation = useCallback((id: string) => {
    // Only update if different conversation
    if (id === currentConversationId) return;

    // Update state
    setCurrentConversationId(id);

    // Add to browser history for back/forward navigation
    const url = new URL(window.location.href);
    url.searchParams.set('c', id);
    window.history.pushState({}, '', url.toString());
  }, [currentConversationId]);

  /**
   * Load messages when conversation is selected
   * This ensures messages are fetched from database after page refresh
   */
  useEffect(() => {
    if (!currentConversationId) return;

    // Wait for conversations to load before fetching messages
    // This prevents unnecessary API calls for invalid/deleted conversation IDs
    if (conversations.length === 0 && isAuthenticated) return;

    // Verify conversation exists before loading messages
    const conversationExists = conversations.some(c => c.id === currentConversationId);
    if (!conversationExists && conversations.length > 0) return;

    const loadMessages = async () => {
      try {
        const messages = await apiService.getConversationHistory(currentConversationId);

        setConversations((prev) =>
          prev.map((conv) => {
            if (conv.id !== currentConversationId) return conv;
            // Don't overwrite if conversation already has messages locally —
            // prevents race condition where sendMessage adds messages right after
            // createConversation, but this async fetch returns [] and wipes them.
            if (conv.messages.length > 0) return conv;
            return { ...conv, messages };
          })
        );
      } catch (error) {
        console.error('Failed to load conversation messages:', error);
      }
    };

    loadMessages();
  }, [currentConversationId, conversations.length, isAuthenticated]);

  /**
   * Delete a conversation
   * If the conversation is currently streaming, abort its stream first.
   */
  const deleteConversation = useCallback(async (id: string) => {
    // Abort stream if this conversation is currently streaming
    const streamState = streamingRefsMap.current.get(id);
    if (streamState) {
      console.log(`🛑 useChat: Aborting stream for deleted conversation ${id}`);
      streamState.abortController.abort();
      streamingRefsMap.current.delete(id);
      setStreamingConversationIds(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }

    try {
      // Delete from server
      await apiService.deleteConversation(id);

      // Update local state
      setConversations((prev) => prev.filter((c) => c.id !== id));

      if (currentConversationId === id) {
        setCurrentConversationId(null);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      // Still remove from UI even if server delete fails (eventual consistency)
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (currentConversationId === id) {
        setCurrentConversationId(null);
      }
    }
  }, [currentConversationId]);

  /**
   * Send a message and stream the response
   *
   * Per-conversation isolation: each call creates its own StreamingState in the
   * streamingRefsMap. The captured `conversation.id` scopes all SSE handlers to
   * that conversation, allowing multiple concurrent streams.
   *
   * @param content - The message content to send
   * @param addUserMessage - Whether to add a new user message (false for regenerate/edit)
   * @param selectedContext - Optional selected text from PDF for additional context
   * @param attachments - Optional file attachments for multimodal context
   */
  const sendMessage = useCallback(
    async (content: string, addUserMessage: boolean = true, selectedContext?: {
      text: string;
      documentId: string;
      documentName: string;
      pageStart: number;
      pageEnd: number;
    } | null, attachments?: Attachment[] | null, webSearchEnabled?: boolean) => {
      // Content validation (conversation-independent)
      if (!content.trim() && (!attachments || attachments.length === 0)) {
        return;
      }

      // Get current conversation using currentConversationId
      let conversation: Conversation;

      if (currentConversationId) {
        // Find conversation from REF (not state) to ensure we have latest version
        // This is critical for edit/regenerate where state might have been updated
        // in the same tick (truncation) but not yet committed to the 'conversations' variable
        const found = conversationsRef.current.find((c) => c.id === currentConversationId);
        if (found) {
          conversation = found;
        } else {
          conversation = await createConversation();
        }
      } else {
        conversation = await createConversation();
      }

      // Per-conversation streaming guard: block double-send to the SAME conversation
      if (streamingRefsMap.current.has(conversation.id)) {
        console.log(`⚠️ useChat: Conversation ${conversation.id} is already streaming, ignoring send`);
        return;
      }

      let updatedConversation: Conversation;

      if (addUserMessage) {
        // Create and add user message
        const userMessage: Message = {
          id: `msg_${Date.now()}_user`,
          role: 'user',
          content: content.trim() || (attachments?.length ? `[${attachments.length} attachment(s)]` : ''),
          timestamp: new Date().toISOString(),
          // Store selected context metadata for display below message
          selectedContext: selectedContext ? {
            documentId: selectedContext.documentId,
            documentName: selectedContext.documentName,
            lineCount: selectedContext.text.split('\n').filter(line => line.trim()).length || 1,
            pageStart: selectedContext.pageStart,
            pageEnd: selectedContext.pageEnd,
          } : undefined,
          // Store attachment metadata (without base64) for display
          attachments: attachments?.map(att => ({
            filename: att.filename,
            mimeType: att.mimeType,
            sizeBytes: att.sizeBytes,
          })),
        };

        // Add user message to conversation
        updatedConversation = {
          ...conversation,
          messages: [...conversation.messages, userMessage],
          updatedAt: new Date().toISOString(),
          title: conversation.title,  // Let backend generate LLM title
        };
      } else {
        // Regenerate/edit mode - use conversation as-is
        updatedConversation = {
          ...conversation,
          updatedAt: new Date().toISOString(),
        };
      }

      setConversations((prev) =>
        prev.map((c) => (c.id === updatedConversation.id ? updatedConversation : c))
      );

      // Create per-conversation streaming state
      const streamState: StreamingState = {
        currentMessage: {
          id: `msg_${Date.now()}_assistant`,
          role: 'assistant',
          content: '',
          timestamp: new Date().toISOString(),
          toolCalls: [],
          agentProgress: {
            currentAgent: 'orchestrator', // Start with orchestrator immediately
            currentMessage: 'Initializing...',
            completedAgents: [],
            activeTools: [],
            isStreaming: true // Keep indicator visible until done event
          }
        },
        currentToolCalls: new Map(),
        abortController: new AbortController(),
      };
      streamingRefsMap.current.set(conversation.id, streamState);

      // IMMEDIATE UPDATE: Add placeholder message to state so UI shows "Thinking..." instantly
      // This prevents the "empty gap" before first backend event arrives
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== updatedConversation.id) return c;
          return {
            ...c,
            messages: [...c.messages, streamState.currentMessage!]
          };
        })
      );

      // Add to streaming set (triggers UI re-render for spinners/buttons)
      setStreamingConversationIds(prev => new Set([...prev, conversation.id]));

      try {
        // Prepare message history for context (last 10 pairs = 20 messages)
        // Exclude the current user message if we just added it to avoid duplication
        const existingMessages = addUserMessage
          ? updatedConversation.messages.slice(0, -1)  // Exclude just-added user message
          : updatedConversation.messages;
        const historyLimit = 20; // Last 10 user+assistant pairs
        const messageHistory = existingMessages
          .slice(-historyLimit)
          .filter(msg => msg.role === 'user' || msg.role === 'assistant')
          .map(msg => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
          }));

        // Format selected context for API (convert camelCase to snake_case)
        const apiSelectedContext = selectedContext ? {
          text: selectedContext.text,
          document_id: selectedContext.documentId,
          document_name: selectedContext.documentName,
          page_start: selectedContext.pageStart,
          page_end: selectedContext.pageEnd,
        } : null;

        // Convert attachments to API format (camelCase -> snake_case)
        const apiAttachments = attachments?.length
          ? attachments.map(att => ({
              filename: att.filename,
              mime_type: att.mimeType,
              base64_data: att.base64Data,
            }))
          : null;

        // Stream response from backend with abort signal for page refresh handling
        for await (const event of apiService.streamChat(
          content,
          conversation.id,
          !addUserMessage,  // Skip saving user message when regenerating/editing (already exists)
          messageHistory,   // Pass conversation history for context
          streamState.abortController.signal,  // Allow cancellation on page refresh
          apiSelectedContext,  // Pass selected text from PDF for additional context
          apiAttachments,  // Pass file attachments for multimodal context
          webSearchEnabled,  // Per-request web search toggle
        )) {
          // Handle routing decision (8B router → 30B worker)
          if (event.event === 'routing') {
            if (streamState.currentMessage && streamState.currentMessage.agentProgress) {
              const { decision, complexity, thinking_budget } = event.data;
              if (decision === 'classifying') {
                streamState.currentMessage.agentProgress.currentMessage = 'Classifying query...';
              } else if (decision === 'delegate') {
                streamState.currentMessage.agentProgress.currentMessage =
                  `Delegating to thinking agent (${complexity}, budget: ${thinking_budget})...`;
              } else if (decision === 'direct') {
                streamState.currentMessage.agentProgress.currentMessage = 'Answering directly...';
              }

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;
                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];
                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...streamState.currentMessage! };
                  } else {
                    messages.push({ ...streamState.currentMessage! });
                  }
                  return { ...c, messages };
                })
              );
            }
          }
          // Handle tool health check (first event)
          else if (event.event === 'tool_health') {
            // Log tool health status (visible in browser console)
            if (!event.data.healthy) {
              console.warn('Tool health warning:', event.data.summary);
            } else {
              console.info('Tool health:', event.data.summary);
            }
            // Store tool health in message metadata for debugging
            if (streamState.currentMessage) {
              streamState.currentMessage.toolHealth = {
                healthy: event.data.healthy,
                summary: event.data.summary,
                unavailableTools: event.data.unavailable_tools,
                degradedTools: event.data.degraded_tools,
              };
            }
          }
          // Handle agent progress events
          else if (event.event === 'agent_start') {
            if (streamState.currentMessage && streamState.currentMessage.agentProgress) {
              // Mark previous agent as completed
              if (streamState.currentMessage.agentProgress.currentAgent) {
                streamState.currentMessage.agentProgress.completedAgents.push(
                  streamState.currentMessage.agentProgress.currentAgent
                );
              }

              // Set new current agent and clear activeTools
              streamState.currentMessage.agentProgress.currentAgent = event.data.agent;
              streamState.currentMessage.agentProgress.currentMessage = event.data.message;
              streamState.currentMessage.agentProgress.activeTools = [];

              // Update UI - IMPORTANT: Deep copy agentProgress to trigger React re-render
              // Capture snapshot of current message to avoid race conditions where ref becomes null
              const currentMessageSnapshot = { ...streamState.currentMessage };
              const currentAgentProgressSnapshot = streamState.currentMessage.agentProgress
                ? {
                  ...streamState.currentMessage.agentProgress,
                  activeTools: [...(streamState.currentMessage.agentProgress.activeTools || [])],
                  completedAgents: [...(streamState.currentMessage.agentProgress.completedAgents || [])]
                }
                : undefined;

              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  // Deep copy message with agentProgress to ensure React detects changes
                  const updatedMessage = {
                    ...currentMessageSnapshot,
                    agentProgress: currentAgentProgressSnapshot
                  };

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = updatedMessage;
                  } else {
                    messages.push(updatedMessage);
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'tool_call') {
            // Tool call event (running/completed/failed)
            // Show tool activity in thinking stream instead of inline ToolCallDisplay
            if (streamState.currentMessage && streamState.currentMessage.agentProgress) {
              const { tool, status, timestamp } = event.data;

              if (status === 'running') {
                // Add new tool to activeTools
                streamState.currentMessage.agentProgress.activeTools.push({
                  tool,
                  status,
                  timestamp
                });
                // Log to thinking stream
                streamState.currentMessage.thinkingContent =
                  (streamState.currentMessage.thinkingContent || '') + `▶ ${tool}\n`;
                streamState.currentMessage.isThinking = true;
              } else if (status === 'completed' || status === 'failed') {
                // Update status of existing tool
                const toolIndex = streamState.currentMessage.agentProgress.activeTools.findIndex(
                  t => t.tool === tool && t.status === 'running'
                );
                if (toolIndex >= 0) {
                  streamState.currentMessage.agentProgress.activeTools[toolIndex].status = status;
                }
                // Log to thinking stream
                const icon = status === 'completed' ? '✓' : '✗';
                streamState.currentMessage.thinkingContent =
                  (streamState.currentMessage.thinkingContent || '') + `${icon} ${tool}\n`;
              }

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...streamState.currentMessage! };
                  } else {
                    messages.push({ ...streamState.currentMessage! });
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'thinking_delta') {
            // Append thinking content (live from local LLM)
            if (!streamState.currentMessage) {
              console.warn('thinking_delta received but no currentMessage exists');
            } else if (streamState.currentMessage) {
              streamState.currentMessage.thinkingContent =
                (streamState.currentMessage.thinkingContent || '') + event.data.content;
              streamState.currentMessage.isThinking = true;

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...streamState.currentMessage! };
                  } else {
                    messages.push({ ...streamState.currentMessage! });
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'thinking_done') {
            // Thinking phase finished
            if (!streamState.currentMessage) {
              console.warn('thinking_done received but no currentMessage exists');
            } else if (streamState.currentMessage) {
              streamState.currentMessage.isThinking = false;
              // Keep thinkingContent for now; clear on first text_delta

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...streamState.currentMessage! };
                  } else {
                    messages.push({ ...streamState.currentMessage! });
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'text_delta') {
            // Append text delta
            if (streamState.currentMessage) {
              // Keep thinkingContent (shown as collapsible bar) but mark thinking as done
              if (streamState.currentMessage.isThinking) {
                streamState.currentMessage.isThinking = false;
              }

              // Hide progress indicator only when we have substantive text content
              // (not just whitespace artifacts from think-tag transitions)
              const hasSubstantiveContent = (streamState.currentMessage.content + event.data.content).trim().length > 0;
              if (hasSubstantiveContent && streamState.currentMessage.agentProgress && streamState.currentMessage.agentProgress.currentAgent) {
                streamState.currentMessage.agentProgress.currentAgent = null;
                // Keep isStreaming=true — it's cleared by the 'done' event
              }

              streamState.currentMessage.content += event.data.content;

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    // Update existing assistant message
                    messages[messages.length - 1] = { ...streamState.currentMessage! };
                  } else {
                    // Add new assistant message
                    messages.push({ ...streamState.currentMessage! });
                  }

                  return { ...c, messages };
                })
              );
            }
          }
          else if (event.event === 'tool_result') {
            // Tool execution completed
            const callId = event.data.call_id;
            const existingToolCall = streamState.currentToolCalls.get(callId);

            if (existingToolCall) {
              existingToolCall.result = event.data.result;
              existingToolCall.executionTimeMs = event.data.execution_time_ms;
              existingToolCall.success = event.data.success;
              existingToolCall.status = event.data.success ? 'completed' : 'failed';

              if (streamState.currentMessage) {
                streamState.currentMessage.toolCalls = Array.from(
                  streamState.currentToolCalls.values()
                );
              }

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...streamState.currentMessage! };

                  return { ...c, messages };
                })
              );
            } else {
              // Tool result without matching tool call (race condition or backend error)
              console.error('❌ Received tool_result for unknown call_id:', {
                callId,
                knownCallIds: Array.from(streamState.currentToolCalls.keys()),
                resultData: event.data
              });

              // Add warning message to chat
              if (streamState.currentMessage) {
                streamState.currentMessage.content +=
                  `\n\n[Warning: Received result for tool call ${callId} but call not found]`;
              }
            }
          } else if (event.event === 'tool_calls_summary') {
            // Tool calls summary from backend
            if (streamState.currentMessage && event.data.tool_calls) {
              // Convert backend tool calls to frontend format
              streamState.currentMessage.toolCalls = event.data.tool_calls.map((tc: any, idx: number) => ({
                id: tc.id || `summary-${idx}`,
                name: tc.name,
                input: tc.input || {},
                result: tc.result,
                executionTimeMs: tc.executionTimeMs,
                success: tc.success ?? true,
                status: tc.success === false ? 'failed' as const : 'completed' as const,
                explicitParams: tc.explicitParams,
              }));

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...streamState.currentMessage! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'cost_summary') {
            // Cost tracking summary with per-agent breakdown
            console.log('💰 FRONTEND: Cost summary received:', event.data);

            if (streamState.currentMessage) {
              // Validate and sanitize cost data
              const totalCost = typeof event.data.total_cost === 'number' ? event.data.total_cost : 0;
              const inputTokens = typeof event.data.total_input_tokens === 'number' ? event.data.total_input_tokens : 0;
              const outputTokens = typeof event.data.total_output_tokens === 'number' ? event.data.total_output_tokens : 0;
              const agentBreakdown = Array.isArray(event.data.agent_breakdown) ? event.data.agent_breakdown : [];

              // Log validation warnings
              if (totalCost < 0 || isNaN(totalCost)) {
                console.error('❌ Invalid cost value received:', event.data.total_cost);
              }
              if (inputTokens < 0 || outputTokens < 0) {
                console.error('❌ Invalid token counts:', { inputTokens, outputTokens });
              }

              streamState.currentMessage.cost = {
                totalCost: Math.max(0, totalCost),
                inputTokens: Math.max(0, inputTokens),
                outputTokens: Math.max(0, outputTokens),
                cachedTokens: event.data.cache_stats?.cache_read_tokens || 0,
                agentBreakdown,
                cacheStats: event.data.cache_stats || { cache_read_tokens: 0, cache_creation_tokens: 0 },
              };

              // Update UI
              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== updatedConversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...streamState.currentMessage! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'clarification_needed') {
            // HITL: Agent needs clarification
            console.log('🤔 FRONTEND: Clarification needed:', event.data);

            setClarificationData(event.data);
            setAwaitingClarification(true);

            // Don't emit "done" - workflow is paused
            return;
          } else if (event.event === 'title_update') {
            // Title was generated by LLM for new conversation
            const newTitle = event.data?.title;
            if (newTitle) {
              updateConversationById(setConversations, updatedConversation.id, { title: newTitle });
            }
          } else if (event.event === 'message_saved') {
            // Message saved to database - capture message ID for feedback
            if (streamState.currentMessage && event.data?.message_id) {
              streamState.currentMessage.dbMessageId = event.data.message_id;
            }
          } else if (event.event === 'done') {
            // Stream completed - mark streaming as finished
            if (streamState.currentMessage?.agentProgress) {
              streamState.currentMessage.agentProgress.isStreaming = false;
              streamState.currentMessage.agentProgress.currentAgent = null;
            }
            // Keep thinkingContent for collapsible display, just ensure isThinking is false
            if (streamState.currentMessage) {
              streamState.currentMessage.isThinking = false;
            }
            // Capture run_id for feedback correlation (LangSmith trace ID)
            if (streamState.currentMessage && event.data?.run_id) {
              streamState.currentMessage.runId = event.data.run_id;
            }
            // Don't break yet - wait for message_saved event
          } else if (event.event === 'error') {
            // Error occurred
            console.error('Stream error:', event.data);

            // Handle spending limit exceeded error specially
            if (event.data.type === 'SpendingLimitExceeded') {
              setSpendingLimitError({
                message_cs: event.data.message_cs || 'Byl dosažen limit výdajů.',
                message_en: event.data.message_en || 'Spending limit reached.',
                total_spent_czk: event.data.total_spent_czk || 0,
                spending_limit_czk: event.data.spending_limit_czk || 0,
              });
              // Don't add error to message - show modal instead
              break;
            }

            if (streamState.currentMessage) {
              streamState.currentMessage.content += `\n\n[Error: ${event.data.error}]`;
            }
          }
        }

        // Ensure final message is in state before saving
        // Capture the ref value before async operations
        const finalMessage = streamState.currentMessage;
        if (finalMessage) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== updatedConversation.id) return c;

              const messages = [...c.messages];
              const lastMsg = messages[messages.length - 1];

              // Update or add final assistant message
              if (lastMsg?.role === 'assistant') {
                messages[messages.length - 1] = { ...finalMessage };
              } else {
                messages.push({ ...finalMessage });
              }

              // Clean messages - remove any incomplete/invalid messages
              const cleanedMessages = cleanMessages(messages);

              // Sync messageCount with actual messages length
              const finalConv = {
                ...c,
                messages: cleanedMessages,
                messageCount: cleanedMessages.length,
                updatedAt: new Date().toISOString()
              };

              return finalConv;
            })
          );
        }
      } catch (error) {
        // Check if this was an intentional abort (page refresh)
        if (error instanceof Error && error.name === 'AbortError') {
          console.log('🔄 useChat: Stream aborted (page refresh or navigation)');
          // Don't show error to user - this is expected behavior
        } else {
          console.error('❌ Error during streaming:', error);
          console.error('Error stack:', (error as Error).stack);

          // Add error message
          if (streamState.currentMessage) {
            streamState.currentMessage.content += `\n\n[Error: ${(error as Error).message}]`;
          }
        }
      } finally {
        // Clean up per-conversation streaming state
        streamingRefsMap.current.delete(conversation.id);
        setStreamingConversationIds(prev => {
          const next = new Set(prev);
          next.delete(conversation.id);
          return next;
        });
        // Invalidate spending cache and trigger refresh after each message
        apiService.invalidateSpendingCache();
        setSpendingRefreshTrigger((prev) => prev + 1);
      }
    },
    [createConversation, currentConversationId, conversations]
  );

  /**
   * Edit a user message and resend
   */
  const editMessage = useCallback(
    async (messageId: string, newContent: string) => {
      // Per-conversation guard: block edit only if THIS conversation is streaming
      if (streamingRefsMap.current.has(currentConversationId ?? '') || !newContent.trim()) return;

      // Use functional update to get current conversation state
      let shouldSend = false;
      let messageIndex = -1;
      let conversationId = '';

      setConversations((prev) => {
        // Find the current conversation
        const currentConv = prev.find((c) => c.id === currentConversationId);
        if (!currentConv) return prev;

        // Find the message index
        messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);
        if (messageIndex === -1 || currentConv.messages[messageIndex].role !== 'user') {
          return prev;
        }

        conversationId = currentConv.id;
        shouldSend = true;

        // Remove all messages after this one (including assistant response)
        const updatedMessages = currentConv.messages.slice(0, messageIndex);

        // Clean messages before saving
        const cleanedMessages = cleanMessages(updatedMessages);

        // Update conversation with truncated messages
        const updatedConversation: Conversation = {
          ...currentConv,
          messages: cleanedMessages,
          updatedAt: new Date().toISOString(),
        };

        // Return updated conversations array
        return prev.map((c) => (c.id === currentConversationId ? updatedConversation : c));
      });

      // If validation passed, delete from backend and send the edited message
      if (shouldSend) {
        // Delete from backend in background (don't await - fire and forget for immediate UI response)
        apiService.truncateMessagesAfter(conversationId, messageIndex).catch((error) => {
          console.error('Failed to truncate messages in database:', error);
          // Non-critical - frontend state is already updated
        });

        // Race condition fix: Wait for React to commit state update before sending
        // Without this, sendMessage may read stale conversation state (pre-update)
        // causing duplicate messages or incorrect context.
        //
        // Why 10ms: Minimum delay that ensures React has flushed state to DOM in all browsers
        // (0ms would queue immediately, potentially before state commit in concurrent mode)
        //
        // Alternative considered: useEffect with dependency, but creates unnecessary rerender
        await new Promise(resolve => setTimeout(resolve, 10));
        await sendMessage(newContent);
      }
    },
    [sendMessage, currentConversationId]
  );

  /**
   * Regenerate the last assistant response
   */
  const regenerateMessage = useCallback(
    async (messageId: string) => {
      // Per-conversation guard: block regenerate only if THIS conversation is streaming
      if (streamingRefsMap.current.has(currentConversationId ?? '')) {
        return;
      }

      // SYNCHRONOUSLY read current conversation from ref (always fresh)
      const currentConv = conversationsRef.current.find((c) => c.id === currentConversationId);

      if (!currentConv) {
        return;
      }

      // Find the message index
      const messageIndex = currentConv.messages.findIndex((m) => m.id === messageId);

      if (messageIndex === -1) {
        return;
      }

      const message = currentConv.messages[messageIndex];

      if (message.role !== 'assistant') {
        return;
      }

      // Find the user message before this assistant message
      const userMessageIndex = messageIndex - 1;

      if (userMessageIndex < 0) {
        return;
      }

      const userMessage = currentConv.messages[userMessageIndex];

      if (userMessage.role !== 'user') {
        return;
      }

      const userMessageContent = userMessage.content;
      const conversationId = currentConv.id;

      // Remove all messages after (and including) the assistant message we want to regenerate
      const updatedMessages = currentConv.messages.slice(0, messageIndex);

      // Clean messages before saving
      const cleanedMessages = cleanMessages(updatedMessages);

      // Update conversation
      const updatedConversation: Conversation = {
        ...currentConv,
        messages: cleanedMessages,
        updatedAt: new Date().toISOString(),
      };

      // IMMEDIATELY update state to remove message from UI
      setConversations((prev) =>
        prev.map((c) => (c.id === conversationId ? updatedConversation : c))
      );

      // Delete from backend in background (don't await - fire and forget for immediate UI response)
      apiService.truncateMessagesAfter(conversationId, messageIndex).catch((error) => {
        console.error('Failed to truncate messages in database:', error);
        // Non-critical - frontend state is already updated
      });

      // Race condition fix: Wait for React to commit state update before sending
      // Without this, sendMessage may read stale conversation state (pre-truncation)
      // causing duplicate messages or incorrect context.
      //
      // Why 10ms: Minimum delay that ensures React has flushed state to DOM in all browsers
      // (0ms would queue immediately, potentially before state commit in concurrent mode)
      //
      // Alternative considered: useEffect with dependency, but creates unnecessary rerender
      await new Promise(resolve => setTimeout(resolve, 10));
      // Pass false to prevent adding a new user message (we're regenerating from existing)
      await sendMessage(userMessageContent, false);
    },
    [sendMessage, currentConversationId, cleanMessages]
  );

  /**
   * Submit clarification response and resume workflow
   */
  const submitClarification = useCallback(
    async (response: string) => {
      if (!clarificationData || !response.trim()) {
        return;
      }

      const { thread_id } = clarificationData;

      // Reset clarification state
      setClarificationData(null);
      setAwaitingClarification(false);

      // Get current conversation
      const conversation = conversations.find((c) => c.id === currentConversationId);
      if (!conversation) {
        console.error('No current conversation found');
        return;
      }

      // Create per-conversation streaming state for clarification
      const streamState: StreamingState = {
        currentMessage: {
          id: `msg_${Date.now()}_assistant`,
          role: 'assistant',
          content: '',
          timestamp: new Date().toISOString(),
          toolCalls: [],
        },
        currentToolCalls: new Map(),
        abortController: new AbortController(),
      };
      streamingRefsMap.current.set(conversation.id, streamState);
      setStreamingConversationIds(prev => new Set([...prev, conversation.id]));

      try {
        // Stream clarification response from backend
        for await (const event of apiService.streamClarification(thread_id, response)) {
          console.log('📨 FRONTEND: Received clarification event:', event.event);

          if (event.event === 'text_delta') {
            // Append text delta
            if (streamState.currentMessage) {
              streamState.currentMessage.content += event.data.content;

              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== conversation.id) return c;

                  const messages = [...c.messages];
                  const lastMsg = messages[messages.length - 1];

                  if (lastMsg?.role === 'assistant') {
                    messages[messages.length - 1] = { ...streamState.currentMessage! };
                  } else {
                    messages.push({ ...streamState.currentMessage! });
                  }

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'cost_summary') {
            // Cost tracking update
            if (streamState.currentMessage) {
              streamState.currentMessage.cost = {
                totalCost: event.data.total_cost,
                inputTokens: event.data.input_tokens,
                outputTokens: event.data.output_tokens,
                cachedTokens: event.data.cached_tokens,
                agentBreakdown: event.data.agent_breakdown,
              };

              setConversations((prev) =>
                prev.map((c) => {
                  if (c.id !== conversation.id) return c;

                  const messages = [...c.messages];
                  messages[messages.length - 1] = { ...streamState.currentMessage! };

                  return { ...c, messages };
                })
              );
            }
          } else if (event.event === 'message_saved') {
            // Message saved to database - capture message ID for feedback
            if (streamState.currentMessage && event.data?.message_id) {
              streamState.currentMessage.dbMessageId = event.data.message_id;
            }
          } else if (event.event === 'done') {
            // Stream completed - mark streaming as finished
            if (streamState.currentMessage?.agentProgress) {
              streamState.currentMessage.agentProgress.isStreaming = false;
              streamState.currentMessage.agentProgress.currentAgent = null;
            }
            // Capture run_id for feedback correlation (LangSmith trace ID)
            if (streamState.currentMessage && event.data?.run_id) {
              streamState.currentMessage.runId = event.data.run_id;
            }
            // Don't break yet - wait for message_saved event
          } else if (event.event === 'error') {
            // Error occurred
            console.error('Clarification stream error:', event.data);

            if (streamState.currentMessage) {
              streamState.currentMessage.content += `\n\n[Error: ${event.data.error}]`;
            }
          }
        }

        // Save final message
        const finalMessage = streamState.currentMessage;
        if (finalMessage) {
          setConversations((prev) =>
            prev.map((c) => {
              if (c.id !== conversation.id) return c;

              const messages = [...c.messages];
              const lastMsg = messages[messages.length - 1];

              if (lastMsg?.role === 'assistant') {
                messages[messages.length - 1] = { ...finalMessage };
              } else {
                messages.push({ ...finalMessage });
              }

              const cleanedMessages = cleanMessages(messages);
              // Sync messageCount with actual messages length
              const finalConv = {
                ...c,
                messages: cleanedMessages,
                messageCount: cleanedMessages.length,
                updatedAt: new Date().toISOString()
              };

              return finalConv;
            })
          );
        }
      } catch (error) {
        console.error('❌ Error during clarification streaming:', error);

        if (streamState.currentMessage) {
          streamState.currentMessage.content += `\n\n[Error: ${(error as Error).message}]`;
        }
      } finally {
        // Clean up per-conversation streaming state
        streamingRefsMap.current.delete(conversation.id);
        setStreamingConversationIds(prev => {
          const next = new Set(prev);
          next.delete(conversation.id);
          return next;
        });
      }
    },
    [clarificationData, conversations, currentConversationId, cleanMessages]
  );

  /**
   * Cancel clarification and continue with original query
   */
  const cancelClarification = useCallback(() => {
    console.log('🚫 FRONTEND: Clarification cancelled - continuing with original query');
    setClarificationData(null);
    setAwaitingClarification(false);

    // Clean up streaming state for current conversation if it was streaming
    if (currentConversationId) {
      const streamState = streamingRefsMap.current.get(currentConversationId);
      if (streamState) {
        streamState.abortController.abort();
        streamingRefsMap.current.delete(currentConversationId);
      }
      setStreamingConversationIds(prev => {
        const next = new Set(prev);
        next.delete(currentConversationId);
        return next;
      });
    }
  }, [currentConversationId]);

  /**
   * Rename a conversation
   * @param id - Conversation ID
   * @param newTitle - New title for the conversation
   */
  const renameConversation = useCallback(
    async (id: string, newTitle: string) => {
      if (!newTitle.trim()) {
        console.warn('Cannot rename conversation to empty title');
        return;
      }

      try {
        // Update on server
        await apiService.updateConversationTitle(id, newTitle.trim());

        // Update local state
        updateConversationById(setConversations, id, {
          title: newTitle.trim(),
          updatedAt: new Date().toISOString(),
        });
      } catch (error) {
        console.error('Failed to rename conversation:', error);
        // Don't throw - UI should handle gracefully
      }
    },
    []
  );

  /**
   * Delete a message from the current conversation
   * @param messageId - ID of the message to delete
   */
  const deleteMessage = useCallback(
    async (messageId: string) => {
      if (!currentConversationId) {
        console.warn('No conversation selected');
        return;
      }

      // Try backend notification (non-blocking)
      try {
        await apiService.deleteMessage(currentConversationId, messageId);
      } catch (error) {
        console.error('Backend delete notification failed:', error);
      }

      // Always update UI (frontend-centric approach)
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== currentConversationId) return c;

          const updatedMessages = c.messages.filter((m) => m.id !== messageId);
          const updatedConv = {
            ...c,
            messages: updatedMessages,
            updatedAt: new Date().toISOString(),
          };

          return updatedConv;
        })
      );

      console.log(`✓ Message ${messageId} deleted locally`);
    },
    [currentConversationId]
  );

  /**
   * Cancel ongoing streaming for a specific conversation
   * @param conversationId - Optional conversation ID to cancel. Defaults to current conversation.
   */
  const cancelStreaming = useCallback((conversationId?: string) => {
    const targetId = conversationId || currentConversationId;
    if (!targetId) {
      console.log('🛑 useChat: cancelStreaming called but no target conversation');
      return;
    }

    const streamState = streamingRefsMap.current.get(targetId);
    if (!streamState) {
      console.log('🛑 useChat: cancelStreaming called but no active stream for', targetId);
      return;
    }

    console.log('🛑 useChat: User cancelled streaming for conversation', targetId);
    streamState.abortController.abort();

    // Clear agent progress in the message state (fixes progress animation on cancel)
    if (streamState.currentMessage) {
      const messageId = streamState.currentMessage.id;
      setConversations((prev) =>
        prev.map((conv) => {
          if (conv.id !== targetId) return conv;
          return {
            ...conv,
            messages: conv.messages.map((msg) =>
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

    // Clean up streaming state
    streamingRefsMap.current.delete(targetId);
    setStreamingConversationIds(prev => {
      const next = new Set(prev);
      next.delete(targetId);
      return next;
    });
  }, [currentConversationId]);

  /**
   * Clear spending limit error (dismiss modal)
   */
  const clearSpendingLimitError = useCallback(() => {
    setSpendingLimitError(null);
  }, []);

  return {
    conversations,
    currentConversation,
    isStreaming,
    streamingConversationIds,
    clarificationData,
    awaitingClarification,
    spendingLimitError,
    spendingRefreshTrigger,
    createConversation,
    selectConversation,
    deleteConversation,
    renameConversation,
    sendMessage,
    editMessage,
    regenerateMessage,
    deleteMessage,
    submitClarification,
    cancelClarification,
    cancelStreaming,
    clearSpendingLimitError,
  };
}
