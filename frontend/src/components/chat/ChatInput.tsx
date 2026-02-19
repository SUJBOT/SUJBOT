/**
 * ChatInput Component - Message input textarea with send/stop button and file attachments
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Square, FileText, X, Paperclip, Globe } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useAuth } from '../../contexts/AuthContext';
import { apiService, type SpendingInfo } from '../../services/api';
import type { TextSelection, Attachment } from '../../types';
import { AttachmentPreviewModal } from './AttachmentPreviewModal';
import { AttachmentChip } from './AttachmentChip';

const ALLOWED_MIME_TYPES = new Set([
  'image/png',
  'image/jpeg',
  'image/gif',
  'image/webp',
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  'text/markdown',
  'text/html',
  'application/x-tex',
  'text/x-tex',
  'application/x-latex',
]);

// Browsers often misidentify file types — use extension as fallback
const EXTENSION_MIME_MAP: Record<string, string> = {
  '.md': 'text/markdown',
  '.tex': 'application/x-tex',
  '.latex': 'application/x-latex',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.txt': 'text/plain',
  '.html': 'text/html',
  '.htm': 'text/html',
};

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
const MAX_ATTACHMENTS = 5;

interface ChatInputProps {
  onSend: (message: string, attachments?: Attachment[], webSearchEnabled?: boolean) => void;
  onCancel?: () => void;  // Cancel streaming
  isStreaming: boolean;   // Whether currently streaming
  disabled: boolean;      // Disabled for other reasons (not streaming)
  refreshSpendingTrigger?: number; // Increment to refresh spending data
  selectedText?: TextSelection | null;  // Selected text from PDF
  onClearSelection?: () => void;  // Clear selection callback
  webSearchEnabled: boolean;  // Web search toggle state (owned by useChat hook)
  onToggleWebSearch: () => void;  // Toggle callback
}

export function ChatInput({ onSend, onCancel, isStreaming, disabled, refreshSpendingTrigger, selectedText, onClearSelection, webSearchEnabled, onToggleWebSearch }: ChatInputProps) {
  const { t } = useTranslation();
  const { user } = useAuth();

  // Calculate line count for selection (filter out empty lines)
  const selectionLineCount = selectedText
    ? selectedText.text.split('\n').filter(line => line.trim()).length || 1
    : 0;
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [spending, setSpending] = useState<SpendingInfo | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Backend limit: 50,000 characters (see backend/models.py ChatRequest)
  const MAX_MESSAGE_LENGTH = 50000;
  const isMessageTooLong = message.length > MAX_MESSAGE_LENGTH;

  // Can send if there's text OR attachments
  const hasContent = message.trim().length > 0 || attachments.length > 0;

  // Calculate spending status for color coding
  const spendingPercentage = spending
    ? Math.min((spending.total_spent_czk / spending.spending_limit_czk) * 100, 100)
    : 0;
  const isBlocked = spending
    ? spending.total_spent_czk >= spending.spending_limit_czk
    : false;

  // Fetch spending data with rate limit protection
  const fetchSpending = useCallback(async () => {
    try {
      const data = await apiService.getSpending();
      setSpending(data);
    } catch (error) {
      // Log 429 (rate limit) but don't show error to user - will retry on next trigger
      if (error instanceof Error && error.message.includes('429')) {
        console.warn('Rate limited when fetching spending, will retry on next trigger');
        return;
      }
      console.error('Failed to fetch spending:', error);
    }
  }, []);

  // Fetch spending on mount and when trigger changes
  // Use a small delay to prevent simultaneous fetches from multiple ChatInput instances
  useEffect(() => {
    const delay = Math.random() * 100; // Random 0-100ms delay to stagger requests
    const timeoutId = setTimeout(fetchSpending, delay);
    return () => clearTimeout(timeoutId);
  }, [fetchSpending, refreshSpendingTrigger]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [message]);

  // Clear attachment error after 5 seconds
  useEffect(() => {
    if (attachmentError) {
      const timer = setTimeout(() => setAttachmentError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [attachmentError]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    // Snapshot files BEFORE resetting input — FileList is a live DOM reference
    // that gets emptied when input.value is cleared
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    // Reset file input for re-selection of same file
    e.target.value = '';

    const remainingSlots = MAX_ATTACHMENTS - attachments.length;
    if (remainingSlots <= 0) {
      setAttachmentError(t('attachments.tooManyFiles'));
      return;
    }

    const filesToProcess = files.slice(0, remainingSlots);
    if (files.length > remainingSlots) {
      setAttachmentError(t('attachments.tooManyFiles'));
    }

    for (const file of filesToProcess) {
      // Resolve MIME type with extension fallback (browsers misidentify .md, .tex, etc.)
      let mimeType = file.type;
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED_MIME_TYPES.has(mimeType) && ext && EXTENSION_MIME_MAP[ext]) {
        mimeType = EXTENSION_MIME_MAP[ext];
      }

      // Validate type
      if (!ALLOWED_MIME_TYPES.has(mimeType)) {
        setAttachmentError(t('attachments.unsupportedType'));
        continue;
      }

      // Validate size
      if (file.size > MAX_FILE_SIZE) {
        setAttachmentError(t('attachments.fileTooLarge'));
        continue;
      }

      // Read as base64
      try {
        const base64 = await readFileAsBase64(file);
        const attachment: Attachment = {
          id: `att_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          filename: file.name,
          mimeType: mimeType,
          base64Data: base64,
          sizeBytes: file.size,
        };
        setAttachments(prev => [...prev, attachment]);
        setAttachmentError(null);
      } catch (err) {
        console.error('Failed to read file:', err);
        setAttachmentError(t('attachments.unsupportedType'));
      }
    }
  }, [attachments.length, t]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (hasContent && !disabled && !isStreaming && !isMessageTooLong) {
      onSend(message.trim(), attachments.length > 0 ? attachments : undefined, webSearchEnabled);
      setMessage('');
      setAttachments([]);
    }
  };

  const handleCancel = () => {
    if (isStreaming && onCancel) {
      onCancel();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <form onSubmit={handleSubmit} className="p-6">
      <div className="max-w-4xl mx-auto">
        {/* Attachment preview chips */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2" style={{ animation: 'chipIn 0.15s ease-out' }}>
            {attachments.map((att, idx) => (
              <AttachmentChip
                key={att.id}
                filename={att.filename}
                mimeType={att.mimeType}
                sizeBytes={att.sizeBytes}
                onClick={() => setPreviewIndex(idx)}
                onRemove={() => removeAttachment(att.id)}
                removeTitle={t('attachments.remove')}
              />
            ))}
          </div>
        )}

        {/* Attachment error */}
        {attachmentError && (
          <div className="mb-2 text-xs text-red-600 dark:text-red-400" style={{ animation: 'chipIn 0.15s ease-out' }}>
            {attachmentError}
          </div>
        )}

        <div
          className={cn(
            'flex gap-3 p-2',
            'bg-white dark:bg-accent-900',
            'border border-accent-200 dark:border-accent-800',
            'rounded-2xl',
            'shadow-lg',
            'transition-all duration-300',
            'hover:shadow-xl',
            !disabled && 'hover:border-accent-300 dark:hover:border-accent-700'
          )}
        >
          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/gif,image/webp,application/pdf,.docx,.txt,.md,.html,.htm,.tex,.latex"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />

          {/* Attach button */}
          <button
            type="button"
            onClick={handleAttachClick}
            disabled={disabled || isStreaming || attachments.length >= MAX_ATTACHMENTS}
            className={cn(
              'flex-shrink-0 self-center',
              'w-10 h-10 rounded-xl',
              'flex items-center justify-center',
              'transition-all duration-200',
              disabled || isStreaming || attachments.length >= MAX_ATTACHMENTS
                ? 'text-accent-300 dark:text-accent-700 cursor-not-allowed'
                : cn(
                    'text-accent-500 dark:text-accent-400',
                    'hover:text-accent-700 dark:hover:text-accent-200',
                    'hover:bg-accent-100 dark:hover:bg-accent-800',
                  )
            )}
            title={t('attachments.attach')}
          >
            <Paperclip size={18} />
          </button>

          {/* Web search toggle */}
          <button
            type="button"
            onClick={onToggleWebSearch}
            disabled={disabled || isStreaming}
            className={cn(
              'flex-shrink-0 self-center',
              'w-10 h-10 rounded-xl',
              'flex items-center justify-center',
              'transition-all duration-200',
              disabled || isStreaming
                ? 'text-accent-300 dark:text-accent-700 cursor-not-allowed'
                : webSearchEnabled
                  ? cn(
                      'text-blue-600 dark:text-blue-400',
                      'bg-blue-50 dark:bg-blue-900/30',
                      'hover:bg-blue-100 dark:hover:bg-blue-900/50',
                    )
                  : cn(
                      'text-accent-400 dark:text-accent-600',
                      'hover:text-accent-600 dark:hover:text-accent-300',
                      'hover:bg-accent-100 dark:hover:bg-accent-800',
                    )
            )}
            title={webSearchEnabled ? t('chat.webSearchOn') : t('chat.webSearchOff')}
          >
            <Globe size={18} />
          </button>

          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={t('chat.placeholder')}
            disabled={disabled || isStreaming}
            className={cn(
              'flex-1 resize-none px-4 py-3',
              'bg-transparent',
              'text-accent-900 dark:text-accent-100',
              'placeholder:text-accent-400 dark:placeholder:text-accent-600',
              'focus:outline-none',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'scrollbar-hide overflow-y-auto'
            )}
            rows={1}
          />
          {isStreaming ? (
            /* Stop button - shown during streaming */
            <button
              type="button"
              onClick={handleCancel}
              className={cn(
                'flex-shrink-0 self-center',
                'w-12 h-12 rounded-xl',
                'bg-red-600 dark:bg-red-500',
                'text-white',
                'hover:bg-red-700 dark:hover:bg-red-600',
                'hover:scale-105 active:scale-95',
                'transition-all duration-200',
                'flex items-center justify-center',
                'shadow-md hover:shadow-lg'
              )}
              title={t('chat.stop')}
            >
              <Square size={20} fill="currentColor" />
            </button>
          ) : (
            /* Send button - shown when not streaming */
            <button
              type="submit"
              disabled={disabled || !hasContent || isMessageTooLong}
              className={cn(
                'flex-shrink-0 self-center',
                'w-12 h-12 rounded-xl',
                'flex items-center justify-center',
                'transition-all duration-200',
                disabled || !hasContent || isMessageTooLong
                  ? 'text-accent-400 dark:text-accent-600 opacity-50 cursor-not-allowed'
                  : cn(
                      'bg-accent-900 dark:bg-accent-100',
                      'text-white dark:text-accent-900',
                      'hover:bg-accent-800 dark:hover:bg-accent-200',
                      'hover:scale-105 active:scale-95',
                      'shadow-md hover:shadow-lg'
                    )
              )}
              title={
                disabled
                  ? t('chat.processing')
                  : isMessageTooLong
                  ? `${t('chat.messageTooLong')} (${message.length.toLocaleString()}/${MAX_MESSAGE_LENGTH.toLocaleString()})`
                  : t('chat.placeholder')
              }
            >
              <Send size={20} />
            </button>
          )}
        </div>
        {/* Bottom row: selection chip (left) + user info (right) */}
        <div className="mt-2 flex justify-between items-center text-xs">
          {/* Selection chip - left side */}
          {selectedText && onClearSelection ? (
            <div
              className={cn(
                'inline-flex items-center gap-1.5 px-2 py-0.5',
                'bg-gray-100 dark:bg-gray-800',
                'text-gray-600 dark:text-gray-400',
                'rounded-full',
                'border border-gray-200 dark:border-gray-700'
              )}
              style={{ animation: 'chipIn 0.15s ease-out' }}
            >
              <FileText size={10} className="text-gray-400 dark:text-gray-500" />
              <span>{t('selection.lines', { count: selectionLineCount })}</span>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <span className="truncate max-w-[120px]" title={selectedText.documentName}>
                {selectedText.documentName.replace(/_/g, ' ')}
              </span>
              <button
                type="button"
                onClick={onClearSelection}
                className={cn(
                  'p-0.5 rounded-full -mr-0.5',
                  'text-gray-400 dark:text-gray-500',
                  'hover:bg-gray-200 dark:hover:bg-gray-700',
                  'hover:text-gray-600 dark:hover:text-gray-300',
                  'transition-colors duration-150'
                )}
                title={t('selection.clearSelection')}
                aria-label={t('selection.clearSelection')}
              >
                <X size={10} />
              </button>
            </div>
          ) : (
            <div />
          )}

          {/* User email + spending - right side */}
          <div className="text-right flex-shrink-0">
            {user && (
              <div className="text-accent-500 dark:text-accent-500 mb-0.5">
                {user.email}
              </div>
            )}
            {spending && (
              <div
                className={cn(
                  'font-medium transition-colors duration-200',
                  isBlocked
                    ? 'text-red-600 dark:text-red-400'
                    : spendingPercentage >= 90
                      ? 'text-red-600 dark:text-red-400'
                      : spendingPercentage >= 70
                        ? 'text-amber-600 dark:text-amber-400'
                        : 'text-green-600 dark:text-green-400'
                )}
              >
                {spending.total_spent_czk.toFixed(2)} / {spending.spending_limit_czk.toFixed(2)} Kč
                {isBlocked && (
                  <span className="ml-2 px-1.5 py-0.5 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded text-[10px]">
                    {t('chat.blocked')}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

      </div>

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
    </form>
  );
}

/**
 * Read file as base64 string (without data URL prefix).
 */
function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix (e.g., "data:image/png;base64,")
      const base64 = result.split(',')[1];
      if (!base64) {
        reject(new Error('Failed to extract base64 data'));
        return;
      }
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
