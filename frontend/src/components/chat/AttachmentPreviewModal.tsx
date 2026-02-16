import { useState, useEffect, useCallback } from 'react';
import { X, Download, ChevronLeft, ChevronRight, FileText, File } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useFadeIn } from '../../design-system/animations/hooks/useFadeIn';
import type { MessageAttachmentMeta } from '../../types';
import { apiService } from '../../services/api';

export interface PreviewAttachment {
  meta: MessageAttachmentMeta;
  base64Data?: string;
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

  useEffect(() => {
    setCurrentIndex(initialIndex);
  }, [initialIndex, isOpen]);

  const navigatePrev = useCallback(() => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : attachments.length - 1));
  }, [attachments.length]);

  const navigateNext = useCallback(() => {
    setCurrentIndex((prev) => (prev < attachments.length - 1 ? prev + 1 : 0));
  }, [attachments.length]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') navigatePrev();
      if (e.key === 'ArrowRight') navigateNext();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, navigatePrev, navigateNext]);

  if (!isOpen || attachments.length === 0) return null;

  const current = attachments[currentIndex];
  if (!current) return null;

  const isImage = current.meta.mimeType.startsWith('image/');
  const isPdf = current.meta.mimeType === 'application/pdf';

  const getUrl = (): string | null => {
    if (current.base64Data) {
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
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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
            <span className="text-blue-500">
              {isImage ? <FileText size={16} /> : <File size={16} />}
            </span>
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

        {/* Navigation arrows */}
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
