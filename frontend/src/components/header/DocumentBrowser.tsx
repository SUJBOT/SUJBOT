/**
 * DocumentBrowser Component
 *
 * Dropdown menu for browsing and selecting available PDF documents.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { FolderOpen, X, Loader2, RefreshCw, Upload, CheckCircle, AlertCircle, Scale, BookOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { apiService } from '../../services/api';
import type { DocumentInfo } from '../../types';
import { useCitationContext } from '../../contexts/CitationContext';
import { useDocumentUpload } from '../../hooks/useDocumentUpload';

interface DocumentBrowserProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DocumentBrowser({ isOpen, onClose }: DocumentBrowserProps) {
  const { t } = useTranslation();
  const { openPdf } = useCitationContext();
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { isUploading, progress, error: uploadError, result: uploadResult, startUpload, cancelUpload, reset: resetUpload } = useDocumentUpload();
  const [validationError, setValidationError] = useState<string | null>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<'documentation' | 'legislation'>('documentation');

  // Load documents when opening
  useEffect(() => {
    if (isOpen) {
      loadDocuments();
    } else {
      setPendingFile(null);
    }
  }, [isOpen]);

  // Handle click outside to close
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    // Delay to prevent immediate close on open click
    const timeoutId = setTimeout(() => {
      document.addEventListener('mousedown', handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const docs = await apiService.getDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError(t('documentBrowser.loadError'));
    } finally {
      setIsLoading(false);
    }
  };

  // Refresh document list when upload completes, auto-dismiss after 5s
  useEffect(() => {
    if (uploadResult) {
      loadDocuments();
      const timer = setTimeout(() => resetUpload(), 5000);
      return () => clearTimeout(timer);
    }
  }, [uploadResult]);

  // Auto-dismiss upload error after 5s
  useEffect(() => {
    if (uploadError) {
      const timer = setTimeout(() => resetUpload(), 5000);
      return () => clearTimeout(timer);
    }
  }, [uploadError]);

  // Auto-dismiss validation error after 5s
  useEffect(() => {
    if (validationError) {
      const timer = setTimeout(() => setValidationError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [validationError]);

  const handleSelectDocument = useCallback((doc: DocumentInfo) => {
    openPdf(doc.document_id, doc.display_name, 1);
    onClose();
  }, [openPdf, onClose]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset file input so same file can be re-selected
    e.target.value = '';
    setValidationError(null);

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setValidationError(t('documentBrowser.pdfOnly'));
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setValidationError(t('documentBrowser.fileTooLarge'));
      return;
    }

    resetUpload();
    setPendingFile(file);
    setSelectedCategory('documentation');
  }, [resetUpload, t]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(0)} KB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  if (!isOpen) return null;

  return (
    <div
      ref={dropdownRef}
      className={cn(
        'absolute top-full right-0 mt-2',
        'w-80 max-h-96 overflow-hidden',
        'bg-white dark:bg-accent-900',
        'border border-accent-200 dark:border-accent-700',
        'rounded-xl shadow-lg',
        'z-50',
        'animate-in fade-in slide-in-from-top-2 duration-200'
      )}
    >
      {/* Header */}
      <div className={cn(
        'flex items-center justify-between',
        'px-4 py-3',
        'border-b border-accent-200 dark:border-accent-700',
        'bg-accent-50 dark:bg-accent-800/50'
      )}>
        <div className="flex items-center gap-2">
          <FolderOpen size={18} className="text-accent-600 dark:text-accent-400" />
          <h3 className="font-medium text-sm text-accent-900 dark:text-accent-100">
            {t('documentBrowser.title')}
          </h3>
        </div>
        <div className="flex items-center gap-1">
          {/* Upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className={cn(
              'p-1.5 rounded-lg',
              'text-accent-500 hover:text-accent-700',
              'dark:text-accent-400 dark:hover:text-accent-200',
              'hover:bg-accent-100 dark:hover:bg-accent-700',
              'transition-colors',
              'disabled:opacity-50'
            )}
            title={t('documentBrowser.upload')}
          >
            <Upload size={14} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={loadDocuments}
            disabled={isLoading}
            className={cn(
              'p-1.5 rounded-lg',
              'text-accent-500 hover:text-accent-700',
              'dark:text-accent-400 dark:hover:text-accent-200',
              'hover:bg-accent-100 dark:hover:bg-accent-700',
              'transition-colors',
              'disabled:opacity-50'
            )}
            title="Refresh"
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={onClose}
            className={cn(
              'p-1.5 rounded-lg',
              'text-accent-500 hover:text-accent-700',
              'dark:text-accent-400 dark:hover:text-accent-200',
              'hover:bg-accent-100 dark:hover:bg-accent-700',
              'transition-colors'
            )}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Upload progress */}
      {(isUploading || uploadResult || uploadError || validationError) && (
        <div className={cn(
          'px-4 py-3',
          'border-b border-accent-200 dark:border-accent-700'
        )}>
          {isUploading && progress && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-accent-600 dark:text-accent-400">
                <span>{t(`documentBrowser.${progress.stage}` as any, progress.message)}</span>
                <div className="flex items-center gap-2">
                  <span>{progress.percent}%</span>
                  <button
                    onClick={cancelUpload}
                    className={cn(
                      'p-0.5 rounded',
                      'text-accent-400 hover:text-red-500',
                      'dark:text-accent-500 dark:hover:text-red-400',
                      'transition-colors'
                    )}
                    title={t('common.cancel')}
                  >
                    <X size={12} />
                  </button>
                </div>
              </div>
              <div className="w-full h-1.5 bg-accent-200 dark:bg-accent-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
            </div>
          )}
          {uploadResult && (
            <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
              <CheckCircle size={16} />
              <span>{t('documentBrowser.uploadComplete')} — {uploadResult.display_name} ({uploadResult.pages} {t('pdfPanel.pages')})</span>
            </div>
          )}
          {uploadError && (
            <div className="flex items-center gap-2 text-sm text-red-500 dark:text-red-400">
              <AlertCircle size={16} />
              <span>{uploadError}</span>
            </div>
          )}
          {validationError && (
            <div className="flex items-center gap-2 text-sm text-red-500 dark:text-red-400">
              <AlertCircle size={16} />
              <span>{validationError}</span>
            </div>
          )}
        </div>
      )}

      {/* Category selection dialog */}
      {pendingFile && !isUploading && (
        <div className={cn(
          'px-4 py-3',
          'border-b border-accent-200 dark:border-accent-700'
        )}>
          <p className="text-sm font-medium text-accent-800 dark:text-accent-200 mb-2">
            {t('documentBrowser.selectCategory')}
          </p>
          <div className="flex gap-2 mb-3">
            <button
              onClick={() => setSelectedCategory('legislation')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm border transition-colors',
                selectedCategory === 'legislation'
                  ? 'bg-amber-100 dark:bg-amber-900/30 border-amber-400 dark:border-amber-600 text-amber-800 dark:text-amber-300'
                  : 'bg-accent-50 dark:bg-accent-800 border-accent-200 dark:border-accent-600 text-accent-600 dark:text-accent-400 hover:bg-accent-100 dark:hover:bg-accent-700'
              )}
            >
              <Scale size={14} />
              {t('documentBrowser.legislation')}
            </button>
            <button
              onClick={() => setSelectedCategory('documentation')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm border transition-colors',
                selectedCategory === 'documentation'
                  ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-400 dark:border-blue-600 text-blue-800 dark:text-blue-300'
                  : 'bg-accent-50 dark:bg-accent-800 border-accent-200 dark:border-accent-600 text-accent-600 dark:text-accent-400 hover:bg-accent-100 dark:hover:bg-accent-700'
              )}
            >
              <BookOpen size={14} />
              {t('documentBrowser.documentation')}
            </button>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-accent-500 dark:text-accent-400 truncate max-w-[60%]">
              {pendingFile.name}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPendingFile(null)}
                className={cn(
                  'text-xs px-3 py-1.5 rounded-lg',
                  'text-accent-600 dark:text-accent-400',
                  'hover:bg-accent-100 dark:hover:bg-accent-700',
                  'transition-colors'
                )}
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={() => {
                  startUpload(pendingFile, selectedCategory);
                  setPendingFile(null);
                }}
                className={cn(
                  'text-xs px-3 py-1.5 rounded-lg',
                  'bg-blue-500 text-white',
                  'hover:bg-blue-600',
                  'transition-colors'
                )}
              >
                {t('documentBrowser.startUpload')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="overflow-y-auto max-h-72">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-accent-400" size={24} />
          </div>
        )}

        {error && !isLoading && (
          <div className="px-4 py-6 text-center">
            <p className="text-red-500 dark:text-red-400 text-sm mb-2">{error}</p>
            <button
              onClick={loadDocuments}
              className={cn(
                'text-xs px-3 py-1.5 rounded-lg',
                'bg-accent-100 dark:bg-accent-800',
                'hover:bg-accent-200 dark:hover:bg-accent-700',
                'text-accent-700 dark:text-accent-300',
                'transition-colors'
              )}
            >
              {t('common.retry') || 'Retry'}
            </button>
          </div>
        )}

        {!isLoading && !error && documents.length === 0 && (
          <div className="px-4 py-6 text-center text-accent-500 dark:text-accent-400 text-sm">
            {t('documentBrowser.noDocuments')}
          </div>
        )}

        {!isLoading && !error && documents.length > 0 && (() => {
          const legislation = documents.filter(d => d.category === 'legislation');
          const documentation = documents.filter(d => d.category !== 'legislation');

          const renderDocItem = (doc: DocumentInfo) => (
            <button
              key={doc.document_id}
              onClick={() => handleSelectDocument(doc)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'rounded-lg transition-colors',
                'text-left group'
              )}
            >
              {doc.category === 'legislation' ? (
                <Scale
                  size={16}
                  className="text-amber-500 dark:text-amber-400 flex-shrink-0"
                />
              ) : (
                <BookOpen
                  size={16}
                  className="text-blue-500 dark:text-blue-400 flex-shrink-0"
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="truncate font-medium text-sm text-accent-800 dark:text-accent-200">
                  {doc.display_name}
                </div>
                <div className="text-xs text-accent-500 dark:text-accent-400">
                  {formatSize(doc.size_bytes)}
                </div>
              </div>
            </button>
          );

          return (
            <div className="p-2 space-y-1">
              {legislation.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 px-3 pt-1 pb-1.5">
                    <Scale size={14} className="text-amber-500 dark:text-amber-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                      {t('documentBrowser.legislation')}
                    </span>
                    <span className="text-xs text-accent-400 dark:text-accent-500">
                      ({legislation.length})
                    </span>
                  </div>
                  {legislation.map(renderDocItem)}
                </div>
              )}
              {legislation.length > 0 && documentation.length > 0 && (
                <div className="border-t border-accent-200 dark:border-accent-700 my-1" />
              )}
              {documentation.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 px-3 pt-1 pb-1.5">
                    <BookOpen size={14} className="text-blue-500 dark:text-blue-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
                      {t('documentBrowser.documentation')}
                    </span>
                    <span className="text-xs text-accent-400 dark:text-accent-500">
                      ({documentation.length})
                    </span>
                  </div>
                  {documentation.map(renderDocItem)}
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}
