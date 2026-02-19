/**
 * UploadModal — Batch document upload with drag & drop.
 *
 * Features:
 * - Drag-and-drop files (+ click to browse)
 * - Per-file category toggle switch (legislation / documentation)
 * - Global access level selector (secret / public)
 * - Sequential upload with per-file progress
 * - Summary of results (success / failure counts)
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Upload,
  X,
  Scale,
  BookOpen,
  Lock,
  Globe,
  CheckCircle,
  AlertCircle,
  FileText,
  Trash2,
  Image as ImageIcon,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { apiService } from '../../services/api';
import type { UploadProgress, UploadResult } from '../../hooks/useDocumentUpload';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete: () => void;
}

const SUPPORTED_EXTENSIONS = [
  '.pdf', '.docx', '.txt', '.md', '.html', '.htm', '.tex', '.latex',
  '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp',
];
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp']);
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB

type Category = 'documentation' | 'legislation';

interface FileEntry {
  id: string;
  file: File;
  category: Category;
}

interface UploadFileResult {
  id: string;
  filename: string;
  success: boolean;
  result?: UploadResult;
  error?: string;
}

export function UploadModal({ isOpen, onClose, onUploadComplete }: UploadModalProps) {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [files, setFiles] = useState<FileEntry[]>([]);
  const [accessLevel, setAccessLevel] = useState<'public' | 'secret'>('public');
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Upload state
  const [isUploading, setIsUploading] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentProgress, setCurrentProgress] = useState<UploadProgress | null>(null);
  const [uploadResults, setUploadResults] = useState<UploadFileResult[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const [currentLabel, setCurrentLabel] = useState<string>('');
  const [totalJobs, setTotalJobs] = useState(0);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setFiles([]);
      setAccessLevel('public');
      setIsDragOver(false);
      setValidationError(null);
      setIsUploading(false);
      setCurrentIndex(0);
      setCurrentProgress(null);
      setUploadResults([]);
      abortRef.current = null;
      setCurrentLabel('');
      setTotalJobs(0);
    }
  }, [isOpen]);

  // Close on Escape (only when not uploading)
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isUploading) onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isUploading, onClose]);

  const isValidFile = useCallback((file: File): boolean => {
    const hasValidExt = SUPPORTED_EXTENSIONS.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );
    return hasValidExt && file.size <= MAX_FILE_SIZE;
  }, []);

  const isImageFile = useCallback((file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    return IMAGE_EXTENSIONS.has(ext);
  }, []);

  const addFiles = useCallback(
    (newFiles: File[]) => {
      setValidationError(null);

      const valid: FileEntry[] = [];
      let skipped = 0;

      for (const file of newFiles) {
        if (!isValidFile(file)) {
          skipped++;
          continue;
        }
        valid.push({
          id: crypto.randomUUID(),
          file,
          category: 'documentation',
        });
      }

      if (skipped > 0 && valid.length === 0) {
        setValidationError(t('documentBrowser.unsupportedFormat'));
      } else if (skipped > 0) {
        setValidationError(t('documentBrowser.someFilesSkipped', { count: skipped }));
      }

      if (valid.length > 0) {
        setFiles((prev) => [...prev, ...valid]);
        // Clear results from previous batch
        setUploadResults([]);
      }
    },
    [isValidFile, t]
  );

  // ── Event handlers ──

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = Array.from(e.target.files || []);
      if (fileList.length > 0) addFiles(fileList);
      e.target.value = '';
    },
    [addFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);
      const fileList = Array.from(e.dataTransfer.files);
      if (fileList.length > 0) addFiles(fileList);
    },
    [addFiles]
  );

  const removeFile = useCallback((id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const toggleCategory = useCallback((id: string) => {
    setFiles((prev) =>
      prev.map((f) =>
        f.id === id
          ? { ...f, category: f.category === 'documentation' ? 'legislation' : 'documentation' }
          : f
      )
    );
  }, []);

  const setAllCategories = useCallback((category: Category) => {
    setFiles((prev) => prev.map((f) => ({ ...f, category })));
  }, []);

  // ── Sequential upload ──

  const handleUpload = useCallback(async () => {
    if (files.length === 0) return;
    setIsUploading(true);
    setUploadResults([]);

    // Partition into images and documents
    const imageEntries = files.filter((f) => isImageFile(f.file));
    const docEntries = files.filter((f) => !isImageFile(f.file));

    // Build ordered list: image batch first (as one item), then individual docs
    type UploadJob = { type: 'image-batch'; entries: FileEntry[] } | { type: 'document'; entry: FileEntry };
    const jobs: UploadJob[] = [];
    if (imageEntries.length > 0) {
      jobs.push({ type: 'image-batch', entries: imageEntries });
    }
    for (const entry of docEntries) {
      jobs.push({ type: 'document', entry });
    }

    setTotalJobs(jobs.length);

    const results: UploadFileResult[] = [];
    let anySuccess = false;

    for (let i = 0; i < jobs.length; i++) {
      const job = jobs[i];
      setCurrentIndex(i);
      setCurrentProgress(null);

      if (job.type === 'image-batch') {
        setCurrentLabel(`${job.entries.length} images`);
      } else {
        setCurrentLabel(job.entry.file.name);
      }

      abortRef.current = new AbortController();

      try {
        let fileResult: UploadResult | undefined;
        let fileError: string | undefined;

        const stream = job.type === 'image-batch'
          ? apiService.uploadImages(
              job.entries.map((e) => e.file),
              abortRef.current.signal,
              job.entries[0].category,
              accessLevel
            )
          : apiService.uploadDocument(
              job.entry.file,
              abortRef.current.signal,
              job.entry.category,
              accessLevel
            );

        for await (const event of stream) {
          if (event.event === 'progress') {
            setCurrentProgress(event.data as UploadProgress);
          } else if (event.event === 'complete') {
            fileResult = event.data as UploadResult;
            anySuccess = true;
          } else if (event.event === 'error') {
            fileError = event.data.error || event.data.message || 'Upload failed';
          }
        }

        if (job.type === 'image-batch') {
          const batchName = `${job.entries.length} images`;
          results.push({
            id: job.entries[0].id,
            filename: batchName,
            success: !!fileResult,
            result: fileResult,
            error: fileError ?? (!fileResult ? 'Upload ended without confirmation' : undefined),
          });
        } else {
          results.push({
            id: job.entry.id,
            filename: job.entry.file.name,
            success: !!fileResult,
            result: fileResult,
            error: fileError ?? (!fileResult ? 'Upload ended without confirmation' : undefined),
          });
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          const label = job.type === 'image-batch'
            ? `${job.entries.length} images`
            : job.entry.file.name;
          const id = job.type === 'image-batch' ? job.entries[0].id : job.entry.id;
          results.push({ id, filename: label, success: false, error: 'Cancelled' });
          break;
        }
        const label = job.type === 'image-batch'
          ? `${job.entries.length} images`
          : job.entry.file.name;
        const id = job.type === 'image-batch' ? job.entries[0].id : job.entry.id;
        results.push({
          id,
          filename: label,
          success: false,
          error: err instanceof Error ? err.message : 'Unexpected error',
        });
      }
    }

    // Mark remaining jobs as not attempted (after cancel break)
    const attemptedCount = results.length;
    for (let j = attemptedCount; j < jobs.length; j++) {
      const job = jobs[j];
      const label = job.type === 'image-batch'
        ? `${job.entries.length} images`
        : job.entry.file.name;
      const id = job.type === 'image-batch' ? job.entries[0].id : job.entry.id;
      results.push({ id, filename: label, success: false, error: 'Cancelled' });
    }

    setUploadResults(results);
    setIsUploading(false);
    setCurrentProgress(null);
    abortRef.current = null;

    if (anySuccess) onUploadComplete();
  }, [files, accessLevel, isImageFile, onUploadComplete]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const handleClose = useCallback(() => {
    if (isUploading) return;
    onClose();
  }, [isUploading, onClose]);

  const formatSize = (bytes: number): string => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  if (!isOpen) return null;

  const showForm = !isUploading && uploadResults.length === 0;
  const showProgress = isUploading;
  const showResults = !isUploading && uploadResults.length > 0;

  const successCount = uploadResults.filter((r) => r.success).length;
  const failedCount = uploadResults.filter((r) => !r.success).length;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={handleClose}
    >
      <div
        className={cn(
          'relative w-full max-w-2xl mx-4',
          'bg-white dark:bg-accent-900',
          'rounded-2xl shadow-2xl',
          'border border-accent-200 dark:border-accent-700',
          'animate-in fade-in zoom-in-95 duration-200',
          'flex flex-col max-h-[85vh]'
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className={cn(
            'flex items-center justify-between px-6 py-4 flex-shrink-0',
            'border-b border-accent-200 dark:border-accent-700'
          )}
        >
          <div className="flex items-center gap-2">
            <Upload size={20} className="text-blue-500" />
            <h2 className="text-lg font-semibold text-accent-900 dark:text-accent-100">
              {t('documentBrowser.upload')}
            </h2>
          </div>
          <button
            onClick={handleClose}
            disabled={isUploading}
            className={cn(
              'p-1.5 rounded-lg transition-colors',
              'text-accent-500 hover:text-accent-700 hover:bg-accent-100',
              'dark:text-accent-400 dark:hover:text-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            <X size={18} />
          </button>
        </div>

        {/* Body — scrollable */}
        <div className="px-6 py-5 space-y-5 overflow-y-auto flex-1 min-h-0">
          {/* ── Form: dropzone + file list + selectors ── */}
          {showForm && (
            <>
              {/* Drop zone */}
              <div
                onDragOver={handleDragOver}
                onDragEnter={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  'flex flex-col items-center justify-center gap-3',
                  'py-8 px-6 rounded-xl cursor-pointer transition-all duration-200',
                  'border-2 border-dashed',
                  isDragOver
                    ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-500'
                    : 'border-accent-300 dark:border-accent-600 hover:border-blue-300 dark:hover:border-blue-500 hover:bg-accent-50 dark:hover:bg-accent-800/50'
                )}
              >
                <Upload
                  size={32}
                  className={cn(
                    'transition-colors',
                    isDragOver ? 'text-blue-500' : 'text-accent-400 dark:text-accent-500'
                  )}
                />
                <div className="text-center">
                  <p className="text-sm text-accent-700 dark:text-accent-300">
                    {t('documentBrowser.dropzone')}{' '}
                    <span className="text-accent-400 dark:text-accent-500">
                      {t('documentBrowser.dropzoneOr')}
                    </span>{' '}
                    <span className="text-blue-500 hover:text-blue-600 font-medium">
                      {t('documentBrowser.dropzoneBrowse')}
                    </span>
                  </p>
                  <p className="text-xs text-accent-400 dark:text-accent-500 mt-1">
                    {t('documentBrowser.dropzoneFormats')}
                  </p>
                </div>
              </div>

              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt,.md,.html,.htm,.tex,.latex,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp"
                multiple
                onChange={handleInputChange}
                className="hidden"
              />

              {/* Validation error */}
              {validationError && (
                <div className="flex items-center gap-2 text-sm text-red-500 dark:text-red-400">
                  <AlertCircle size={16} className="flex-shrink-0" />
                  <span>{validationError}</span>
                </div>
              )}

              {/* ── File list with per-file category switch ── */}
              {files.length > 0 && (
                <div className="space-y-2">
                  {/* Bulk category controls */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-accent-700 dark:text-accent-300">
                      {t('documentBrowser.filesSelected', { count: files.length })}
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setAllCategories('legislation')}
                        className={cn(
                          'flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors',
                          'text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20'
                        )}
                        title={t('documentBrowser.legislation')}
                      >
                        <Scale size={12} />
                        {t('documentBrowser.legislation')}
                      </button>
                      <span className="text-accent-300 dark:text-accent-600">|</span>
                      <button
                        onClick={() => setAllCategories('documentation')}
                        className={cn(
                          'flex items-center gap-1 px-2 py-1 rounded text-xs transition-colors',
                          'text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20'
                        )}
                        title={t('documentBrowser.documentation')}
                      >
                        <BookOpen size={12} />
                        {t('documentBrowser.documentation')}
                      </button>
                    </div>
                  </div>

                  {/* File rows */}
                  <div className="space-y-1 max-h-52 overflow-y-auto rounded-lg border border-accent-200 dark:border-accent-700 divide-y divide-accent-100 dark:divide-accent-800">
                    {files.map((entry) => (
                      <div
                        key={entry.id}
                        className="flex items-center gap-3 px-3 py-2 group"
                      >
                        {isImageFile(entry.file) ? (
                          <ImageIcon size={16} className="text-accent-400 dark:text-accent-500 flex-shrink-0" />
                        ) : (
                          <FileText size={16} className="text-accent-400 dark:text-accent-500 flex-shrink-0" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm text-accent-800 dark:text-accent-200 truncate">
                            {entry.file.name}
                          </p>
                          <p className="text-xs text-accent-400 dark:text-accent-500">
                            {formatSize(entry.file.size)}
                          </p>
                        </div>

                        {/* Category switch */}
                        <button
                          onClick={() => toggleCategory(entry.id)}
                          className={cn(
                            'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
                            'transition-all duration-200 flex-shrink-0',
                            entry.category === 'legislation'
                              ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'
                              : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                          )}
                          title={t('documentBrowser.selectCategory')}
                        >
                          {entry.category === 'legislation' ? (
                            <><Scale size={12} /> {t('documentBrowser.legislation')}</>
                          ) : (
                            <><BookOpen size={12} /> {t('documentBrowser.documentation')}</>
                          )}
                        </button>

                        {/* Remove */}
                        <button
                          onClick={() => removeFile(entry.id)}
                          className={cn(
                            'p-1 rounded transition-colors flex-shrink-0',
                            'text-accent-300 dark:text-accent-600',
                            'hover:text-red-500 dark:hover:text-red-400',
                            'hover:bg-red-50 dark:hover:bg-red-900/20',
                            'opacity-0 group-hover:opacity-100'
                          )}
                          title={t('documentBrowser.remove')}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Access level selector */}
              {files.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-accent-700 dark:text-accent-300 mb-2">
                    {t('documentBrowser.accessLevel')}
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setAccessLevel('secret')}
                      className={cn(
                        'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border transition-colors flex-1 justify-center',
                        accessLevel === 'secret'
                          ? 'bg-red-100 dark:bg-red-900/30 border-red-400 dark:border-red-600 text-red-800 dark:text-red-300'
                          : 'bg-accent-50 dark:bg-accent-800 border-accent-200 dark:border-accent-600 text-accent-600 dark:text-accent-400 hover:bg-accent-100 dark:hover:bg-accent-700'
                      )}
                    >
                      <Lock size={16} />
                      {t('documentBrowser.secret')}
                    </button>
                    <button
                      onClick={() => setAccessLevel('public')}
                      className={cn(
                        'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm border transition-colors flex-1 justify-center',
                        accessLevel === 'public'
                          ? 'bg-green-100 dark:bg-green-900/30 border-green-400 dark:border-green-600 text-green-800 dark:text-green-300'
                          : 'bg-accent-50 dark:bg-accent-800 border-accent-200 dark:border-accent-600 text-accent-600 dark:text-accent-400 hover:bg-accent-100 dark:hover:bg-accent-700'
                      )}
                    >
                      <Globe size={16} />
                      {t('documentBrowser.public')}
                    </button>
                  </div>
                </div>
              )}
            </>
          )}

          {/* ── Upload progress ── */}
          {showProgress && (
            <div className="space-y-4 py-4">
              {/* Overall progress */}
              <div className="text-sm font-medium text-accent-700 dark:text-accent-300 text-center">
                {t('documentBrowser.uploadProgress', {
                  current: currentIndex + 1,
                  total: totalJobs,
                })}
              </div>

              {/* Current file */}
              <div className="flex items-center gap-2 px-3 py-2 bg-accent-50 dark:bg-accent-800/50 rounded-lg">
                <FileText size={16} className="text-blue-500 flex-shrink-0" />
                <span className="text-sm text-accent-700 dark:text-accent-300 truncate">
                  {currentLabel}
                </span>
              </div>

              {/* Stage + bar */}
              {currentProgress && (
                <>
                  <div className="flex items-center justify-between text-sm text-accent-600 dark:text-accent-400">
                    <span>{t(`documentBrowser.${currentProgress.stage}` as any, currentProgress.message)}</span>
                    <span className="font-mono text-accent-500">{currentProgress.percent}%</span>
                  </div>
                  <div className="w-full h-2 bg-accent-200 dark:bg-accent-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${currentProgress.percent}%` }}
                    />
                  </div>
                </>
              )}

              {/* Overall bar (files completed / total) */}
              <div className="w-full h-1 bg-accent-200 dark:bg-accent-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full transition-all duration-300"
                  style={{ width: `${((currentIndex) / totalJobs) * 100}%` }}
                />
              </div>

              <div className="text-center">
                <button
                  onClick={handleCancel}
                  className={cn(
                    'text-sm px-4 py-1.5 rounded-lg transition-colors',
                    'text-red-500 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20'
                  )}
                >
                  {t('common.cancel')}
                </button>
              </div>
            </div>
          )}

          {/* ── Results summary ── */}
          {showResults && (
            <div className="space-y-4 py-4">
              {/* Summary header */}
              <div className="flex flex-col items-center gap-2">
                {failedCount === 0 ? (
                  <>
                    <CheckCircle size={40} className="text-green-500" />
                    <p className="text-sm font-medium text-green-700 dark:text-green-400">
                      {t('documentBrowser.uploadAllComplete', { count: successCount })}
                    </p>
                  </>
                ) : (
                  <>
                    <AlertCircle size={40} className="text-amber-500" />
                    <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
                      {t('documentBrowser.someUploadsFailed', {
                        success: successCount,
                        total: uploadResults.length,
                        failed: failedCount,
                      })}
                    </p>
                  </>
                )}
              </div>

              {/* Per-file results */}
              <div className="space-y-1 max-h-48 overflow-y-auto rounded-lg border border-accent-200 dark:border-accent-700 divide-y divide-accent-100 dark:divide-accent-800">
                {uploadResults.map((r) => (
                  <div key={r.id} className="flex items-center gap-3 px-3 py-2">
                    {r.success ? (
                      <CheckCircle size={16} className="text-green-500 flex-shrink-0" />
                    ) : (
                      <AlertCircle size={16} className="text-red-500 flex-shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-accent-800 dark:text-accent-200 truncate">
                        {r.filename}
                      </p>
                      {r.error && (
                        <p className="text-xs text-red-500 dark:text-red-400 truncate">
                          {r.error}
                        </p>
                      )}
                      {r.result && (
                        <p className="text-xs text-accent-500 dark:text-accent-400">
                          {r.result.pages} {t('pdfPanel.pages').toLowerCase()}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          className={cn(
            'flex items-center justify-end gap-3 px-6 py-4 flex-shrink-0',
            'border-t border-accent-200 dark:border-accent-700'
          )}
        >
          {showForm && (
            <>
              <button
                onClick={handleClose}
                className={cn(
                  'px-4 py-2 rounded-lg text-sm transition-colors',
                  'text-accent-600 dark:text-accent-400',
                  'hover:bg-accent-100 dark:hover:bg-accent-700'
                )}
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleUpload}
                disabled={files.length === 0}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  'bg-blue-500 text-white hover:bg-blue-600',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                <Upload size={16} />
                {t('documentBrowser.startUpload')}
                {files.length > 0 && (
                  <span className="ml-0.5 px-1.5 py-0.5 bg-blue-400/30 rounded text-xs">
                    {files.length}
                  </span>
                )}
              </button>
            </>
          )}
          {showResults && (
            <button
              onClick={onClose}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                'bg-accent-100 dark:bg-accent-800 text-accent-700 dark:text-accent-300',
                'hover:bg-accent-200 dark:hover:bg-accent-700'
              )}
            >
              {t('common.ok')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
