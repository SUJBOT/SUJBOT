/**
 * Hook for document upload with SSE progress tracking.
 *
 * The indexing pipeline runs as a background task on the backend — a page
 * refresh reconnects to the same in-progress upload via GET /upload-status.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { apiService } from '../services/api';

export interface UploadProgress {
  stage: string;
  percent: number;
  message: string;
}

export interface UploadResult {
  document_id: string;
  pages: number;
  display_name: string;
}

export function useDocumentUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResult | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Reconnect to active upload on mount (survives page refresh)
  useEffect(() => {
    let cancelled = false;

    async function checkActiveUpload() {
      let hasEvents = false;
      try {
        for await (const event of apiService.getUploadStatus()) {
          if (cancelled) break;

          if (!hasEvents) {
            hasEvents = true;
            setIsUploading(true);
          }

          if (event.event === 'progress') {
            setProgress(event.data as UploadProgress);
          } else if (event.event === 'complete') {
            setResult(event.data as UploadResult);
          } else if (event.event === 'error') {
            setError(event.data.error || event.data.message || 'Upload failed');
          }
        }
      } catch {
        // Reconnect check failed — no active upload
      } finally {
        if (!cancelled && hasEvents) {
          setIsUploading(false);
        }
      }
    }

    checkActiveUpload();

    return () => { cancelled = true; };
  }, []);

  const startUpload = useCallback(async (file: File, category?: string) => {
    setIsUploading(true);
    setProgress(null);
    setError(null);
    setResult(null);

    abortRef.current = new AbortController();

    try {
      for await (const event of apiService.uploadDocument(file, abortRef.current.signal, category)) {
        if (event.event === 'progress') {
          setProgress(event.data as UploadProgress);
        } else if (event.event === 'complete') {
          setResult(event.data as UploadResult);
        } else if (event.event === 'error') {
          setError(event.data.error || event.data.message || 'Upload failed');
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return; // User cancelled — not an error
      }
      console.error('Upload stream error:', err);
      setError(err instanceof Error ? err.message : 'Upload failed unexpectedly');
    } finally {
      setIsUploading(false);
      abortRef.current = null;
    }
  }, []);

  const cancelUpload = useCallback(async () => {
    // Abort local SSE stream
    abortRef.current?.abort();
    // Cancel backend pipeline
    try {
      await apiService.cancelUpload();
    } catch {
      // Best effort — backend may already be done
    }
    // Reset local state
    setIsUploading(false);
    setProgress(null);
    setError(null);
    setResult(null);
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setResult(null);
    setProgress(null);
  }, []);

  return { isUploading, progress, error, result, startUpload, cancelUpload, reset };
}
