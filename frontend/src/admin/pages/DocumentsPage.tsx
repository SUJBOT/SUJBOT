/**
 * Admin Document Management Page
 *
 * Lists all documents with metadata (page count, size, indexed date, category).
 * Supports upload (with category selection), delete, reindex, and category editing.
 * SSE progress streaming for upload and reindex operations.
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  FormControl,
  FormControlLabel,
  FormLabel,
  IconButton,
  LinearProgress,
  Radio,
  RadioGroup,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import { RefreshCw, Trash2, RotateCcw, Upload } from 'lucide-react';
import { API_BASE_URL } from '../../config';

type DocumentCategory = 'documentation' | 'legislation';

interface AdminDocument {
  document_id: string;
  display_name: string;
  filename: string;
  size_bytes: number;
  page_count: number;
  created_at: string | null;
  category: DocumentCategory;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Parse SSE text stream from a fetch Response.
 * Calls onEvent for each parsed event, onDone when stream ends.
 */
async function consumeSSE(
  response: Response,
  onEvent: (eventType: string, data: string) => void,
  onDone: () => void,
  onError: (err: string) => void,
) {
  const reader = response.body?.getReader();
  if (!reader) {
    onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on double newline (SSE event boundary)
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        if (!part.trim()) continue;

        let eventType = 'message';
        const dataLines: string[] = [];

        for (const line of part.split('\n')) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
          }
        }

        const data = dataLines.join('\n');

        if (data) {
          onEvent(eventType, data);
        }
      }
    }
  } catch (err) {
    onError(err instanceof Error ? err.message : 'Stream error');
    return;
  }

  onDone();
}

export const DocumentsPage = () => {
  const { t } = useTranslation();
  const [documents, setDocuments] = useState<AdminDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  // Action dialogs
  const [deleteTarget, setDeleteTarget] = useState<AdminDocument | null>(null);
  const [reindexTarget, setReindexTarget] = useState<AdminDocument | null>(null);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);

  // SSE progress dialog
  const [progressOpen, setProgressOpen] = useState(false);
  const [progressStage, setProgressStage] = useState('');
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [progressDone, setProgressDone] = useState(false);
  const [progressError, setProgressError] = useState<string | null>(null);

  // Upload dialog
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [uploadCategory, setUploadCategory] = useState<DocumentCategory>('documentation');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const fetchDocuments = useCallback(async () => {
    try {
      setRefreshing(true);
      const response = await fetch(`${API_BASE_URL}/admin/documents`, {
        credentials: 'include',
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) throw new Error('Failed to fetch documents');

      const data = await response.json();
      setDocuments(data.documents);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const stageLabel = (stage: string): string => {
    const key = `admin.documents.${stage}`;
    const translated = t(key);
    return translated !== key ? translated : stage;
  };

  // --- Delete ---
  const handleDelete = async () => {
    if (!deleteTarget) return;
    setActionInProgress(deleteTarget.document_id);
    setDeleteTarget(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/admin/documents/${deleteTarget.document_id}`,
        {
          method: 'DELETE',
          credentials: 'include',
          headers: { 'Accept': 'application/json' },
        },
      );

      if (!response.ok) throw new Error('Delete failed');

      await fetchDocuments();
    } catch {
      setError(t('admin.documents.deleteError'));
    } finally {
      setActionInProgress(null);
    }
  };

  // --- Reindex ---
  const handleReindex = async () => {
    if (!reindexTarget) return;
    const docId = reindexTarget.document_id;
    setReindexTarget(null);
    setActionInProgress(docId);
    setProgressOpen(true);
    setProgressDone(false);
    setProgressError(null);
    setProgressStage('');
    setProgressPercent(0);
    setProgressMessage('');

    try {
      const response = await fetch(
        `${API_BASE_URL}/admin/documents/${docId}/reindex`,
        {
          method: 'POST',
          credentials: 'include',
          headers: { 'Accept': 'text/event-stream' },
        },
      );

      if (!response.ok) throw new Error('Reindex request failed');

      await consumeSSE(
        response,
        (eventType, data) => {
          let parsed: Record<string, unknown>;
          try {
            parsed = JSON.parse(data);
          } catch {
            return;
          }
          if (eventType === 'progress') {
            setProgressStage(parsed.stage as string);
            setProgressPercent(parsed.percent as number);
            setProgressMessage(parsed.message as string);
          } else if (eventType === 'complete') {
            setProgressDone(true);
            setProgressMessage(t('admin.documents.reindexSuccess'));
          } else if (eventType === 'error') {
            setProgressError(parsed.message as string);
          }
        },
        () => {
          setActionInProgress(null);
          fetchDocuments();
        },
        (err) => {
          setProgressError(err);
          setActionInProgress(null);
        },
      );
    } catch {
      setProgressError(t('admin.documents.reindexError'));
      setActionInProgress(null);
    }
  };

  // --- Upload ---
  const handleFileSelected = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Reset input so the same file can be re-uploaded
    if (fileInputRef.current) fileInputRef.current.value = '';

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError(t('documentBrowser.pdfOnly'));
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError(t('documentBrowser.fileTooLarge'));
      return;
    }

    setSelectedFile(file);
    setUploadDialogOpen(true);
  };

  const handleUploadConfirm = async () => {
    if (!selectedFile) return;
    const file = selectedFile;
    const category = uploadCategory;

    setUploadDialogOpen(false);
    setSelectedFile(null);
    setActionInProgress('upload');
    setProgressOpen(true);
    setProgressDone(false);
    setProgressError(null);
    setProgressStage('uploading');
    setProgressPercent(0);
    setProgressMessage('Uploading...');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('category', category);

      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      if (response.status === 409) {
        setProgressError(t('documentBrowser.alreadyExists'));
        setActionInProgress(null);
        return;
      }
      if (!response.ok) throw new Error('Upload failed');

      await consumeSSE(
        response,
        (eventType, data) => {
          let parsed: Record<string, unknown>;
          try {
            parsed = JSON.parse(data);
          } catch {
            return;
          }
          if (eventType === 'progress') {
            setProgressStage(parsed.stage as string);
            setProgressPercent(parsed.percent as number);
            setProgressMessage(parsed.message as string);
          } else if (eventType === 'complete') {
            setProgressDone(true);
            setProgressMessage(t('admin.documents.uploadSuccess'));
          } else if (eventType === 'error') {
            setProgressError(parsed.message as string);
          }
        },
        () => {
          setActionInProgress(null);
          fetchDocuments();
        },
        (err) => {
          setProgressError(err);
          setActionInProgress(null);
        },
      );
    } catch {
      setProgressError(t('admin.documents.uploadError'));
      setActionInProgress(null);
    }
  };

  // --- Category change ---
  const handleCategoryChange = async (docId: string, newCategory: DocumentCategory) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/admin/documents/${docId}/category`,
        {
          method: 'PATCH',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ category: newCategory }),
        },
      );

      if (!response.ok) throw new Error('Category update failed');

      // Update local state
      setDocuments(prev =>
        prev.map(doc =>
          doc.document_id === docId ? { ...doc, category: newCategory } : doc
        )
      );
    } catch {
      setError(t('admin.documents.categoryUpdateError'));
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Box>
          <Typography variant="h4" gutterBottom>
            {t('admin.documents.title')}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            {t('admin.documents.subtitle')}
          </Typography>
        </Box>
        <Box display="flex" gap={1}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            hidden
            onChange={handleFileSelected}
          />
          <Button
            variant="contained"
            startIcon={<Upload size={18} />}
            onClick={() => fileInputRef.current?.click()}
            disabled={actionInProgress !== null}
          >
            {t('admin.documents.upload')}
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />}
            onClick={fetchDocuments}
            disabled={refreshing}
          >
            {t('admin.health.refresh')}
          </Button>
        </Box>
      </Box>

      {error && (
        <Box mb={2}>
          <Typography color="error">{t('common.error')}: {error}</Typography>
        </Box>
      )}

      {/* Document Table */}
      <Card>
        <CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}>
          {documents.length === 0 ? (
            <Box p={4} textAlign="center">
              <Typography color="textSecondary">{t('admin.documents.noDocuments')}</Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>{t('admin.documents.name')}</TableCell>
                    <TableCell>{t('admin.documents.category')}</TableCell>
                    <TableCell>{t('admin.documents.filename')}</TableCell>
                    <TableCell align="right">{t('admin.documents.size')}</TableCell>
                    <TableCell align="right">{t('admin.documents.pages')}</TableCell>
                    <TableCell>{t('admin.documents.created')}</TableCell>
                    <TableCell align="right">{t('admin.documents.actions')}</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {documents.map((doc) => (
                    <TableRow key={doc.document_id}>
                      <TableCell>
                        <Typography variant="body2" fontWeight={500}>
                          {doc.display_name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Select
                          size="small"
                          value={doc.category}
                          onChange={(e) => handleCategoryChange(doc.document_id, e.target.value as DocumentCategory)}
                          disabled={actionInProgress !== null}
                          sx={{ minWidth: 130, fontSize: '0.8rem' }}
                        >
                          <MenuItem value="documentation">
                            <Chip label={t('admin.documents.categoryDocumentation')} size="small" />
                          </MenuItem>
                          <MenuItem value="legislation">
                            <Chip label={t('admin.documents.categoryLegislation')} size="small" color="primary" />
                          </MenuItem>
                        </Select>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="textSecondary" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                          {doc.filename}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{formatBytes(doc.size_bytes)}</TableCell>
                      <TableCell align="right">
                        {doc.page_count > 0 ? doc.page_count : (
                          <Typography variant="body2" color="textSecondary">
                            {t('admin.documents.notIndexed')}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        {doc.created_at
                          ? new Date(doc.created_at).toLocaleDateString()
                          : '—'}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title={t('admin.documents.reindex')}>
                          <span>
                            <IconButton
                              size="small"
                              onClick={() => setReindexTarget(doc)}
                              disabled={actionInProgress !== null}
                            >
                              <RotateCcw size={16} />
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Tooltip title={t('admin.documents.delete')}>
                          <span>
                            <IconButton
                              size="small"
                              color="error"
                              onClick={() => setDeleteTarget(doc)}
                              disabled={actionInProgress !== null}
                            >
                              <Trash2 size={16} />
                            </IconButton>
                          </span>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Upload Category Dialog */}
      <Dialog
        open={uploadDialogOpen}
        onClose={() => { setUploadDialogOpen(false); setSelectedFile(null); }}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>{t('admin.documents.upload')}</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            {selectedFile?.name}
          </DialogContentText>
          <FormControl>
            <FormLabel>{t('admin.documents.selectCategory')}</FormLabel>
            <RadioGroup
              value={uploadCategory}
              onChange={(e) => setUploadCategory(e.target.value as DocumentCategory)}
            >
              <FormControlLabel
                value="documentation"
                control={<Radio />}
                label={t('admin.documents.categoryDocumentation')}
              />
              <FormControlLabel
                value="legislation"
                control={<Radio />}
                label={t('admin.documents.categoryLegislation')}
              />
            </RadioGroup>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setUploadDialogOpen(false); setSelectedFile(null); }}>
            {t('feedback.cancel')}
          </Button>
          <Button onClick={handleUploadConfirm} variant="contained">
            {t('admin.documents.upload')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteTarget !== null} onClose={() => setDeleteTarget(null)}>
        <DialogTitle>{t('admin.documents.delete')}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {t('admin.documents.deleteConfirm', { name: deleteTarget?.display_name })}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)}>{t('feedback.cancel')}</Button>
          <Button onClick={handleDelete} color="error" variant="contained">
            {t('admin.documents.delete')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Reindex Confirmation Dialog */}
      <Dialog open={reindexTarget !== null} onClose={() => setReindexTarget(null)}>
        <DialogTitle>{t('admin.documents.reindex')}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            {t('admin.documents.reindexConfirm', { name: reindexTarget?.display_name })}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReindexTarget(null)}>{t('feedback.cancel')}</Button>
          <Button onClick={handleReindex} color="primary" variant="contained">
            {t('admin.documents.reindex')}
          </Button>
        </DialogActions>
      </Dialog>

      {/* SSE Progress Dialog */}
      <Dialog open={progressOpen} maxWidth="sm" fullWidth>
        <DialogTitle>{t('admin.documents.indexingProgress')}</DialogTitle>
        <DialogContent>
          {progressError ? (
            <Typography color="error">{progressError}</Typography>
          ) : (
            <Box>
              <Typography variant="body2" gutterBottom>
                {stageLabel(progressStage)}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={progressPercent}
                sx={{ mb: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="caption" color="textSecondary">
                {progressMessage}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setProgressOpen(false)}
            disabled={!progressDone && !progressError}
          >
            {t('admin.documents.close')}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
