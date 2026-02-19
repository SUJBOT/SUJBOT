/**
 * PDFSidePanel Component
 *
 * Right-side panel for viewing PDF documents with text selection.
 * Renders all pages as a scrollable document with lazy loading.
 *
 * Features:
 * - Seamless multi-page scrolling with lazy loading
 * - Text selection for agent context
 * - Zoom controls
 * - Keyboard navigation (Escape to close)
 * - Mobile fullscreen overlay mode
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useMediaQuery } from '../../hooks/useMediaQuery';
import { Document, Page, pdfjs } from 'react-pdf';
import type { TextItem } from 'pdfjs-dist/types/src/display/api';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import {
  X,
  ZoomIn,
  ZoomOut,
  Loader2,
  AlertCircle,
  FileText,
  Search,
  ChevronUp,
  ChevronDown,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import type { TextSelection } from '../../types';

// Configure PDF.js worker
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

import { API_BASE_URL } from '../../config';

// Minimum selection length to trigger context capture
const MIN_SELECTION_LENGTH = 10;

interface PDFSidePanelProps {
  isOpen: boolean;
  documentId: string;
  documentName: string;
  initialPage?: number;
  chunkContent?: string;
  onClose: () => void;
  onTextSelected: (selection: TextSelection) => void;
}

/**
 * Extract significant phrases from chunk content for fuzzy matching.
 */
function extractSearchPhrases(content: string, maxPhrases: number = 15): string[] {
  if (!content) return [];
  const normalized = content.slice(0, 800).toLowerCase().replace(/\s+/g, ' ').trim();
  const words = normalized.split(' ').filter(w => w.length > 2);
  if (words.length < 4) return [];
  const phrases: string[] = [];
  for (let i = 0; i < words.length - 3 && phrases.length < maxPhrases; i++) {
    phrases.push(words.slice(i, i + 4).join(' '));
  }
  return phrases;
}

function textContainsPhrase(text: string, phrases: string[]): boolean {
  if (!text || phrases.length === 0) return false;
  const normalizedText = text.toLowerCase().replace(/\s+/g, ' ');
  return phrases.some(phrase => normalizedText.includes(phrase));
}

/**
 * Strip diacritics and lowercase for accent-insensitive search.
 * Handles:
 * 1. Standard Unicode combining marks (U+0300–U+036F)
 * 2. LaTeX-style spacing modifier letters (ˇ U+02C7, ˘˙˚˛˜˝ U+02D8–U+02DD)
 *    which appear in LaTeX-generated PDFs as separate characters before
 *    the base letter, often with whitespace: "ˇ c" instead of "č".
 *    We strip the modifier AND any adjacent whitespace to rejoin the word.
 */
const normalizeText = (text: string): string =>
  text
    .replace(/\s*[\u02c7\u02d8-\u02dd]\s*/g, '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();

/**
 * Memoized Page wrapper — only re-renders when pageNumber, scale, or
 * customTextRenderer change. Prevents all loaded pages from re-rendering
 * when search state or loadedPages set updates.
 */
const MemoizedPage = React.memo(function MemoizedPage({
  pageNumber,
  scale,
  onRenderSuccess,
  customTextRenderer,
}: {
  pageNumber: number;
  scale: number;
  onRenderSuccess: () => void;
  customTextRenderer?: (textItem: TextItem) => string;
}) {
  return (
    <Page
      pageNumber={pageNumber}
      scale={scale}
      renderTextLayer={true}
      renderAnnotationLayer={true}
      onRenderSuccess={onRenderSuccess}
      customTextRenderer={customTextRenderer}
      loading={
        <div className="flex items-center justify-center w-full h-96 bg-white">
          <Loader2 size={24} className="animate-spin text-accent-400" />
        </div>
      }
    />
  );
});

export function PDFSidePanel({
  isOpen,
  documentId,
  documentName,
  initialPage = 1,
  chunkContent,
  onClose,
  onTextSelected,
}: PDFSidePanelProps) {
  const { t } = useTranslation();
  const isMobile = useMediaQuery('(max-width: 767px)');
  const [numPages, setNumPages] = useState<number | null>(null);
  const [scale, setScale] = useState(1.0);
  const [error, setError] = useState<string | null>(null);
  const [_isLoading, setIsLoading] = useState(true);
  const [pdfData, setPdfData] = useState<ArrayBuffer | null>(null);
  const [currentVisiblePage, setCurrentVisiblePage] = useState(initialPage);
  const [loadedPages, setLoadedPages] = useState<Set<number>>(new Set());

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Array<{ pageNumber: number; matchIndex: number }>>([]);
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const searchInputRef = useRef<HTMLInputElement>(null);
  const pdfDocRef = useRef<any>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const observerRef = useRef<IntersectionObserver | null>(null);
  const scrolledToInitialRef = useRef(false);
  const renderedPagesRef = useRef(new Set<number>());

  // Text content cache: avoids re-extracting page text on every search
  const textCacheRef = useRef<Map<number, string>>(new Map());
  // Pending scroll target: resolved by handlePageRenderSuccess when page renders
  const pendingScrollRef = useRef<{ pageNumber: number; matchIndex: number } | null>(null);

  // Highlight phrases for chunk matching
  const highlightPhrases = useMemo(() => {
    if (!chunkContent) return [];
    return extractSearchPhrases(chunkContent);
  }, [chunkContent]);

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(searchQuery), 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Execute search when debounced query changes (uses cached text content)
  useEffect(() => {
    if (!debouncedQuery.trim() || !pdfDocRef.current || !numPages) {
      setSearchResults([]);
      setCurrentMatchIndex(0);
      return;
    }

    const query = normalizeText(debouncedQuery);
    let cancelled = false;

    (async () => {
      const results: Array<{ pageNumber: number; matchIndex: number }> = [];
      const cache = textCacheRef.current;

      for (let i = 1; i <= numPages; i++) {
        if (cancelled) return;

        let pageText = cache.get(i);
        if (pageText === undefined) {
          try {
            const page = await pdfDocRef.current.getPage(i);
            const textContent = await page.getTextContent();
            pageText = textContent.items.map((item: any) => item.str).join('');
            cache.set(i, pageText);
          } catch (err) {
            console.warn(`Failed to extract text from page ${i}:`, err);
            cache.set(i, '');
            continue;
          }
        }

        const normalizedPageText = normalizeText(pageText);
        let startIdx = 0;
        let matchIdx = 0;
        while ((startIdx = normalizedPageText.indexOf(query, startIdx)) !== -1) {
          results.push({ pageNumber: i, matchIndex: matchIdx++ });
          startIdx += query.length;
        }
      }
      if (!cancelled) {
        setSearchResults(results);
        setCurrentMatchIndex(0);
      }
    })();

    return () => { cancelled = true; };
  }, [debouncedQuery, numPages]);

  // Scroll helper: calculate offset relative to scroll container
  const scrollToTarget = useCallback((el: Element) => {
    const container = containerRef.current;
    if (!container) return;
    let offset = 0;
    let current: Element | null = el;
    while (current && current !== container) {
      offset += (current as HTMLElement).offsetTop;
      current = (current as HTMLElement).offsetParent;
    }
    container.scrollTo({ top: offset - container.clientHeight / 2, behavior: 'instant' });
  }, []);

  // Try to refine scroll to the exact highlight span within a rendered page.
  // Called immediately after render or when highlights are applied.
  const refineScrollToHighlight = useCallback((targetPage: number, matchOnPage: number) => {
    const el = pageRefs.current.get(targetPage);
    if (!el) return false;
    const highlights = el.querySelectorAll('mark.search-hl');
    if (highlights.length > 0) {
      const hl = highlights[Math.min(matchOnPage, highlights.length - 1)] ?? highlights[0];
      scrollToTarget(hl);
      return true;
    }
    return false;
  }, [scrollToTarget]);

  // Navigate to current match — scroll to the page, set pending scroll for render callback
  useEffect(() => {
    if (searchResults.length === 0 || currentMatchIndex >= searchResults.length) return;
    const targetPage = searchResults[currentMatchIndex].pageNumber;
    const matchOnPage = searchResults[currentMatchIndex].matchIndex;

    // Ensure the page is loaded
    setLoadedPages(prev => {
      const next = new Set(prev);
      next.add(targetPage);
      if (targetPage > 1) next.add(targetPage - 1);
      if (numPages && targetPage < numPages) next.add(targetPage + 1);
      return next;
    });

    // Phase 1: Immediately scroll to the page wrapper (rough positioning)
    const pageEl = pageRefs.current.get(targetPage);
    if (pageEl) {
      scrollToTarget(pageEl);
    }

    // Phase 2: If page is already rendered, try to refine to highlight immediately.
    // Otherwise, store pending scroll — handlePageRenderSuccess will pick it up.
    if (renderedPagesRef.current.has(targetPage)) {
      // Page already rendered; highlights may need a tick to be applied by the DOM effect
      requestAnimationFrame(() => {
        if (!refineScrollToHighlight(targetPage, matchOnPage)) {
          // Highlights not yet in DOM — store pending for the highlight effect
          pendingScrollRef.current = { pageNumber: targetPage, matchIndex: matchOnPage };
        }
      });
    } else {
      pendingScrollRef.current = { pageNumber: targetPage, matchIndex: matchOnPage };
    }
  }, [currentMatchIndex, searchResults, numPages, scrollToTarget, refineScrollToHighlight]);

  const handleSearchNext = useCallback(() => {
    if (searchResults.length === 0) return;
    setCurrentMatchIndex(prev => (prev + 1) % searchResults.length);
  }, [searchResults.length]);

  const handleSearchPrev = useCallback(() => {
    if (searchResults.length === 0) return;
    setCurrentMatchIndex(prev => (prev - 1 + searchResults.length) % searchResults.length);
  }, [searchResults.length]);

  const handleSearchKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) handleSearchPrev();
      else handleSearchNext();
    } else if (e.key === 'Escape') {
      setSearchOpen(false);
      setSearchQuery('');
    }
  }, [handleSearchNext, handleSearchPrev]);

  // PDF URL
  const pdfUrl = documentId
    ? `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/pdf`
    : '';

  // Fetch PDF with credentials
  useEffect(() => {
    if (!documentId || !isOpen) return;

    const fetchPdf = async () => {
      setIsLoading(true);
      setError(null);
      setPdfData(null);
      setLoadedPages(new Set());

      try {
        const response = await fetch(pdfUrl, {
          credentials: 'include',
        });

        if (!response.ok) {
          const errorMessages: Record<number, string> = {
            401: t('pdfPanel.error401', 'Session expired. Please log in again.'),
            403: t('pdfPanel.error403', 'You do not have permission to view this document.'),
            404: t('pdfPanel.error404', 'Document not found.'),
            500: t('pdfPanel.error500', 'Server temporarily unavailable.'),
          };
          const message = errorMessages[response.status] || t('pdfPanel.errorGeneric', `Error loading PDF (${response.status})`);
          throw new Error(message);
        }

        const arrayBuffer = await response.arrayBuffer();
        setPdfData(arrayBuffer);
      } catch (err) {
        const message = err instanceof Error ? err.message : t('pdfPanel.errorGeneric', 'Failed to load PDF');
        setError(message);
        setIsLoading(false);
      }
    };

    fetchPdf();
  }, [documentId, pdfUrl, isOpen, t]);

  // Reset state when document changes (full reload)
  useEffect(() => {
    setError(null);
    setIsLoading(true);
    setNumPages(null);
    setPdfData(null);
    setCurrentVisiblePage(initialPage);
    setLoadedPages(new Set());
    scrolledToInitialRef.current = false;
    renderedPagesRef.current = new Set();
    textCacheRef.current = new Map();
    pendingScrollRef.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId]);

  // Navigate to new page when initialPage changes within the same loaded document
  useEffect(() => {
    if (!numPages || !pdfData) return;

    setCurrentVisiblePage(initialPage);
    scrolledToInitialRef.current = false;

    // Ensure target page and neighbors are loaded
    setLoadedPages(prev => {
      const next = new Set(prev);
      for (let i = Math.max(1, initialPage - 1); i <= Math.min(numPages, initialPage + 2); i++) {
        next.add(i);
      }
      return next;
    });

    // If page is already rendered, scroll immediately
    const el = pageRefs.current.get(initialPage);
    if (el && renderedPagesRef.current.has(initialPage)) {
      requestAnimationFrame(() => {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        scrolledToInitialRef.current = true;
      });
    }
    // Otherwise handlePageRenderSuccess will scroll when the page renders
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPage]);

  // Handle document load success
  const onDocumentLoadSuccess = useCallback(
    (pdf: any) => {
      const pages = pdf.numPages;
      pdfDocRef.current = pdf;
      setNumPages(pages);
      setIsLoading(false);
      setError(null);
      // Pre-load initial page and neighbors
      const pagesToLoad = new Set<number>();
      for (let i = Math.max(1, initialPage - 1); i <= Math.min(pages, initialPage + 2); i++) {
        pagesToLoad.add(i);
      }
      setLoadedPages(pagesToLoad);
    },
    [initialPage]
  );

  // Handle document load error
  const onDocumentLoadError = useCallback((error: Error) => {
    setError(`Failed to load PDF: ${error.message}`);
    setIsLoading(false);
  }, []);

  // Setup IntersectionObserver for lazy loading pages
  useEffect(() => {
    if (!numPages || !containerRef.current) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          const pageNum = parseInt(entry.target.getAttribute('data-page') || '0');
          if (entry.isIntersecting && pageNum > 0) {
            // Load this page and neighbors
            setLoadedPages(prev => {
              const next = new Set(prev);
              for (let i = Math.max(1, pageNum - 1); i <= Math.min(numPages, pageNum + 2); i++) {
                next.add(i);
              }
              return next;
            });
            // Update current visible page (use the one with most visibility)
            if (entry.intersectionRatio > 0.5) {
              setCurrentVisiblePage(pageNum);
            }
          }
        });
      },
      {
        root: containerRef.current,
        rootMargin: '200px 0px',
        threshold: [0, 0.5, 1],
      }
    );

    // Observe all page placeholders
    pageRefs.current.forEach((ref, _pageNum) => {
      if (observerRef.current && ref) {
        observerRef.current.observe(ref);
      }
    });

    return () => {
      observerRef.current?.disconnect();
    };
  }, [numPages]);

  // Scroll compensation: when a page above the viewport renders, its height changes
  // from placeholder (800px) to actual size. Compensate scrollTop to prevent viewport shift.
  // Also resolves pending search-scroll targets.
  const handlePageRenderSuccess = useCallback((pageNum: number) => {
    if (renderedPagesRef.current.has(pageNum)) return;

    const el = pageRefs.current.get(pageNum);
    if (!el) return;

    const actualHeight = el.getBoundingClientRect().height;
    renderedPagesRef.current.add(pageNum);

    // Scroll to initial page when it first renders
    if (pageNum === initialPage && !scrolledToInitialRef.current) {
      scrolledToInitialRef.current = true;
      requestAnimationFrame(() => {
        el.scrollIntoView({ behavior: 'instant', block: 'start' });
      });
      return;
    }

    // Compensate scroll for pages that rendered above the viewport
    if (containerRef.current && scrolledToInitialRef.current) {
      const container = containerRef.current;
      const pageRect = el.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();

      if (pageRect.bottom < containerRect.top) {
        // Page is above viewport — compensate for height change from placeholder
        const heightDiff = actualHeight - 800;
        if (Math.abs(heightDiff) > 5) {
          container.scrollTop += heightDiff;
        }
      }
    }

    // Resolve pending search scroll — page just rendered, scroll to it
    const pending = pendingScrollRef.current;
    if (pending && pending.pageNumber === pageNum) {
      pendingScrollRef.current = null;
      // Scroll to page first, then refine after highlights are applied
      scrollToTarget(el);
    }
  }, [initialPage, scrollToTarget]);

  // Text selection handler
  const handleSelectionChange = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const text = selection.toString().trim();
    if (text.length < MIN_SELECTION_LENGTH) return;

    // Check if selection is within this PDF panel
    const range = selection.getRangeAt(0);
    const container = containerRef.current;
    if (!container || !container.contains(range.commonAncestorContainer)) return;

    // Extract page numbers from selection
    const getPageFromNode = (node: Node): number => {
      let element = node.nodeType === Node.TEXT_NODE ? node.parentElement : node as Element;
      while (element && !element.getAttribute('data-page')) {
        element = element.parentElement;
      }
      return parseInt(element?.getAttribute('data-page') || '1');
    };

    const pageStart = getPageFromNode(range.startContainer);
    const pageEnd = getPageFromNode(range.endContainer);

    onTextSelected({
      text,
      documentId,
      documentName,
      pageStart,
      pageEnd,
      charCount: text.length,
    });
  }, [documentId, documentName, onTextSelected]);

  // Listen for selection changes
  useEffect(() => {
    if (!isOpen) return;
    document.addEventListener('mouseup', handleSelectionChange);
    return () => document.removeEventListener('mouseup', handleSelectionChange);
  }, [isOpen, handleSelectionChange]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+F / Cmd+F to open search
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        setSearchOpen(true);
        setTimeout(() => searchInputRef.current?.focus(), 50);
        return;
      }
      switch (e.key) {
        case 'Escape':
          if (searchOpen) {
            setSearchOpen(false);
            setSearchQuery('');
          } else {
            onClose();
          }
          break;
        case '+':
        case '=':
          setScale(s => Math.min(3, s + 0.2));
          break;
        case '-':
          setScale(s => Math.max(0.5, s - 0.2));
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, searchOpen]);

  // Custom text renderer for chunk highlighting only
  const customTextRenderer = useCallback(
    (textItem: TextItem) => {
      const str = textItem.str;
      if (!str.trim()) return str;

      if (highlightPhrases.length > 0 && textContainsPhrase(str, highlightPhrases)) {
        return `<mark class="chunk-highlight">${str}</mark>`;
      }

      return str;
    },
    [highlightPhrases]
  );

  // DOM-based search highlighting: wraps only the matched character range in <mark>.
  // Works across TextItem boundaries (needed for LaTeX PDFs where diacritics are
  // separate characters, splitting words across multiple spans).
  useEffect(() => {
    // Remove all previous highlight <mark> wrappers — unwrap their text back into parent
    containerRef.current?.querySelectorAll('mark.search-hl').forEach(mark => {
      const parent = mark.parentNode;
      if (!parent) return;
      while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
      parent.removeChild(mark);
      parent.normalize(); // merge adjacent text nodes
    });

    if (!debouncedQuery.trim() || searchResults.length === 0) return;

    const query = normalizeText(debouncedQuery);
    const matchPages = new Set(searchResults.map(r => r.pageNumber));

    const applyHighlights = () => {
      matchPages.forEach(pageNum => {
        const pageEl = pageRefs.current.get(pageNum);
        if (!pageEl) return;

        const textLayer = pageEl.querySelector('.react-pdf__Page__textContent');
        if (!textLayer) return;

        // Get all text spans (react-pdf renders each TextItem as a <span>)
        const spans = Array.from(textLayer.querySelectorAll('span'));
        if (spans.length === 0) return;

        // Build concatenated text and track span boundaries
        let fullText = '';
        const spanRanges: Array<{ span: Element; start: number; end: number }> = [];
        for (const span of spans) {
          const text = span.textContent || '';
          spanRanges.push({ span, start: fullText.length, end: fullText.length + text.length });
          fullText += text;
        }

        // Normalize with position mapping (original index for each normalized char)
        const normChars: string[] = [];
        const normToOrig: number[] = [];
        let i = 0;
        while (i < fullText.length) {
          const code = fullText.charCodeAt(i);

          // Spacing modifier (ˇ˘˙˚˛˜˝): skip it + surrounding whitespace
          if (code >= 0x02c7 && code <= 0x02dd) {
            // Remove preceding whitespace from output
            while (normChars.length > 0 && normChars[normChars.length - 1] === ' ') {
              normChars.pop();
              normToOrig.pop();
            }
            i++; // skip modifier
            while (i < fullText.length && /\s/.test(fullText[i])) i++; // skip trailing whitespace
            continue;
          }

          // NFD decompose, strip combining marks, lowercase
          const decomposed = fullText[i].normalize('NFD');
          for (const ch of decomposed) {
            if (ch.charCodeAt(0) >= 0x0300 && ch.charCodeAt(0) <= 0x036f) continue;
            normChars.push(ch.toLowerCase());
            normToOrig.push(i);
          }
          i++;
        }

        const normalizedFull = normChars.join('');

        // Collect match ranges in original text coordinates (process in reverse
        // so that DOM mutations don't shift positions of earlier matches)
        const matchRanges: Array<{ origStart: number; origEnd: number }> = [];
        let searchIdx = 0;
        while ((searchIdx = normalizedFull.indexOf(query, searchIdx)) !== -1) {
          const origStart = normToOrig[searchIdx];
          const origEnd = normToOrig[Math.min(searchIdx + query.length - 1, normToOrig.length - 1)] + 1;
          matchRanges.push({ origStart, origEnd });
          searchIdx += query.length;
        }

        // Wrap only the matched character range within each overlapping span.
        // Process matches in reverse so earlier DOM positions stay valid.
        for (let m = matchRanges.length - 1; m >= 0; m--) {
          const { origStart, origEnd } = matchRanges[m];

          for (const { span, start: spanStart, end: spanEnd } of spanRanges) {
            if (spanStart >= origEnd || spanEnd <= origStart) continue;

            // Character range within this span's text
            const hlStart = Math.max(0, origStart - spanStart);
            const hlEnd = Math.min(spanEnd - spanStart, origEnd - spanStart);

            const textNode = span.firstChild;
            if (!textNode || textNode.nodeType !== Node.TEXT_NODE) continue;
            const text = textNode.textContent || '';
            if (hlStart >= text.length) continue;

            // Split: [before][match][after]
            const before = text.slice(0, hlStart);
            const match = text.slice(hlStart, hlEnd);
            const after = text.slice(hlEnd);

            // Build fragment: before text + <mark> + after text
            const frag = document.createDocumentFragment();
            if (before) frag.appendChild(document.createTextNode(before));
            const mark = document.createElement('mark');
            mark.className = 'search-hl';
            mark.textContent = match;
            frag.appendChild(mark);
            if (after) frag.appendChild(document.createTextNode(after));

            span.replaceChild(frag, textNode);
          }
        }
      });
    };

    // Delay to ensure text layer has rendered, then resolve any pending scroll
    const timer = setTimeout(() => {
      applyHighlights();
      // After highlights are in the DOM, resolve any pending scroll refinement
      const pending = pendingScrollRef.current;
      if (pending) {
        pendingScrollRef.current = null;
        refineScrollToHighlight(pending.pageNumber, pending.matchIndex);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [debouncedQuery, searchResults, loadedPages, refineScrollToHighlight]);

  if (!isOpen) return null;

  return (
    <div
      className={cn(
        'flex flex-col',
        'bg-white dark:bg-accent-900',
        'border-l border-accent-200 dark:border-accent-700',
        // Mobile: fullscreen overlay
        isMobile && 'fixed inset-0 z-50',
        // Desktop: side panel (width controlled by parent)
        !isMobile && 'h-full'
      )}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between',
          'px-4 py-3',
          'bg-accent-50 dark:bg-accent-800',
          'border-b border-accent-200 dark:border-accent-700',
          'flex-shrink-0'
        )}
      >
        {/* Document info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <FileText size={20} className="text-accent-500 dark:text-accent-400 flex-shrink-0" />
          <h2 className="text-sm font-semibold text-accent-900 dark:text-accent-100 truncate">
            {documentName.replace(/_/g, ' ')}
          </h2>
        </div>

        {/* Page indicator */}
        {numPages && (
          <div className="text-xs text-accent-500 dark:text-accent-400 mx-4">
            {t('pdfPanel.page', 'Page')} {currentVisiblePage} / {numPages}
          </div>
        )}

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setScale(s => Math.max(0.5, s - 0.2))}
            disabled={scale <= 0.5}
            className={cn(
              'p-1.5 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title={t('pdfPanel.zoomOut', 'Zoom out (-)')}
          >
            <ZoomOut size={16} />
          </button>

          <span className="text-xs text-accent-600 dark:text-accent-400 w-12 text-center">
            {Math.round(scale * 100)}%
          </span>

          <button
            onClick={() => setScale(s => Math.min(3, s + 0.2))}
            disabled={scale >= 3}
            className={cn(
              'p-1.5 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
            title={t('pdfPanel.zoomIn', 'Zoom in (+)')}
          >
            <ZoomIn size={16} />
          </button>
        </div>

        {/* Search button */}
        <button
          onClick={() => {
            setSearchOpen(prev => !prev);
            if (!searchOpen) setTimeout(() => searchInputRef.current?.focus(), 50);
          }}
          className={cn(
            'p-1.5 rounded transition-colors ml-2',
            'hover:bg-accent-200 dark:hover:bg-accent-700',
            searchOpen
              ? 'text-blue-500 dark:text-blue-400'
              : 'text-accent-500 dark:text-accent-400'
          )}
          title={t('pdfPanel.search', 'Search in document')}
        >
          <Search size={16} />
        </button>

        {/* Close button */}
        <button
          onClick={onClose}
          className={cn(
            'p-1.5 rounded transition-colors ml-1',
            'hover:bg-accent-200 dark:hover:bg-accent-700',
            'text-accent-500 dark:text-accent-400'
          )}
          title={t('pdfPanel.close', 'Close (Esc)')}
        >
          <X size={18} />
        </button>
      </div>

      {/* Search bar (collapsible) */}
      {searchOpen && (
        <div className={cn(
          'flex items-center gap-2 px-3 py-2',
          'bg-white dark:bg-accent-800',
          'border-b border-accent-200 dark:border-accent-700',
          'flex-shrink-0'
        )}>
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            placeholder={t('pdfPanel.searchPlaceholder', 'Search...')}
            className={cn(
              'flex-1 px-2 py-1 text-sm rounded',
              'bg-accent-50 dark:bg-accent-900',
              'border border-accent-200 dark:border-accent-700',
              'text-accent-900 dark:text-accent-100',
              'placeholder:text-accent-400',
              'focus:outline-none focus:ring-1 focus:ring-blue-400'
            )}
          />
          <span className="text-xs text-accent-500 dark:text-accent-400 whitespace-nowrap min-w-[3rem] text-center">
            {searchResults.length > 0
              ? `${currentMatchIndex + 1} / ${searchResults.length}`
              : debouncedQuery.trim()
                ? t('pdfPanel.noResults', 'No results')
                : ''
            }
          </span>
          <button
            onClick={handleSearchPrev}
            disabled={searchResults.length === 0}
            className={cn(
              'p-1 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-30'
            )}
          >
            <ChevronUp size={14} />
          </button>
          <button
            onClick={handleSearchNext}
            disabled={searchResults.length === 0}
            className={cn(
              'p-1 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'disabled:opacity-30'
            )}
          >
            <ChevronDown size={14} />
          </button>
          <button
            onClick={() => { setSearchOpen(false); setSearchQuery(''); }}
            className={cn(
              'p-1 rounded transition-colors',
              'hover:bg-accent-200 dark:hover:bg-accent-700',
              'text-accent-500 dark:text-accent-400'
            )}
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* PDF content - scrollable */}
      <div
        ref={containerRef}
        className={cn(
          'flex-1 overflow-auto',
          'bg-accent-100 dark:bg-accent-950'
        )}
      >
        {error ? (
          <div className="flex flex-col items-center justify-center gap-4 p-8 text-red-500">
            <AlertCircle size={48} />
            <p className="text-lg font-medium text-center">{error}</p>
          </div>
        ) : !pdfData ? (
          <div className="flex items-center justify-center gap-3 p-8 text-accent-500">
            <Loader2 size={24} className="animate-spin" />
            <span>{t('pdfPanel.loading', 'Loading PDF...')}</span>
          </div>
        ) : (
          <Document
            file={pdfData}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={onDocumentLoadError}
            loading={
              <div className="flex items-center justify-center gap-3 p-8 text-accent-500">
                <Loader2 size={24} className="animate-spin" />
                <span>{t('pdfPanel.rendering', 'Rendering...')}</span>
              </div>
            }
            className="flex flex-col items-center py-4 gap-4"
          >
            {numPages && Array.from({ length: numPages }, (_, i) => i + 1).map(pageNum => (
              <div
                key={pageNum}
                ref={(el) => {
                  if (el) {
                    pageRefs.current.set(pageNum, el);
                    observerRef.current?.observe(el);
                  }
                }}
                data-page={pageNum}
                className="shadow-lg"
                style={{ minHeight: loadedPages.has(pageNum) ? undefined : '800px' }}
              >
                {loadedPages.has(pageNum) ? (
                  <MemoizedPage
                    pageNumber={pageNum}
                    scale={scale}
                    onRenderSuccess={() => handlePageRenderSuccess(pageNum)}
                    customTextRenderer={
                      pageNum === initialPage && highlightPhrases.length > 0
                        ? customTextRenderer
                        : undefined
                    }
                  />
                ) : (
                  <div className="flex items-center justify-center w-full h-[800px] bg-white dark:bg-accent-800">
                    <span className="text-accent-400 text-sm">
                      {t('pdfPanel.page', 'Page')} {pageNum}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </Document>
        )}
      </div>
    </div>
  );
}
