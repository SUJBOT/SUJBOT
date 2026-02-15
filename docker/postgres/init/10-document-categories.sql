-- Document category registry
-- Tracks document-level metadata (category) separately from page-level data (vl_pages).
-- Category is document-level, not page-level — a single row per document avoids
-- denormalization across 50+ pages.

CREATE TABLE IF NOT EXISTS vectors.documents (
    document_id TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT 'documentation'
        CHECK (category IN ('documentation', 'legislation')),
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON vectors.documents(category);
