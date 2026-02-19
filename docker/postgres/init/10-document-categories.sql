-- Document metadata registry
-- Tracks document-level metadata (category, access level, display name) separately
-- from page-level data (vl_pages). One row per document avoids denormalization
-- across 50+ pages.

CREATE TABLE IF NOT EXISTS vectors.documents (
    document_id TEXT PRIMARY KEY,
    category TEXT NOT NULL DEFAULT 'documentation'
        CHECK (category IN ('documentation', 'legislation')),
    access_level TEXT NOT NULL DEFAULT 'public'
        CHECK (access_level IN ('public', 'secret')),
    display_name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON vectors.documents(category);
