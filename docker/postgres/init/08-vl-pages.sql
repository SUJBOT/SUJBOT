-- VL (Vision-Language) page embeddings table
-- Stores Jina Embeddings v4 vectors (2048-dim) for PDF page images.
-- Used by VL retrieval pipeline as alternative to OCR-based text chunking.
--
-- No ANN index (HNSW/IVFFlat): with ~500 pages, exact cosine scan is <10ms
-- and gives perfect recall. Add an index only if dataset grows to 10k+ pages.

CREATE TABLE IF NOT EXISTS vectors.vl_pages (
    id SERIAL PRIMARY KEY,
    page_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    embedding vector(2048) NOT NULL,
    image_path TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vl_pages_document_id
    ON vectors.vl_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_vl_pages_page_number
    ON vectors.vl_pages(document_id, page_number);
