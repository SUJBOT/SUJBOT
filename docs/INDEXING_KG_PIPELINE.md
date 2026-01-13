# SUJBOT Indexing & Knowledge Graph Pipeline

**Last Updated:** 2026-01-13
**Version:** PHASE 1-5 Complete

This document provides comprehensive technical documentation for the SUJBOT indexing pipeline, covering PDF extraction through knowledge graph construction and retrieval.

---

## Table of Contents

1. [Pipeline Architecture Overview](#1-pipeline-architecture-overview)
2. [Phase 1: PDF Extraction](#2-phase-1-pdf-extraction)
3. [Phase 2: Summary Generation](#3-phase-2-summary-generation)
4. [Phase 3: Multi-Layer Chunking](#4-phase-3-multi-layer-chunking)
5. [Phase 4: Embedding & Storage](#5-phase-4-embedding--storage)
6. [Phase 5A: Knowledge Graph](#6-phase-5a-knowledge-graph)
7. [Phase 5C: Retrieval](#7-phase-5c-retrieval)
8. [Data Structures Reference](#8-data-structures-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Complete Pipeline Example](#10-complete-pipeline-example)
11. [Research Basis](#11-research-basis)

---

## 1. Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SUJBOT INDEXING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────┐  │
│  │  PHASE 1 │    │  PHASE 2 │    │  PHASE 3 │    │  PHASE 4 │    │ 5A/B │  │
│  │          │───▶│          │───▶│          │───▶│          │───▶│      │  │
│  │   PDF    │    │ SUMMARY  │    │ CHUNKING │    │ EMBEDDING│    │  KG  │  │
│  │EXTRACTION│    │GENERATION│    │   SAC    │    │ STORAGE  │    │      │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────┘  │
│       │               │               │               │              │      │
│       ▼               ▼               ▼               ▼              ▼      │
│   Gemini 2.5      Claude/GPT      Multi-Layer     Qwen3-8B       Graphiti  │
│   Flash          Batch API         Tokens        pgvector         Neo4j    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                               OUTPUT FILES                                  │
│  phase1_extraction.json → phase2_summaries.json → phase3_chunks.json        │
│                                    ↓                                        │
│                         PostgreSQL vectors schema                           │
│                         Neo4j knowledge graph                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| PDF Extraction | Gemini 2.5 Flash | Direct PDF upload, structured JSON output |
| Summaries | Claude Sonnet 4.5 / GPT-4o-mini | Hierarchical section summarization |
| Chunking | tiktoken / HuggingFace tokenizers | Token-aware boundary detection |
| Embeddings | Qwen/Qwen3-Embedding-8B (4096 dims) | Semantic vector representations |
| Vector Storage | PostgreSQL + pgvector | HNSW indexes, ACID transactions |
| Knowledge Graph | Graphiti + Neo4j | Temporal entity/relationship storage |
| Retrieval | HyDE + Expansion Fusion | 4-signal weighted fusion |

### Design Principles

1. **Token-Aware Processing**: All chunking uses tokenizers aligned with embedding models
2. **Hierarchical Summaries**: Section → Document (no raw document summarization)
3. **Multi-Layer Embeddings**: 3 separate indexes for different retrieval granularities
4. **Summary-Augmented Chunking (SAC)**: Context prepended during embedding
5. **Research-Backed Decisions**: All parameters based on published research

---

## 2. Phase 1: PDF Extraction

**File:** `src/gemini_extractor.py` (1445 lines)

### Overview

Phase 1 extracts structured content from PDF documents using Google's Gemini 2.5 Flash model. The extractor uses direct PDF upload via the File API (NOT Base64 encoding) for optimal performance with Gemini's 1M token context window.

### Key Components

#### GeminiExtractionConfig (`src/gemini_extractor.py:468-512`)

```python
@dataclass
class GeminiExtractionConfig:
    """Configuration for Gemini-based PDF extraction."""

    model: str = "gemini-2.5-flash"  # Current best for document extraction
    temperature: float = 0.0  # Deterministic output for reproducibility
    max_output_tokens: int = 65536  # Maximum JSON response size
    chunk_size_pages: int = 100  # Pages per extraction chunk
    overlap_pages: int = 5  # Page overlap for chunked extraction
    retry_attempts: int = 3  # Retries for API failures
    request_timeout: int = 600  # 10-minute timeout for large PDFs
```

#### JSON Repair Strategies (`src/gemini_extractor.py:49-125`)

The extractor implements SOTA JSON repair for truncated LLM responses:

```python
def repair_truncated_json(text: str) -> str:
    """
    Repair truncated JSON from LLM output.

    Strategies (in order of application):
    1. Strip markdown code blocks (```json ... ```)
    2. Close unclosed strings (missing ")
    3. Close unclosed arrays (missing ])
    4. Close unclosed objects (missing })
    5. Remove trailing commas before closers
    6. Validate with json.loads()
    """
```

**Why this matters:** LLM responses often exceed token limits, causing JSON truncation. These repair strategies recover ~95% of truncated responses.

### Extraction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PDF EXTRACTION FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FILE UPLOAD                                                 │
│     └── Upload PDF to Gemini File API                           │
│     └── Receive file_uri for reference                          │
│                                                                 │
│  2. SIZE CHECK                                                  │
│     └── If PDF > 10MB or > 100 pages: CHUNKED extraction        │
│     └── Else: SINGLE-PASS extraction                            │
│                                                                 │
│  3. EXTRACTION                                                  │
│     └── Gemini processes PDF with structured output prompt      │
│     └── Returns JSON with sections, content, metadata           │
│     └── response_mime_type="application/json"                   │
│                                                                 │
│  4. REPAIR & VALIDATION                                         │
│     └── Apply JSON repair if truncated                          │
│     └── Parse into ExtractedDocument dataclass                  │
│     └── Interpolate page numbers (if missing)                   │
│                                                                 │
│  5. OUTPUT                                                      │
│     └── phase1_extraction.json                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Chunked Extraction for Large PDFs

For PDFs exceeding size limits, the extractor processes pages in chunks:

```python
def extract_chunked(self, pdf_path: Path, chunk_size: int = 100) -> ExtractedDocument:
    """
    Extract large PDF in chunks with overlap.

    Process:
    1. Split PDF into page ranges: [1-100], [96-200], [196-300], ...
    2. Extract each chunk separately
    3. Merge results with deduplication
    4. _deduplicate_sections_by_path() merges TOC vs content sections
    """
```

**Section Deduplication:** When Gemini extracts TOC pages first, it creates stub sections (title only). Later chunks create full sections (title + content). The deduplication logic (`_deduplicate_sections_by_path()`) merges these, keeping the version with the longest content.

### Page Number Interpolation (`src/gemini_extractor.py:1323-1381`)

When Gemini doesn't return page numbers, they're interpolated from character positions:

```python
def _interpolate_page_numbers(
    self,
    doc: ExtractedDocument,
    total_chars: int,
    total_pages: int
) -> None:
    """
    Estimate page numbers from character positions.

    Assumes uniform character distribution across pages.
    Formula: page = (char_start / total_chars) * total_pages + 1
    """
```

### Output Format

Phase 1 produces `phase1_extraction.json`:

```json
{
  "document_id": "BZ_VR1",
  "title": "Bezpečnostní zpráva jaderného zařízení VR-1",
  "full_text": "...",
  "document_summary": null,
  "hierarchy_depth": 4,
  "sections": [
    {
      "section_id": "sec_1",
      "title": "1. Úvod",
      "path": "1. Úvod",
      "level": 1,
      "depth": 1,
      "content": "Tato bezpečnostní zpráva...",
      "summary": null,
      "page_number": 5,
      "char_start": 0,
      "char_end": 2500
    }
  ],
  "metadata": {
    "extraction_backend": "gemini",
    "model": "gemini-2.5-flash",
    "total_pages": 150,
    "extraction_time_seconds": 45.2
  }
}
```

---

## 3. Phase 2: Summary Generation

**File:** `src/summary_generator.py` (941 lines)

### Overview

Phase 2 generates hierarchical summaries using a two-stage approach based on Reuter et al. (2024) research:

1. **Section Summaries**: Each section gets a 150-character summary
2. **Document Summary**: Aggregated from section summaries (100-1000 chars)

**Critical:** Generic summaries outperform expert-guided summaries for retrieval (counterintuitive but research-proven).

### Key Components

#### SummarizationConfig

```python
@dataclass
class SummarizationConfig:
    """Configuration for summary generation."""

    model: str = "claude-sonnet-4-5"  # Primary model
    fallback_model: str = "gpt-4o-mini"  # Cheaper fallback
    section_summary_length: int = 150  # Characters per section
    document_summary_min: int = 100  # Minimum doc summary
    document_summary_max: int = 1000  # Maximum doc summary
    batch_size: int = 10  # Sections per batch
    use_openai_batch_api: bool = True  # 50% cost savings
```

### SummaryGenerator Class

```python
class SummaryGenerator:
    """
    Generates hierarchical summaries for RAG.

    Research basis: Reuter et al. (2024) - Summary-Augmented Chunking
    Key finding: Generic summaries improve retrieval vs expert-guided

    Architecture:
    1. Section summaries (parallel batching for 10-15× speedup)
    2. Document summary (aggregated from section summaries)

    Never passes full document text to LLM - uses hierarchical approach.
    """
```

### Summary Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  HIERARCHICAL SUMMARIZATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STAGE 1: SECTION SUMMARIES                                     │
│                                                                 │
│  Section 1 ──┐                                                  │
│  Section 2 ──┼── Batch 1 ──► LLM ──► 150-char summaries        │
│  ...        ─┤                                                  │
│  Section 10 ─┘                                                  │
│                                                                 │
│  Section 11 ─┐                                                  │
│  Section 12 ─┼── Batch 2 ──► LLM ──► 150-char summaries        │
│  ...        ─┤                                                  │
│  Section 20 ─┘                                                  │
│                                                                 │
│  STAGE 2: DOCUMENT SUMMARY                                      │
│                                                                 │
│  All Section     Aggregate     Document Summary                 │
│  Summaries  ────────────────►  (100-1000 chars)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Prompt Loading

Prompts are loaded from `prompts/` directory (SSOT principle):

```python
# prompts/section_summary.txt
# prompts/document_summary.txt

def load_prompt(prompt_name: str) -> str:
    """Load prompt from prompts/ directory."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    return (prompts_dir / f"{prompt_name}.txt").read_text()
```

### OpenAI Batch API Support

For large documents, the generator uses OpenAI's Batch API for 50% cost savings:

```python
async def generate_summaries_batch_api(
    self,
    sections: List[DocumentSection]
) -> List[str]:
    """
    Generate summaries using OpenAI Batch API.

    Benefits:
    - 50% cost reduction
    - Higher rate limits
    - Better for >50 sections

    Tradeoff:
    - Asynchronous (up to 24h turnaround)
    - Not suitable for real-time
    """
```

### Output Format

Phase 2 updates the document with summaries:

```json
{
  "document_id": "BZ_VR1",
  "document_summary": "Bezpečnostní zpráva VR-1 popisuje konstrukci, provoz a bezpečnostní opatření výukového reaktoru ČVUT. Dokument zahrnuje technické specifikace, limity a podmínky provozu, radiační ochranu a havarijní postupy.",
  "sections": [
    {
      "section_id": "sec_1",
      "title": "1. Úvod",
      "summary": "Úvodní kapitola definuje účel bezpečnostní zprávy a základní údaje o jaderném zařízení VR-1."
    }
  ]
}
```

---

## 4. Phase 3: Multi-Layer Chunking

**File:** `src/multi_layer_chunker.py` (1048 lines)

### Overview

Phase 3 creates multi-layer chunks with Summary-Augmented Chunking (SAC):

- **Layer 1**: Document level (1 chunk per document)
- **Layer 2**: Section level (1 chunk per section)
- **Layer 3**: Chunk level (512 tokens, sentence-aware)

### Key Components

#### ChunkMetadata (`src/multi_layer_chunker.py:49-95`)

```python
@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""

    chunk_id: str
    layer: int  # 1=document, 2=section, 3=chunk
    document_id: str
    section_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    # Position info
    page_number: int = 0
    char_start: int = 0
    char_end: int = 0

    # Document info (Layer 1)
    title: Optional[str] = None

    # Hierarchy context (Layer 2-3)
    section_title: Optional[str] = None
    section_path: Optional[str] = None  # Breadcrumb path
    section_level: int = 0
    section_depth: int = 0

    # Semantic clustering (PHASE 4.5)
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    cluster_confidence: Optional[float] = None
```

#### Chunk Dataclass (`src/multi_layer_chunker.py:97-177`)

```python
@dataclass
class Chunk:
    """
    A single chunk with content and metadata.

    For Layer 3 (chunk level), content includes SAC summary prepended.
    For embedding, use 'content'.
    For generation, use 'raw_content' (without SAC summary).
    """

    chunk_id: str
    content: str  # For embedding (with SAC if Layer 3)
    raw_content: str  # For generation (without SAC)
    metadata: ChunkMetadata

    def to_dict(self) -> Dict:
        """
        Serialize chunk to JSON format.

        Returns:
            {
                "chunk_id": "doc_L3_c1_sec_1",
                "context": "SAC context summary",
                "raw_content": "Actual text content",
                "embedding_text": "[breadcrumb]\n\ncontext\n\nraw_content",
                "metadata": {...}
            }
        """
```

### Token-Aware Splitting (`src/multi_layer_chunker.py:528-597`)

```python
def _token_aware_split(self, text: str, max_tokens: int = 512) -> List[str]:
    """
    Split text into token-aware chunks with sentence boundaries.

    IMPROVED: Splits on sentence boundaries first, then groups sentences
    into chunks respecting max_tokens. Prevents mid-sentence splits.

    Process:
    1. Split into sentences (NLTK Czech/English)
    2. Group sentences into chunks <= max_tokens
    3. Handle oversized sentences with token-level split (fallback)

    Uses:
    - tiktoken for OpenAI models
    - HuggingFace AutoTokenizer for Qwen/BGE models
    """
```

### Contextual Retrieval Augmentation

When enabled, chunks receive LLM-generated context:

```python
def _apply_contextual_augmentation_to_hybrid_chunks(
    self,
    chunks: List[Chunk],
    extracted_doc
) -> List[Chunk]:
    """
    Apply Contextual Retrieval augmentation.

    Research: Anthropic, 2024 - 67% reduction in retrieval failures

    Process:
    1. For each chunk, generate context describing what it contains
    2. Prepend context to chunk content for embedding
    3. Store raw_content separately for generation
    """
```

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MULTI-LAYER CHUNKING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LAYER 1 (Document Level)                                       │
│  └── 1 chunk per document                                       │
│  └── Contains: document_summary                                 │
│  └── Purpose: Document identification, global filtering         │
│  └── ID format: {doc_id}_L1                                     │
│                                                                 │
│  LAYER 2 (Section Level)                                        │
│  └── 1 chunk per section                                        │
│  └── Contains: section_summary (NOT full text)                  │
│  └── Purpose: Section-level semantic search                     │
│  └── ID format: {doc_id}_L2_{section_id}                        │
│                                                                 │
│  LAYER 3 (Chunk Level)                                          │
│  └── Multiple chunks per section (512 tokens each)              │
│  └── Contains: SAC context + raw_content                        │
│  └── Purpose: Fine-grained retrieval (PRIMARY)                  │
│  └── ID format: {doc_id}_L3_{counter}                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### SAC (Summary-Augmented Chunking) Format

```
┌─────────────────────────────────────────────────────────────────┐
│                   CHUNK EMBEDDING FORMAT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  embedding_text = "[breadcrumb]\n\ncontext\n\nraw_content"      │
│                                                                 │
│  Example:                                                       │
│  ──────────────────────────────────────────────────────         │
│  [BZ_VR1 > 3. Popis zařízení > 3.2 Aktivní zóna]               │
│                                                                 │
│  Tato sekce popisuje konstrukci aktivní zóny reaktoru VR-1,     │
│  včetně palivových článků a moderátoru.                         │
│                                                                 │
│  Aktivní zóna reaktoru VR-1 obsahuje 16 palivových článků       │
│  typu IRT-4M s obohacením 19,75% U-235. Moderátorem a           │
│  chladivem je demineralizovaná voda...                          │
│  ──────────────────────────────────────────────────────         │
│                                                                 │
│  Components:                                                    │
│  • [breadcrumb]: Document + section hierarchy path              │
│  • context: SAC summary (LLM-generated description)             │
│  • raw_content: Actual document text                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Flat Document Optimization

For documents without hierarchy (depth=1), only Layer 3 is created:

```python
def chunk_document(self, extracted_doc) -> Dict[str, List[Chunk]]:
    """
    OPTIMIZATION: For flat documents (depth=1):
    - Only Layer 3 created
    - Layers 1 and 2 skipped (empty)
    - Saves embedding costs and storage
    """
```

### Output Format (phase3_chunks.json)

```json
{
  "layer1": [
    {
      "chunk_id": "BZ_VR1_L1",
      "context": "",
      "raw_content": "Bezpečnostní zpráva VR-1 popisuje...",
      "embedding_text": "[BZ_VR1]\n\nBezpečnostní zpráva VR-1 popisuje...",
      "metadata": {
        "chunk_id": "BZ_VR1_L1",
        "layer": 1,
        "document_id": "BZ_VR1",
        "title": "Bezpečnostní zpráva VR-1"
      }
    }
  ],
  "layer2": [...],
  "layer3": [
    {
      "chunk_id": "BZ_VR1_L3_1",
      "context": "Úvodní část bezpečnostní zprávy definující účel dokumentu.",
      "raw_content": "Tato bezpečnostní zpráva je základním bezpečnostním dokumentem...",
      "embedding_text": "[BZ_VR1 > 1. Úvod]\n\nÚvodní část bezpečnostní zprávy definující účel dokumentu.\n\nTato bezpečnostní zpráva je základním bezpečnostním dokumentem...",
      "metadata": {
        "chunk_id": "BZ_VR1_L3_1",
        "layer": 3,
        "document_id": "BZ_VR1",
        "section_id": "sec_1",
        "section_title": "1. Úvod",
        "section_path": "1. Úvod",
        "page_number": 5
      }
    }
  ],
  "total_chunks": 285
}
```

---

## 5. Phase 4: Embedding & Storage

### 5.1 Embedding Generation

**File:** `src/embedding_generator.py` (624 lines)

#### EmbeddingGenerator Class

```python
class EmbeddingGenerator:
    """
    Generate embeddings for chunks using multiple providers.

    Supported providers:
    - DeepInfra: Qwen/Qwen3-Embedding-8B (4096 dims) - RECOMMENDED
    - Voyage AI: voyage-3 (1024 dims)
    - OpenAI: text-embedding-3-large (3072 dims)
    - HuggingFace: Local models

    Current default: Qwen/Qwen3-Embedding-8B via DeepInfra
    - 4096 dimensions
    - Excellent multilingual support (Czech!)
    - Cost-effective ($0.016/M tokens)
    """
```

#### Embedding Configuration

```python
@dataclass
class EmbeddingConfig:
    provider: str = "deepinfra"
    model: str = "Qwen/Qwen3-Embedding-8B"
    dimensions: int = 4096
    batch_size: int = 100  # Chunks per API call
    normalize: bool = True  # Required for cosine similarity
    cache_enabled: bool = True  # LRU cache for repeated queries
```

#### Batch Embedding Flow

```python
def embed_chunks(
    self,
    chunks: Dict[str, List[Chunk]]
) -> Dict[str, np.ndarray]:
    """
    Embed all chunks from all layers.

    Process:
    1. Extract embedding_text from each chunk
    2. Batch texts (100 per API call)
    3. Call embedding API
    4. Normalize vectors (L2 norm)
    5. Cache results

    Returns:
        {
            "layer1": np.array([embedding1, ...]),
            "layer2": np.array([embedding1, ...]),
            "layer3": np.array([embedding1, ...])
        }
    """
```

### 5.2 PostgreSQL Vector Storage

**File:** `src/storage/postgres_adapter.py` (1745 lines)

#### PostgresVectorStoreAdapter Class (`src/storage/postgres_adapter.py:210-245`)

```python
class PostgresVectorStoreAdapter(VectorStoreAdapter):
    """
    PostgreSQL + pgvector implementation of vector store.

    Architecture:
    - 3 tables: vectors.layer1, vectors.layer2, vectors.layer3
    - HNSW indexes for fast approximate nearest neighbor search
    - Full-text search integrated (tsvector + GIN indexes)
    - Connection pooling via asyncpg

    Performance:
    - Search latency: 10-50ms (vs FAISS 5-10ms)
    - Supports millions of vectors
    - ACID transactions, persistent storage
    """
```

#### Schema Definition

```sql
-- Schema: vectors (NOT public!)
-- IMPORTANT: Always use vectors.layer{n} when querying

-- Layer 1: Document-level embeddings
CREATE TABLE vectors.layer1 (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    title TEXT,
    embedding VECTOR(4096),  -- Qwen3-Embedding-8B dimensions
    content TEXT,
    cluster_id INTEGER,
    cluster_label TEXT,
    cluster_confidence FLOAT,
    hierarchical_path TEXT,
    page_number INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Layer 2: Section-level embeddings
CREATE TABLE vectors.layer2 (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    section_id TEXT,
    section_title TEXT,
    section_path TEXT,
    section_level INTEGER,
    section_depth INTEGER,
    hierarchical_path TEXT,
    page_number INTEGER,
    embedding VECTOR(4096),
    content TEXT,  -- Section summary (NOT full text)
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    cluster_id INTEGER,
    cluster_label TEXT,
    cluster_confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Layer 3: Chunk-level embeddings (PRIMARY for RAG)
CREATE TABLE vectors.layer3 (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    section_id TEXT,
    section_title TEXT,
    section_path TEXT,
    section_level INTEGER,
    section_depth INTEGER,
    hierarchical_path TEXT,
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    embedding VECTOR(4096),
    content TEXT,  -- SAC-formatted: [breadcrumb]\n\ncontext\n\nraw_content
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    cluster_id INTEGER,
    cluster_label TEXT,
    cluster_confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- HNSW indexes for fast similarity search
CREATE INDEX layer1_embedding_idx ON vectors.layer1
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX layer2_embedding_idx ON vectors.layer2
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX layer3_embedding_idx ON vectors.layer3
    USING hnsw (embedding vector_cosine_ops);

-- GIN indexes for full-text search
CREATE INDEX layer2_content_tsv_idx ON vectors.layer2 USING GIN (content_tsv);
CREATE INDEX layer3_content_tsv_idx ON vectors.layer3 USING GIN (content_tsv);

-- Document filter indexes
CREATE INDEX layer2_document_id_idx ON vectors.layer2 (document_id);
CREATE INDEX layer3_document_id_idx ON vectors.layer3 (document_id);
```

#### Search Methods

**Hierarchical Search:**

```python
def hierarchical_search(
    self,
    query_embedding: np.ndarray,
    k_layer3: int = 6,
    use_doc_filtering: bool = True,
    similarity_threshold_offset: float = 0.25,
    query_text: Optional[str] = None,
    document_filter: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Hierarchical 3-layer search.

    Strategy:
    1. Search Layer 1 → find top document
    2. Use document_id to filter Layer 3 search
    3. Retrieve top-k chunks from Layer 3
    4. Apply similarity threshold filtering
    5. Search Layer 2 for context

    Returns:
        {
            "layer1": [doc_results],
            "layer2": [section_results],
            "layer3": [chunk_results]
        }
    """
```

**Hybrid Search (Vector + BM25):**

```python
async def _hybrid_search_layer(
    self,
    conn: asyncpg.Connection,
    layer: int,
    query_vec: np.ndarray,
    query_text: str,
    k: int,
    document_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Hybrid search: Dense (pgvector) + Sparse (BM25) with RRF fusion.

    RRF (Reciprocal Rank Fusion):
    - score = 1/(k + rank)
    - k=60 (standard parameter from research)

    Fetches MORE candidates for effective fusion (10x requested k).
    """
```

#### Metadata Filtering (`src/storage/postgres_adapter.py:28-147`)

```python
@dataclass
class MetadataFilter:
    """
    Metadata filter for vector search queries.

    Supports filtering by:
    - category: Exact match on category
    - categories: Match any of these categories
    - keywords: Must contain ALL (AND)
    - keywords_any: Must contain ANY (OR)
    - entities: Must contain ANY entity
    - entity_types: Must contain ANY type
    - min_confidence: Minimum category confidence

    Uses PostgreSQL JSONB operators:
    - ->> for text extraction
    - @> for containment
    - ?| for any-of array match
    - ?& for all-of array match
    """
```

---

## 6. Phase 5A: Knowledge Graph

**File:** `src/graph/graphiti_types.py` (781 lines)

### Overview

Phase 5A constructs a temporal knowledge graph using Graphiti and Neo4j. The system defines 55 entity types optimized for Czech legal/nuclear documents.

### GraphitiEntityType Enum (`src/graph/graphiti_types.py:33-117`)

```python
class GraphitiEntityType(str, Enum):
    """
    Extended entity types for Graphiti extraction (55 total).

    Categories:
    - Core (8): STANDARD, ORGANIZATION, DATE, CLAUSE, TOPIC, PERSON, LOCATION, CONTRACT
    - Regulatory (6): REGULATION, DECREE, DIRECTIVE, TREATY, LEGAL_PROVISION, REQUIREMENT
    - Authorization (2): PERMIT, LICENSE_CONDITION
    - Nuclear Technical (9): REACTOR, FACILITY, SYSTEM, SAFETY_FUNCTION, etc.
    - Events (4): INCIDENT, EMERGENCY_CLASSIFICATION, INSPECTION, etc.
    - Liability (1): LIABILITY_REGIME
    - Legal Terminology (2): LEGAL_TERM, DEFINITION
    - Czech Legal (+8): VYHLASKA, NARIZENI, SBIRKA_ZAKONU, METODICKY_POKYN, etc.
    - Technical Parameters (+7): NUMERIC_THRESHOLD, MEASUREMENT_UNIT, TIME_PERIOD, etc.
    - Process Types (+5): RADIATION_ACTIVITY, MAINTENANCE_ACTIVITY, etc.
    - Compliance Types (+3): COMPLIANCE_GAP, RISK_FACTOR, MITIGATION_MEASURE
    """
```

### Entity Type Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    55 ENTITY TYPES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CORE (8)                      REGULATORY (6)                   │
│  • standard                    • regulation                     │
│  • organization                • decree                         │
│  • date                        • directive                      │
│  • clause                      • treaty                         │
│  • topic                       • legal_provision                │
│  • person                      • requirement                    │
│  • location                                                     │
│  • contract                    AUTHORIZATION (2)                │
│                                • permit                         │
│  NUCLEAR TECHNICAL (9)         • license_condition              │
│  • reactor                                                      │
│  • facility                    EVENTS (4)                       │
│  • system                      • incident                       │
│  • safety_function             • emergency_classification       │
│  • fuel_type                   • inspection                     │
│  • isotope                     • decommissioning_phase          │
│  • radiation_source                                             │
│  • waste_category              LIABILITY (1)                    │
│  • dose_limit                  • liability_regime               │
│                                                                 │
│  CZECH LEGAL (+8)              TECHNICAL PARAMETERS (+7)        │
│  • vyhlaska (vyhláška)         • numeric_threshold              │
│  • narizeni (nařízení)         • measurement_unit               │
│  • sbirka_zakonu               • time_period                    │
│  • metodicky_pokyn             • frequency                      │
│  • sujb_rozhodnuti             • percentage                     │
│  • bezpecnostni_dokumentace    • temperature                    │
│  • limitni_stav                • pressure                       │
│  • mezni_hodnota                                                │
│                                                                 │
│  PROCESS TYPES (+5)            COMPLIANCE TYPES (+3)            │
│  • radiation_activity          • compliance_gap                 │
│  • maintenance_activity        • risk_factor                    │
│  • emergency_procedure         • mitigation_measure             │
│  • training_requirement                                         │
│  • documentation_requirement                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pydantic Entity Models

Each entity type has a corresponding Pydantic model for structured extraction:

```python
class VyhlaskaEntity(GraphitiEntityBase):
    """
    Czech decree (vyhláška).
    Example: "vyhláška č. 422/2016 Sb."
    """

    entity_type: Literal[GraphitiEntityType.VYHLASKA] = GraphitiEntityType.VYHLASKA
    cislo: Optional[str] = Field(default=None, description="Decree number (e.g., '422/2016')")
    rok: Optional[int] = Field(default=None, ge=1900, le=2100, description="Year of issue")
    sbirka: Literal["Sb.", "Sb.m.s."] = Field(default="Sb.", description="Collection type")
    nazev: Optional[str] = Field(default=None, description="Full title in Czech")
    ministerstvo: Optional[str] = Field(default=None, description="Issuing ministry")
    ucinnost_od: Optional[datetime] = Field(default=None, description="Effective from date")
    parent_zakon: Optional[str] = Field(default=None, description="Parent law reference")

    @field_validator("cislo")
    @classmethod
    def validate_cislo(cls, v: Optional[str]) -> Optional[str]:
        """Validate Czech decree number format."""
        if v is not None and not re.match(r"^\d+/\d{4}$", v):
            raise ValueError(f"Invalid decree number format: '{v}'")
        return v
```

### Knowledge Graph Storage

Entities and relationships are stored in Neo4j via Graphiti:

```
┌─────────────────────────────────────────────────────────────────┐
│                      NEO4J GRAPH SCHEMA                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NODES                                                          │
│  ├── Entity (55 types)                                          │
│  │   ├── label: str (primary identifier)                        │
│  │   ├── entity_type: GraphitiEntityType                        │
│  │   ├── confidence: float (0.0-1.0)                            │
│  │   ├── description: str                                       │
│  │   └── properties: {...} (type-specific)                      │
│  │                                                              │
│  └── Document                                                   │
│      ├── document_id: str                                       │
│      ├── title: str                                             │
│      └── indexed_at: datetime                                   │
│                                                                 │
│  RELATIONSHIPS                                                  │
│  ├── (Entity)-[:MENTIONED_IN]->(Document)                       │
│  ├── (Entity)-[:REFERENCES]->(Entity)                           │
│  ├── (Requirement)-[:APPLIES_TO]->(System)                      │
│  ├── (Vyhlaska)-[:IMPLEMENTS]->(Regulation)                     │
│  └── (ComplianceGap)-[:VIOLATES]->(Requirement)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Document Entity Merging

Graphiti automatically merges entities across documents:

```python
# Example: "SÚJB" mentioned in multiple documents
# → Single entity node with multiple MENTIONED_IN relationships

# Entity resolution based on:
# 1. Exact label match
# 2. Entity type match
# 3. Similarity threshold for fuzzy matching
```

---

## 7. Phase 5C: Retrieval

**File:** `src/retrieval/fusion_retriever.py` (794 lines)

### Overview

Phase 5C implements HyDE + Expansion Fusion retrieval, combining multiple query representations for improved recall.

### FusionConfig (`src/retrieval/fusion_retriever.py:79-107`)

```python
@dataclass
class FusionConfig:
    """Configuration for fusion retrieval."""

    original_weight: float = 0.5   # Weight for original query (direct match)
    hyde_weight: float = 0.25      # Weight for HyDE scores
    expansion_weight: float = 0.25 # Weight for expansion scores (split between 2)
    default_k: int = 16            # Default number of results
    candidates_multiplier: int = 4 # Retrieve k * multiplier candidates per query
```

### HyDE + Expansion Fusion Algorithm

```
┌─────────────────────────────────────────────────────────────────┐
│                 HyDE + EXPANSION FUSION RETRIEVAL               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: QUERY EXPANSION (LLM)                                  │
│  ┌───────────────────┐                                          │
│  │   User Query      │                                          │
│  │   "What is the    │                                          │
│  │    safety margin?"│                                          │
│  └─────────┬─────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                        LLM                               │   │
│  │  Generates:                                              │   │
│  │  • HyDE document (hypothetical answer)                   │   │
│  │  • Expansion 1 (alternative phrasing)                    │   │
│  │  • Expansion 2 (related terms)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│            │                                                    │
│            ▼                                                    │
│  STEP 2: EMBEDDING (4 vectors)                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │ Original│ │  HyDE   │ │  Exp 1  │ │  Exp 2  │              │
│  │  Query  │ │Document │ │         │ │         │              │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│       │           │           │           │                    │
│       ▼           ▼           ▼           ▼                    │
│  STEP 3: SEARCH (PARALLEL with asyncio.gather)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    PostgreSQL Layer 3                     │  │
│  │  Original: HYBRID (vector + BM25) - 64 candidates        │  │
│  │  HyDE: pure vector - 32 candidates                        │  │
│  │  Exp 1: pure vector - 32 candidates                       │  │
│  │  Exp 2: pure vector - 32 candidates                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│            │                                                    │
│            ▼                                                    │
│  STEP 4: NORMALIZE (min-max per signal)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  For each signal: score_norm = (score - min) / (max - min)│  │
│  │  Missing scores → 0.0                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│            │                                                    │
│            ▼                                                    │
│  STEP 5: WEIGHTED FUSION                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  final = 0.5 × original                                   │  │
│  │        + 0.25 × hyde                                      │  │
│  │        + 0.125 × exp_0                                    │  │
│  │        + 0.125 × exp_1                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│            │                                                    │
│            ▼                                                    │
│  STEP 6: TOP-K SELECTION (O(n) with np.argpartition)           │
│  └── Return top k results sorted by fused score                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### FusionRetriever Class (`src/retrieval/fusion_retriever.py:109-167`)

```python
class FusionRetriever:
    """
    HyDE + Expansion Fusion Retriever.

    Combines HyDE (hypothetical document embeddings) with query expansions
    using weighted score fusion for improved retrieval.

    Features:
    - 4-signal fusion: original + hyde + 2 expansions
    - Parallel async searches (4x faster)
    - Vectorized normalization with NumPy
    - 5-minute result cache (300s TTL)

    Research basis:
    - HyDE: Gao et al. (2022) - +15-30% recall for zero-shot retrieval
    - Query Expansion: Standard IR technique for vocabulary mismatch
    """
```

### Parallel Search Implementation (`src/retrieval/fusion_retriever.py:472-546`)

```python
async def _parallel_search(
    self,
    query_text: str,
    orig_emb: np.ndarray,
    hyde_emb: np.ndarray,
    exp_0_emb: np.ndarray,
    exp_1_emb: np.ndarray,
    k_hybrid: int,
    k_pure: int,
    document_filter: Optional[str],
) -> tuple:
    """
    Execute parallel searches with asyncio.gather.

    Performance: ~4x faster than sequential searches.

    Strategy:
    - Original query: HYBRID search (vector + BM25) with more candidates
    - HyDE/expansions: pure vector search with fewer candidates

    Error handling:
    - return_exceptions=True for graceful partial failures
    - Log errors but continue with available results
    """
```

### Vectorized Fusion (`src/retrieval/fusion_retriever.py:376-470`)

```python
def _batch_normalize_and_fuse(
    self,
    chunk_data: Dict[str, Dict],
    k: int,
) -> List[Dict]:
    """
    Vectorized score normalization and fusion using NumPy.

    Performance: ~3-5x faster than loop-based for 200+ candidates.

    Uses:
    - Matrix operations for min-max normalization
    - np.dot for weighted fusion
    - np.argpartition for O(n) top-k selection
    """
```

### Layer 2 Search

For section-level queries (overview, structure):

```python
def search_layer2(
    self,
    query: str,
    k: Optional[int] = None,
    document_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Search Layer 2 (sections) using HyDE + Expansion fusion.

    Use for:
    - Overview queries ("Co obsahuje kapitola X?")
    - Section discovery ("Které sekce pojednávají o Y?")
    - Document structure questions

    Note: No original query in fusion (only hyde + expansions).
    Weights renormalized: hyde=0.5, expansions=0.5
    """
```

---

## 8. Data Structures Reference

### ExtractedDocument

```python
@dataclass
class ExtractedDocument:
    document_id: str
    title: str
    full_text: str
    document_summary: Optional[str]
    hierarchy_depth: int
    sections: List[DocumentSection]
    metadata: Dict[str, Any]
```

### DocumentSection

```python
@dataclass
class DocumentSection:
    section_id: str
    title: str
    path: str  # Breadcrumb path
    level: int  # Heading level (1-6)
    depth: int  # Nesting depth
    content: str
    summary: Optional[str]
    page_number: int
    char_start: int
    char_end: int
```

### Chunk JSON Format

```json
{
  "chunk_id": "doc_L3_c1_sec_1",
  "context": "SAC context summary (what chunk is about)",
  "raw_content": "Actual text content from the document",
  "embedding_text": "[breadcrumb]\n\ncontext\n\nraw_content",
  "metadata": {
    "chunk_id": "doc_L3_c1_sec_1",
    "layer": 3,
    "document_id": "doc",
    "section_id": "sec_1",
    "section_title": "1. Introduction",
    "section_path": "1. Introduction",
    "page_number": 5,
    "char_start": 0,
    "char_end": 1500
  }
}
```

---

## 9. Configuration Reference

### config.json Settings

```json
{
  "storage": {
    "backend": "postgresql",
    "vectors_schema": "vectors"
  },
  "embedding": {
    "provider": "deepinfra",
    "model": "Qwen/Qwen3-Embedding-8B",
    "dimensions": 4096,
    "batch_size": 100
  },
  "chunking": {
    "max_tokens": 512,
    "tokenizer_model": "Qwen/Qwen3-Embedding-8B",
    "enable_contextual": true,
    "min_chunk_length": 50
  },
  "knowledge_graph": {
    "enabled": true,
    "backend": "graphiti",
    "neo4j_uri": "bolt://localhost:7687"
  },
  "retrieval": {
    "method": "hyde_expansion_fusion",
    "original_weight": 0.5,
    "hyde_weight": 0.25,
    "expansion_weight": 0.25,
    "default_k": 16
  }
}
```

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=xxx          # Gemini extraction
DATABASE_URL=postgresql://user:pass@host:5432/sujbot
DEEPINFRA_API_KEY=xxx       # Embeddings

# Optional
ANTHROPIC_API_KEY=xxx       # Claude for summaries
OPENAI_API_KEY=xxx          # GPT fallback
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxx

# Extraction backend selection
EXTRACTION_BACKEND=auto     # auto | gemini | unstructured
```

---

## 10. Complete Pipeline Example

### Running the Pipeline

```bash
# Index single document
uv run python run_pipeline.py data/bezpecnostni_zprava.pdf

# Index directory
uv run python run_pipeline.py data/documents/

# With verbose output
uv run python run_pipeline.py data/doc.pdf --verbose
```

### Output Directory Structure

```
output/bezpecnostni_zprava/
├── phase1_extraction.json     # Raw extracted content
├── phase2_summaries.json      # With section summaries
├── phase3_chunks.json         # Multi-layer chunks
├── phase4_embeddings.npz      # NumPy embedding arrays
├── phase5_kg_entities.json    # Extracted entities
└── pipeline_stats.json        # Processing statistics
```

### Pipeline Statistics

```json
{
  "document_id": "BZ_VR1",
  "phases": {
    "extraction": {
      "duration_seconds": 45.2,
      "pages": 150,
      "sections": 42,
      "backend": "gemini"
    },
    "summaries": {
      "duration_seconds": 23.5,
      "sections_processed": 42,
      "model": "claude-sonnet-4-5"
    },
    "chunking": {
      "duration_seconds": 5.8,
      "layer1_count": 1,
      "layer2_count": 42,
      "layer3_count": 285,
      "total_chunks": 328
    },
    "embedding": {
      "duration_seconds": 12.3,
      "vectors_created": 328,
      "model": "Qwen/Qwen3-Embedding-8B"
    },
    "knowledge_graph": {
      "duration_seconds": 18.7,
      "entities_extracted": 156,
      "relationships_created": 89
    }
  },
  "total_duration_seconds": 105.5,
  "cost_usd": 0.23
}
```

---

## 11. Research Basis

### Papers Implemented

| Paper | Year | Implementation |
|-------|------|----------------|
| LegalBench-RAG (Pipitone & Alami) | 2024 | 512-token chunks, RCTS |
| Summary-Augmented Chunking (Reuter et al.) | 2024 | SAC, generic summaries |
| Multi-Layer Embeddings (Lima) | 2024 | 3-layer indexing |
| Contextual Retrieval (Anthropic) | 2024 | Context prepending, -67% failures |
| HyDE (Gao et al.) | 2022 | Hypothetical Document Embeddings |
| Query Expansion | Standard IR | 2 query reformulations |

### Key Research Findings

1. **Small chunks optimal**: 512 tokens outperform larger chunks for legal documents
2. **Generic summaries better**: Counterintuitively, generic summaries improve retrieval vs expert-guided
3. **Multi-layer beats single-layer**: 2.3x more essential chunks with 3-layer indexing
4. **SAC reduces drift**: -58% context drift with summary prepending
5. **HyDE improves zero-shot**: +15-30% recall for queries without training data

---

## Summary

The SUJBOT indexing pipeline transforms PDF documents into a queryable knowledge base through 5 phases:

1. **Extraction**: Gemini 2.5 Flash extracts structured content
2. **Summaries**: Hierarchical summarization (section → document)
3. **Chunking**: 3-layer SAC chunks (512 tokens, sentence-aware)
4. **Storage**: pgvector embeddings + PostgreSQL storage
5. **KG**: 55 entity types in Neo4j via Graphiti

The retrieval system uses HyDE + Expansion Fusion with 4-signal weighted fusion for optimal recall and precision.

---

**Authors:** SUJBOT Development Team
**License:** Proprietary
