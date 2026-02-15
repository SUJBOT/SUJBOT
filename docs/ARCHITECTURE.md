# System Architecture

SUJBOT is a production RAG (Retrieval-Augmented Generation) system for legal and technical documents. It uses a single autonomous agent with tool-calling capabilities to answer questions over a corpus of indexed documents.

## System Overview

```
User (Browser)
  |
  v
Frontend (React + Vite, served by nginx)
  |
  v
Nginx (SSL termination, reverse proxy)
  |
  v
Backend (FastAPI, port 8000)
  |
  v
SingleAgentRunner (autonomous tool loop)
  |
  +---> RAG Tools (search, expand_context, get_document_info, ...)
  |       |
  |       v
  |     PostgreSQL (pgvector)
  |       - vectors.vl_pages (VL mode)
  |       - vectors.layer1/layer2/layer3 (OCR mode)
  |       - graph.entities / graph.relationships / graph.communities
  |
  +---> LLM Provider (Anthropic, OpenAI, DeepInfra)
```

## Dual Architecture: OCR vs VL

The system supports two retrieval architectures, switchable via `config.json` → `"architecture": "ocr" | "vl"`.

### VL (Vision-Language) Mode — Current Default

The VL pipeline treats each PDF page as an image and uses a multimodal embedding model to enable visual retrieval.

```
Query
  --> Jina v4 embed_query() (2048-dim)
  --> PostgreSQL exact cosine search (vectors.vl_pages)
  --> Top-k page images (base64 PNG)
  --> Multimodal LLM (vision-capable model)
  --> Answer with page-level citations
```

- **Embedding**: Jina Embeddings v4 (2048 dimensions)
- **Content unit**: Full page images (PNG, 150 DPI)
- **Storage**: `vectors.vl_pages` table
- **Search**: Exact cosine similarity (no ANN index — ~500 pages, <50ms)
- **Token cost**: ~1,600 tokens per page image

### OCR Mode

The OCR pipeline extracts text from PDFs and chunks it for traditional text-based retrieval.

```
Query
  --> HyDE (hypothetical document generation)
  --> Qwen3-Embedding-8B embed (4096-dim)
  --> PostgreSQL search (vectors.layer1/layer2/layer3)
  --> Expansion Fusion (reranking + fusion)
  --> Text chunks
  --> LLM
  --> Answer with chunk-level citations
```

- **Embedding**: Qwen3-Embedding-8B (4096 dimensions)
- **Content unit**: Text chunks (512 tokens max)
- **Storage**: `vectors.layer1` (docs), `vectors.layer2` (sections), `vectors.layer3` (chunks)
- **Search**: HyDE + Expansion Fusion
- **Token cost**: ~100 tokens per chunk

### Comparison

| | OCR mode | VL mode |
|---|---|---|
| Embedding model | Qwen3-Embedding-8B (4096-dim) | Jina v4 (2048-dim) |
| Content unit | Text chunks (512 tokens) | Page images (PNG) |
| DB table | `vectors.layer3` | `vectors.vl_pages` |
| Search method | HyDE + Expansion Fusion | Simple cosine |
| Token cost | ~100 tokens/chunk | ~1,600 tokens/page |
| Strengths | Lower cost, fine-grained | Layout-aware, tables/diagrams |

## Agent Architecture

### SingleAgentRunner

The production agent (`src/single_agent/runner.py`) replaces the legacy multi-agent orchestrator with a single autonomous LLM that has access to all tools.

**How it works:**
1. Load system prompt from `prompts/agents/unified.txt` (VL) or `prompts/agents/unified_ocr.txt` (OCR)
2. Initialize all tools via `ToolRegistry`
3. On each query, enter an autonomous tool loop (max 10 iterations):
   - LLM decides: call a tool OR produce final answer
   - If tool call: execute tool, add result to conversation, continue
   - If final answer: break and return
4. Early stop: exits after 2+ consecutive failed searches
5. Force final answer if max iterations reached

### Agent Variants

Users can select between two agent variants (per-user setting):

| Variant | Model | Provider | Vision |
|---------|-------|----------|--------|
| `remote` (default) | `claude-sonnet-4-5-20250929` | Anthropic | Yes |
| `local` | `Qwen/Qwen3-VL-235B-A22B-Instruct` | DeepInfra | Yes |

Variant configuration lives in `config.json` → `agent_variants`. The backend resolves variant to model via `backend/constants.py:get_variant_model()`.

## Tool Inventory

The agent has access to 8 tools (5 core RAG + 3 graph):

### Core RAG Tools

| Tool | File | Description |
|------|------|-------------|
| `search` | `src/agent/tools/search.py` | Search documents by query. VL mode: cosine on page images. OCR mode: HyDE + Expansion Fusion on text chunks. |
| `expand_context` | `src/agent/tools/expand_context.py` | Get surrounding chunks/pages for a given chunk ID. |
| `get_document_info` | `src/agent/tools/get_document_info.py` | Get metadata for a specific document (title, pages, sections). |
| `get_document_list` | `src/agent/tools/get_document_list.py` | List all available documents in the corpus. |
| `get_stats` | `src/agent/tools/get_stats.py` | Get RAG system statistics (document count, chunk count, etc.). |

### Graph RAG Tools (VL mode only)

| Tool | File | Description |
|------|------|-------------|
| `graph_search` | `src/agent/tools/graph_search.py` | Search entities by semantic embedding similarity (multilingual-e5-small). Cross-language capable. |
| `graph_context` | `src/agent/tools/graph_context.py` | Get entity relationships via N-hop recursive CTE traversal. |
| `graph_communities` | `src/agent/tools/graph_communities.py` | Search community summaries by semantic embedding similarity. |

See [GRAPH_RAG.md](GRAPH_RAG.md) for the full Graph RAG architecture.

## Key Modules (SSOT)

Each feature has a single canonical implementation. These are the authoritative modules:

| Module | Path | Purpose |
|--------|------|---------|
| Exceptions | `src/exceptions.py` | Typed exception hierarchy (`SujbotError` base) |
| Cache | `src/utils/cache.py` | `LRUCache` + `TTLCache` |
| Provider factory | `src/agent/providers/factory.py` | LLM provider creation + `detect_provider_from_model()` |
| VL factory | `src/vl/__init__.py:create_vl_components()` | VL initialization (JinaClient, PageStore, VLRetriever) |
| Variant mapping | `backend/constants.py:get_variant_model()` | Variant name to model ID resolution |
| Prompt loading | `src/multi_agent/prompts/loader.py` | Load prompts from `prompts/` directory |
| Agent init | `src/multi_agent/core/agent_initializer.py` | Agent initialization utilities |
| Graph storage | `src/graph/storage.py` | `GraphStorageAdapter` for PostgreSQL graph schema |

## Database Schemas

PostgreSQL is the sole database. Data is organized into three schemas:

### `vectors` — Vector Storage

- `vectors.layer1` — Document-level embeddings (OCR mode, 4096-dim)
- `vectors.layer2` — Section-level embeddings (OCR mode, 4096-dim)
- `vectors.layer3` — Chunk-level embeddings (OCR mode, 4096-dim)
- `vectors.vl_pages` — Page image embeddings (VL mode, 2048-dim)

### `auth` — Authentication

- User accounts, JWT tokens, spending tracking, conversation history, messages, feedback

### `graph` — Knowledge Graph

- `graph.entities` — Named entities extracted from pages
- `graph.relationships` — Relationships between entities
- `graph.communities` — Leiden community summaries

See [GRAPH_RAG.md](GRAPH_RAG.md) for detailed schema definitions.

## Directory Structure

```
src/
  single_agent/     # Production runner (autonomous tool loop)
  agent/            # Agent CLI, tools, providers
    tools/          # Individual tool implementations
    providers/      # LLM provider adapters (Anthropic, OpenAI, DeepInfra)
  vl/               # Vision-Language module (Jina, PageStore, VLRetriever)
  graph/            # Graph RAG (storage, entity extraction, communities)
  retrieval/        # HyDE + Expansion Fusion pipeline (OCR mode)
  multi_agent/      # Legacy multi-agent system (prompts/loader still used)
  exceptions.py     # Typed exception hierarchy
  config.py         # Configuration loading
backend/            # FastAPI web backend
  main.py           # App entry point, /chat endpoint
  routes/           # Route modules (auth, conversations, admin, etc.)
  agent_adapter.py  # Bridge between backend and SingleAgentRunner
  constants.py      # Variant-to-model mapping
frontend/           # React + Vite web UI
  src/i18n/         # CZ/EN translations
prompts/            # All LLM system prompts (loaded at runtime)
  agents/           # Agent system prompts (unified.txt, unified_ocr.txt)
data/               # PDF documents and page images
  vl_pages/         # Rendered page images ({doc_id}/page_{num}.png)
docker/             # Docker configs (nginx, postgres, frontend, backend)
scripts/            # Utility scripts (graph build, evaluation, etc.)
```
