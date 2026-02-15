# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

**SUJBOT**: Production RAG system for legal/technical documents with multi-agent orchestration.

CRITICAL: See sudo password @sudo.txt

## Debugging with LangSmith

**IMPORTANT:** When debugging conversations/traces, ALWAYS use **LangSmith MCP tools** (`mcp__langsmith__*`), NOT Python scripts.

## LangSmith Evaluation

**When user says "evaluuj v langsmith", run this command:**
```bash
uv run python scripts/langsmith_eval.py \
    --dataset-path dataset/dataset_exp_ver_2.json \
    --dataset-name "sujbot-eval-qa-40" \
    --replace-dataset \
    --experiment-prefix "sujbot-qa-40-optimized" \
    --judge-model anthropic:claude-sonnet-4-5
```

Other: `--limit 5` (quick test), `--upload-only`, `--judge-model openai:gpt-4o-mini`

## Common Commands

```bash
# Development
uv sync                                    # Install dependencies
uv run python run_pipeline.py data/doc.pdf # Index single document
uv run python run_pipeline.py data/        # Index directory

# Testing
uv run pytest tests/ -v                                    # All tests
uv run pytest tests/test_phase4_indexing.py -v             # Single file
uv run pytest tests/agent/test_tool_registry.py::test_name -v  # Single test
uv run pytest tests/ --cov=src --cov-report=html           # With coverage
# NOTE: pytest-asyncio NOT installed. Use @pytest.mark.anyio + conftest anyio_backend="asyncio".

# Production tests (requires running Docker stack)
PROD_BASE_URL="http://localhost:8200" \
PROD_TEST_USER="prodtest@example.com" \
PROD_TEST_PASSWORD="ProdTest123!" \
uv run pytest tests/production/ -v

# Linting & Type Checking
uv run black src/ tests/ --line-length 100                 # Format code
uv run isort src/ tests/ --profile black                   # Sort imports
uv run mypy src/                                           # Type check

# Docker (full stack)
docker compose up -d                       # Start all services
docker compose logs -f backend             # Watch backend logs

# Frontend: Vite hot reload — NO rebuild needed for .tsx/.css changes

# Agent CLI
uv run python -m src.agent.cli             # Interactive mode
uv run python -m src.agent.cli --debug     # Debug mode
```

### Scripts using PostgresVectorStoreAdapter
Scripts that use `PostgresVectorStoreAdapter` must apply `nest_asyncio` before any DB calls:
```python
import nest_asyncio; nest_asyncio.apply()
```
The adapter's sync wrappers (`_run_async_safe`) call `loop.run_until_complete()` which fails
without `nest_asyncio` when already inside `asyncio.run()`. Also: `.env` DATABASE_URL points
to port 5433 (dev, often down) — override with port 5432 for production DB.

## Production Deployment

**CRITICAL: Frontend must be built with empty `VITE_API_BASE_URL` for production!**

```bash
# Build frontend for production (uses relative URLs)
docker build -t sujbot-frontend --target production \
  --build-arg VITE_API_BASE_URL="" \
  -f docker/frontend/Dockerfile frontend/

# Deploy frontend container
docker stop sujbot_frontend && docker rm sujbot_frontend
docker run -d --name sujbot_frontend \
  --network sujbot_sujbot_prod_net \
  --network-alias frontend \
  --restart unless-stopped \
  sujbot-frontend
```

- `--network-alias frontend` is REQUIRED for Docker DNS resolution
- 404 on `/admin` → missing SPA fallback; 502 → missing `--network-alias`; `localhost:8000` in browser → wrong `VITE_API_BASE_URL`

### Backend Deploy (full recreation)
```bash
docker stop sujbot_backend && docker rm sujbot_backend
docker run -d --name sujbot_backend \
  --network sujbot_sujbot_prod_net --network-alias backend \
  --env-file /home/prusemic/SUJBOT/.env \
  -e DATABASE_URL=postgresql://postgres:sujbot_secure_password@sujbot_postgres:5432/sujbot \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/prompts:/app/prompts:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/data/vl_pages:/app/data/vl_pages \
  -v $(pwd)/vector_db:/app/vector_db:ro \
  --restart unless-stopped sujbot-backend
docker network connect sujbot_sujbot_db_net sujbot_backend
```
- MUST connect to BOTH networks: `sujbot_prod_net` (nginx) + `sujbot_db_net` (postgres)
- `--env-file .env` required (API keys); `-e DATABASE_URL=...@sujbot_postgres:5432/...` overrides dev port
- `/app/data` must be writable (no `:ro`) for document uploads
- `/app/vector_db:ro` required even if empty (validation check)

## Architecture Overview

```
User Query → SingleAgentRunner (autonomous tool loop)
  → RAG Tools (search, expand_context, get_document_info, compliance_check, etc.)
  → Retrieval: OCR mode (HyDE + Expansion Fusion) or VL mode (Jina v4 cosine)
  → Storage (PostgreSQL: vectors + checkpoints)
```

**Key directories:**
- `src/single_agent/` - Production runner (autonomous tool loop with unified prompt)
- `src/agent/` - Agent CLI and tools (`tools/` has individual tool files)
- `src/graph/` - Graph RAG (storage, embedder, entity extraction, communities)
- `src/multi_agent/` - Legacy LangGraph-based multi-agent system
- `src/retrieval/` - HyDE + Expansion Fusion retrieval pipeline
- `src/vl/` - Vision-Language RAG module (Jina v4 embeddings, page store, VL retriever)
- `backend/` - FastAPI web backend with auth, routes, middleware
- `frontend/` - React + Vite web UI

**Key SSOT modules (use these, don't duplicate):**
- `src/exceptions.py` - Typed exception hierarchy
- `src/utils/cache.py` - `LRUCache` + `TTLCache`
- `src/multi_agent/core/agent_initializer.py` - Agent initialization
- `src/multi_agent/prompts/loader.py` - Prompt loading from `prompts/`
- `src/agent/providers/factory.py` - Provider creation + `detect_provider_from_model()`
- `src/vl/__init__.py:create_vl_components()` - VL initialization factory (both runners use this)
- `backend/constants.py:get_variant_model()` - Variant→model mapping (falls back for unknown names)
- `src/graph/embedder.py:GraphEmbedder` - multilingual-e5-small (384-dim) for graph semantic search
- `src/graph/storage.py:GraphStorageAdapter` - Graph CRUD + embedding/FTS search

### VL (Vision-Language) Architecture

**Dual architecture: OCR vs VL** — switchable via `config.json` → `"architecture": "ocr" | "vl"`.

```
VL flow: Query → Jina v4 embed_query() → PostgreSQL exact cosine (vectors.vl_pages)
  → top-k page images (base64 PNG) → multimodal tool result → VL-capable LLM
```

- No HNSW index — ~500 pages, exact cosine scan <50ms, 100% recall
- `"local"` variant: `Qwen/Qwen3-VL-235B-A22B-Instruct` (DeepInfra, vision)
- `"remote"` variant: Haiku 4.5 (vision natively)
- DeepInfra provider converts Anthropic image blocks → OpenAI `image_url` format

| | OCR mode | VL mode |
|---|---|---|
| Embedding | Qwen3-Embedding-8B (4096-dim) | Jina v4 (2048-dim) |
| Content unit | Text chunks (512 tokens) | Page images (PNG) |
| DB table | `vectors.layer3` | `vectors.vl_pages` |
| Search | HyDE + Expansion Fusion | Simple cosine |
| Token cost | ~100 tokens/chunk | ~1600 tokens/page |

**Graph search** uses `intfloat/multilingual-e5-small` (384-dim) for cross-language semantic search on entities/communities. Falls back to PostgreSQL FTS when embedder not configured.

## Critical Constraints (DO NOT CHANGE)

Research-backed decisions. **DO NOT modify** without explicit approval.

### 1. SINGLE SOURCE OF TRUTH (SSOT)

- **One canonical implementation** per feature — delete obsolete code immediately
- **No duplicate code** — no legacy wrappers, no re-exports
- **API keys in `.env` ONLY** — NEVER in config.json or code

### 2. Autonomous Agents (NOT Hardcoded)

Agents MUST be LLM-driven. LLM decides tool calling sequence via `BaseAgent.run_autonomous_tool_loop()`. System prompts guide behavior, NOT code. No "step 1, step 2" logic. **NEVER use hardcoded template responses.**

### 3. Hierarchical Document Summaries

`Sections → Section Summaries → Document Summary`. NEVER pass full document text to LLM. Length: 100-1000 chars (adaptive).

### 4. Token-Aware Chunking

512 tokens max (tiktoken). Changing this invalidates ALL vector stores!

### 5. Summary-Augmented Chunking (SAC)

Prepend document summary during embedding, strip during retrieval. -58% context drift (Anthropic, 2024).

### 6. Multi-Layer Embeddings

3 separate indexes (document/section/chunk) — NOT merged. 2.3x essential chunks vs single-layer.

### 6.1 Chunk JSON Format

```json
{
  "chunk_id": "doc_L3_c1_sec_1",
  "context": "SAC context summary",
  "raw_content": "Actual document text",
  "embedding_text": "[breadcrumb]\n\ncontext\n\nraw_content",
  "metadata": { "chunk_id": "...", "layer": 3, "document_id": "..." }
}
```

Do NOT use `content` field in JSON. `PhaseLoaders` use `embedding_text` as Chunk's `content`.

### 6.2 PostgreSQL Vector Schema

Vectors in `vectors` schema (NOT `public`). Tables: `layer1` (docs), `layer2` (sections), `layer3` (chunks), `vl_pages`, `documents`.

- Embeddings: 4096-dim (Qwen3-Embedding-8B) for OCR, 2048-dim (Jina v4) for VL
- Cosine similarity: `1 - (embedding <=> query_vector)`
- `vectors.documents`: Document registry with category (`documentation` | `legislation`). Created at upload, deleted on document removal. Backfill: `scripts/backfill_document_categories.py`.

### 7. No Cohere Reranking

Cohere performs WORSE on legal docs. Use `ms-marco` or `bge-reranker` instead.

### 8. Generic Summaries

GENERIC style (NOT expert terminology). Reuter et al. (2024) — generic summaries improve retrieval.

## Configuration

**Two-file system:** `.env` (secrets, gitignored) + `config.json` (settings, version-controlled). NO secrets in config.json.

**Extraction backend:** `EXTRACTION_BACKEND=auto|gemini|unstructured` (env var, default: auto).

## Best Practices

### System Prompts

**ALL LLM system prompts MUST be loaded from `prompts/` directory!** Never hardcode prompts in Python. Use `load_prompt("agent_name")` for agents, `Path("prompts/template.txt").read_text()` for pipeline components.

### Code Quality

- Type hints for public APIs, Google-style docstrings
- TDD: write tests BEFORE implementing
- Narrow exception catches — catch specific exceptions, not bare `Exception`
- After implementation, run code-simplifier: `Task tool → subagent_type: "code-simplifier:code-simplifier"`

### Error Handling

Use typed exceptions from `src/exceptions.py`: `SujbotError` → `ExtractionError`, `ValidationError`, `ProviderError`, `ToolExecutionError`, `AgentError`, `StorageError`, `RetrievalError`. Each has `message`, `details` dict, optional `cause`. Use `exc_info=True` for unexpected errors.

### Internationalization (i18n)

Frontend CZ/EN via react-i18next. When adding UI text, update BOTH `/frontend/src/i18n/locales/cs.json` and `en.json`. Use `t('key')` — no hardcoded strings.

### Adding New Backend API Routes

New routes MUST be added to nginx regex in `docker/nginx/reverse-proxy.conf` ("Direct backend endpoints" section). Alternative: prefix with `/api/` (auto-proxied, no nginx changes needed). **Symptom if forgotten:** Frontend receives HTML instead of JSON.

### Adding New Models

1. Fetch current pricing from provider API first
2. Add to `config.json` → `model_registry.llm_models` with `id`, `provider`, `pricing`, `context_window`
3. If DeepInfra: add to `agent_variants.deepinfra_supported_models`

### Model Selection

- **Production:** `claude-sonnet-4-5` | **Development:** `gpt-4o-mini` | **Budget:** `claude-haiku-4-5`

### LangSmith Observability

```bash
# .env — use EU endpoint for EU workspaces
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT_NAME=sujbot-multi-agent
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

Common issues: `403` → wrong endpoint (EU vs US); `0 tokens` → token counting doesn't propagate in LangGraph chains.

## Research Papers (DO NOT CONTRADICT)

1. **LegalBench-RAG** (Pipitone & Alami, 2024) — RCTS, 500-char chunks
2. **Summary-Augmented Chunking** (Reuter et al., 2024) — SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) — 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) — Context prepending
5. **HyDE** (Gao et al., 2022) — Hypothetical Document Embeddings

## Documentation

- [`README.md`](README.md) — Installation, quick start
- [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md) — Docker configuration
- [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) — Human-in-the-Loop
- [`docs/WEB_INTERFACE.md`](docs/WEB_INTERFACE.md) — Web UI features
