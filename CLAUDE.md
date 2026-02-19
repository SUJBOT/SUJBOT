# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

**SUJBOT**: Production RAG system for legal/technical documents with autonomous tool-loop agent.

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

# Testing
uv run pytest tests/ -v --ignore=tests/production           # All tests (local)
uv run pytest tests/agent/test_tool_registry.py::test_name -v  # Single test
uv run pytest tests/ --cov=src --cov-report=html           # With coverage
# NOTE: pytest-asyncio NOT installed. Use @pytest.mark.anyio + conftest anyio_backend="asyncio".
# NOTE: pytest-timeout NOT installed. Do not use --timeout flag.

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
  --add-host=host.docker.internal:host-gateway \
  --env-file /home/prusemic/SUJBOT/.env \
  -e DATABASE_URL=postgresql://postgres:sujbot_secure_password@sujbot_postgres:5432/sujbot \
  -e LOCAL_LLM_BASE_URL=http://host.docker.internal:18080/v1 \
  -e LOCAL_EMBEDDING_BASE_URL=http://host.docker.internal:18081/v1 \
  -v $(pwd)/config.json:/app/config.json:ro \
  -v $(pwd)/prompts:/app/prompts:ro \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/data/vl_pages:/app/data/vl_pages \
  -v $(pwd)/vector_db:/app/vector_db:ro \
  --restart unless-stopped sujbot-backend
docker network connect sujbot_sujbot_db_net sujbot_backend
docker network connect bridge sujbot_backend
```
- **After container recreation**: Run `docker exec sujbot_nginx nginx -s reload` to force DNS re-resolution. Without this, nginx connects to old container IPs → 502 errors.
- MUST connect to THREE networks: `sujbot_prod_net` (nginx) + `sujbot_db_net` (postgres) + `bridge` (host.docker.internal access)
- `--add-host=host.docker.internal:host-gateway` required for local model access (vLLM + embedding) via socat bridges
- `--env-file .env` required (API keys); `-e DATABASE_URL=...@sujbot_postgres:5432/...` overrides dev port
- `LOCAL_LLM_BASE_URL` / `LOCAL_EMBEDDING_BASE_URL` override `.env` values (which point to localhost, unreachable from Docker)
- `/app/data` must be writable (no `:ro`) for document uploads
- `/app/vector_db:ro` required even if empty (validation check)

### Local Model Prerequisites (socat bridges)
Before backend can reach local models, SSH tunnels and socat bridges must be running:
```bash
# socat bridges (forward Docker-reachable ports to SSH tunnel ports)
socat TCP-LISTEN:18080,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:8080 &  # LLM 30B (gx10-eb6e)
socat TCP-LISTEN:18081,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:8081 &  # Embedding (gx10-fa34)
socat TCP-LISTEN:18082,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:8082 &  # LLM 8B (gx10-fa34)
```
UFW rules (one-time): `sudo ufw allow from 172.16.0.0/12 to any port 18080 proto tcp` (and 18081, 18082)

## Architecture Overview

```
User Query → SingleAgentRunner (autonomous tool loop)
  → RAG Tools (search, expand_context, get_document_info, compliance_check, web_search, etc.)
  → VL Retrieval: embedder cosine → page images → multimodal LLM
  → Storage (PostgreSQL: vectors + checkpoints)
  → LLM: "remote" (Sonnet 4.5 cloud) or "local" (Qwen3-VL-30B-A3B-Thinking on GB10 via vLLM)
```

**Key directories:**
- `src/single_agent/` - Production runner (autonomous tool loop with unified prompt)
- `src/agent/` - Agent CLI, tools (`tools/`), providers (`providers/`), observability
- `src/graph/` - Graph RAG (storage, embedder, entity extraction, communities)
- `src/vl/` - Vision-Language RAG module (Jina v4 embeddings, page store, VL retriever)
- `src/storage/` - PostgreSQL adapter + conversation mixin
- `src/utils/` - Security, retry, model registry, async helpers, text helpers, caching
- `backend/` - FastAPI web backend with auth, routes, middleware, deps (DI)
- `frontend/` - React + Vite web UI
- `rag_confidence/` - QPP-based retrieval confidence scoring (standalone, by veselm73, currently disabled)

**Key SSOT modules (use these, don't duplicate):**
- `src/exceptions.py` - Typed exception hierarchy
- `src/utils/cache.py` - `LRUCache` + `TTLCache`
- `src/utils/async_helpers.py` - `run_async_safe()` (sync→async bridge) + `vec_to_pgvector()`
- `src/utils/text_helpers.py` - `strip_code_fences()` shared by graph modules
- `src/storage/conversation_mixin.py` - `ConversationStorageMixin` (conversation CRUD, used by both storage adapters)
- `src/agent/providers/factory.py` - Provider creation + `detect_provider_from_model()`
- `src/agent/providers/openai_compat.py` - Shared Anthropic↔OpenAI format conversion helpers
- `src/agent/tools/adapter.py` - `ToolAdapter` (tool lookup, validation, execution, metrics)
- `src/agent/observability.py` - `setup_langsmith()` + `LangSmithIntegration`
- `src/vl/__init__.py:create_vl_components()` - VL initialization factory (embedder selection: jina/local)
- `src/vl/local_embedder.py:LocalVLEmbedder` - Local VL embedding via vLLM (drop-in for JinaClient)
- `src/graph/types.py` - `ENTITY_TYPES` + `RELATIONSHIP_TYPES` constants
- `src/graph/embedder.py:GraphEmbedder` - multilingual-e5-small (384-dim) for graph semantic search
- `src/graph/storage.py:GraphStorageAdapter` - Graph CRUD + embedding/FTS search
- `backend/deps.py` - Centralized backend dependency injection (auth, storage, VL/graph, PDF cache)
- `backend/constants.py:get_variant_model()` - Variant→model mapping (falls back for unknown names)
- `frontend/src/config.ts` - Single `API_BASE_URL` source for all API calls

### Graph RAG Gotchas

- `GraphStorageAdapter(connection_string=db_url)` — MUST use keyword arg. First positional param is `pool`, not connection string.
- `graph.entity_aliases` table stores alternate names for merged entities (created Feb 2026).
- Graph schema tables: `graph.entities`, `graph.relationships`, `graph.communities`, `graph.entity_aliases`.
- Entity types (19): REGULATION, STANDARD, SECTION, ORGANIZATION, PERSON, CONCEPT, REQUIREMENT, FACILITY, ROLE, DOCUMENT, OBLIGATION, PROHIBITION, PERMISSION, EVIDENCE, CONTROL, DEFINITION, SANCTION, DEADLINE, AMENDMENT.
- Relationship types (14): DEFINES, REFERENCES, AMENDS, REQUIRES, REGULATES, PART_OF, APPLIES_TO, SUPERVISES, AUTHORED_BY, SUPERSEDES, DERIVED_FROM, HAS_SANCTION, HAS_DEADLINE, COMPLIES_WITH.
- Legislation-specific types (Feb 2026): DEFINITION/SANCTION/DEADLINE/AMENDMENT entities + SUPERSEDES/DERIVED_FROM/HAS_SANCTION/HAS_DEADLINE/COMPLIES_WITH relationships for legislation navigation and compliance mapping.
- Entity dedup scripts: `scripts/graph_normalize_dedup.py` (normalization + trigram/LLM), `scripts/graph_rebuild_communities.py` (re-embed + detect + summarize).
- When merging entities with unique constraint `(name, entity_type, document_id)`: DELETE duplicates BEFORE UPDATE canonical to avoid constraint violation.

### VL (Vision-Language) Architecture

```
VL flow: Query → embedder embed_query() → PostgreSQL exact cosine (vectors.vl_pages)
  → top-k page images (base64 PNG) → multimodal tool result → VL-capable LLM
```

- 4096-dim embeddings stored in `vectors.vl_pages` (Qwen3-VL-Embedding-8B; was 2048 with Jina v4)
- Embedder selection via `config.json` → `vl.embedder`: `"local"` (Qwen3-VL-Embedding-8B on GB10) or `"jina"` (Jina v4 cloud)
- `src/vl/local_embedder.py:LocalVLEmbedder` — drop-in replacement for JinaClient (same interface)
- `_create_embedder()` factory in `src/vl/__init__.py` handles embedder instantiation
- No HNSW index — ~500 pages, exact cosine scan <50ms, 100% recall
- `"local"` variant: Qwen3-VL-30B-A3B-Thinking via vLLM on GB10 (single DGX Spark, 131K ctx, ~21 tok/s SSE, KV cache 512K tokens/3.9x concurrency)
- **vLLM production flags** (gx10-eb6e, 30B): `--max-model-len 131072 --max-num-batched-tokens 8192 --max-num-seqs 8 --gpu-memory-utilization 0.92 --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors`
- **8B helper model** (gx10-fa34): Qwen3-VL-8B-Instruct-FP8 via vLLM, coexists with embedding server. Container: `vllm-qwen3vl-8b`, port 8082/18082. Flags: `--max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 16 --gpu-memory-utilization 0.60 --enable-auto-tool-choice --tool-call-parser hermes`. Decode: ~20-23 tok/s, TTFT: 0.09-0.38s, KV cache: 405K tokens/12.35x concurrency. GPU: model 10.5 GiB + embedding 17 GiB = ~78 GiB / 119 GiB.
- Benchmark script: `scripts/vllm_benchmark.py` — TTFT, decode throughput, e2e latency for RAG profiles. Supports `--model` and `--compare-model` for cross-model A/B comparison.
- `"remote"` variant: Sonnet 4.5 (vision natively)
- `local_llm` provider: reuses DeepInfraProvider with custom `base_url` (vLLM OpenAI-compatible API)
- Dynamic max_tokens in runner: local_llm→32768 (thinking models emit large `<think>` blocks), Anthropic→4096 if configured > 16384 (SDK rejects high non-streaming values)
- **vLLM Qwen3 thinking tags**: Chat template strips opening `<think>` — only `</think>` appears in stream. Raw completions API shows both tags. `ThinkTagStreamParser(start_thinking=True)` handles this. Test with raw completions API (`client.completions.create`) if debugging tag behavior.
- **Streaming thinking**: `local_llm` uses `_stream_llm_iteration()` in runner — yields `thinking_delta`/`thinking_done`/`text_delta` events. Other providers keep non-streaming `create_message()`. Parser: `src/agent/providers/think_parser.py`.
- DeepInfra provider converts Anthropic image blocks → OpenAI `image_url` format
- ~1600 tokens/page

**Image search:** `search` tool accepts `image_attachment_index` (user attachment) or `image_page_id` (existing page) for image-based queries via `VLRetriever.search_by_image()`.

**Chat attachments:** Users can attach images, PDFs, and documents (DOCX, TXT, Markdown, HTML, LaTeX) to messages (base64 in JSON body, 10MB/file, max 5). Images pass through as multimodal blocks. PDFs rendered to page images via PyMuPDF (max 10 pages). Text documents have text extracted via `DocumentConverter.extract_text()` (max 30K chars). Attachments passed to agent as multimodal content blocks, NOT indexed into vector store. Per-request context (`ToolRegistry.set_request_context()`) makes attachment images available to tools.

**Multi-format upload:** Document upload accepts PDF, DOCX, TXT, Markdown, HTML, LaTeX. Non-PDF formats are converted to PDF first via `src/vl/document_converter.py`, then the unchanged VL pipeline processes the PDF. DOCX uses LibreOffice headless, LaTeX uses pdflatex, text/MD/HTML use PyMuPDF's `fitz.Story`.

**Graph search** uses `intfloat/multilingual-e5-small` (384-dim) for cross-language semantic search on entities/communities. Falls back to PostgreSQL FTS when embedder not configured.

**Web search** uses Gemini's native Google Search grounding (`web_search` tool). Last-resort tool for questions requiring current/external info not in the corpus. Requires `GOOGLE_API_KEY` in `.env`. Config: `config.json` → `agent_tools.web_search` (enabled/model). Citations: `\webcite{url}{title}` renders as clickable external link badges in the UI.

## Critical Constraints (DO NOT CHANGE)

Research-backed decisions. **DO NOT modify** without explicit approval.

### 1. SINGLE SOURCE OF TRUTH (SSOT)

- **One canonical implementation** per feature — delete obsolete code immediately
- **No duplicate code** — no legacy wrappers, no re-exports
- **API keys in `.env` ONLY** — NEVER in config.json or code

### 2. Autonomous Agents (NOT Hardcoded)

Agents MUST be LLM-driven. LLM decides tool calling sequence via `BaseAgent.run_autonomous_tool_loop()`. System prompts guide behavior, NOT code. No "step 1, step 2" logic. **NEVER use hardcoded template responses.**

### 3. PostgreSQL Vector Schema

Vectors in `vectors` schema (NOT `public`). Tables: `vl_pages`, `documents`.
Graph data in `graph` schema. Tables: `entities`, `relationships`, `communities`, `entity_aliases`.

- Embeddings: 4096-dim (Qwen3-VL-Embedding-8B local, was 2048 Jina v4) in `vectors.vl_pages`
- Cosine similarity: `1 - (embedding <=> query_vector)`
- `vectors.documents`: Document registry with category (`documentation` | `legislation`). Created at upload, deleted on document removal. Backfill: `scripts/backfill_document_categories.py`.

## Configuration

**Two-file system:** `.env` (secrets, gitignored) + `config.json` (settings, version-controlled). NO secrets in config.json.

## Best Practices

### Changelog

**After every merged PR or significant change, update `CHANGELOG.md` in the project root.** Keep it organized by feature area with dates. This is the primary record of what was done and when.

### Git Branching

**Do NOT create a new feature branch if you already have an active (unmerged) branch.** Finish, merge, or close the current branch first. Multiple active branches from the same author cause merge conflicts and duplicated work.

**Main branch is protected** — direct push is rejected. All changes must go through a pull request, even cherry-picks.

### System Prompts

**ALL LLM system prompts MUST be loaded from `prompts/` directory!** Never hardcode prompts in Python. Use `load_prompt("agent_name")` for agents, `Path("prompts/template.txt").read_text()` for pipeline components.

### Code Quality

- Type hints for public APIs, Google-style docstrings
- TDD: write tests BEFORE implementing
- Narrow exception catches — catch specific exceptions, not bare `Exception`
- After implementation, run code-simplifier: `Task tool → subagent_type: "code-simplifier:code-simplifier"`

### Error Handling

Use typed exceptions from `src/exceptions.py`: `SujbotError` → `ExtractionError`, `ValidationError`, `ProviderError`, `ToolExecutionError`, `AgentError`, `StorageError`, `RetrievalError`, `ConversionError`. Each has `message`, `details` dict, optional `cause`. Use `exc_info=True` for unexpected errors.

**API error responses:** Always use `sanitize_error(e)` from `src/utils/security` in user-facing error payloads (SSE, JSON responses). Never expose raw exception messages to clients.

### Internationalization (i18n)

Frontend CZ/EN via react-i18next. When adding UI text, update BOTH `/frontend/src/i18n/locales/cs.json` and `en.json`. Use `t('key')` — no hardcoded strings.

### Adding New Backend API Routes

New routes MUST be added to nginx regex in `docker/nginx/reverse-proxy.conf` ("Direct backend endpoints" section). Alternative: prefix with `/api/` (auto-proxied, no nginx changes needed). **Symptom if forgotten:** Frontend receives HTML instead of JSON.

**Shared state:** Import auth, storage, and VL components from `backend/deps.py`. Do NOT define module-level globals for shared instances in route files.

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

Common issues: `403` → wrong endpoint (EU vs US); `0 tokens` → check token counting in provider response conversion.

## Research Papers (DO NOT CONTRADICT)

1. **LegalBench-RAG** (Pipitone & Alami, 2024) — Legal document RAG benchmarks
2. **Contextual Retrieval** (Anthropic, 2024) — Context prepending

## Documentation

- [`README.md`](README.md) — Installation, quick start
- [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md) — Docker configuration
- [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) — Human-in-the-Loop
- [`docs/WEB_INTERFACE.md`](docs/WEB_INTERFACE.md) — Web UI features
