# Changelog — 13.–19. února 2026

## 28. Remove routing runner, rag_confidence module, and dead code (19. února 2026)

### Cleanup
- **Remove RoutingAgentRunner** — 8B router → 30B worker architecture removed. `AgentAdapter` now uses `SingleAgentRunner` directly. Deleted `src/single_agent/routing_runner.py`, `prompts/agents/router.txt`, and `tests/agent/test_routing_runner.py`.
- **Delete `rag_confidence/` module** — QPP-based retrieval confidence scoring (standalone, by veselm73, was disabled). ~7,500 lines removed including models, tests, training scripts, and calibration data.
- **Delete `src/duplicate_detector.py`** — unused dead code.
- **Remove deprecated config fields** — `routing` section, OCR-era fields removed from `config_schema.py` and `config.json`.

### Backend
- **LangSmith config promoted** — from `multi_agent.langsmith` to top-level `langsmith` key. Backward-compat fallback reads old location with deprecation warning.
- **Web search disabling** — per-request `disabled_tools` set in runner replaces tool-level `web_search_enabled` guard. Dead `ToolConfig.web_search_enabled` field removed.
- **Disabled tool guard** — runner rejects LLM hallucinated calls to disabled tools with error feedback.
- Stale multi-agent/routing comments cleaned up across docstrings.

### Frontend
- `routing` SSE event handling removed (no longer emitted by backend).
- `webSearchEnabled` toggle lifted from `ChatInput` to `useChat` hook (persists across conversation switches).
- PDF side panel: `MemoizedPage` wrapper, text cache, pending scroll pattern for search performance.
- `'routing'` removed from `SSEEvent` TypeScript type union.
- i18n: routing-related keys removed from cs.json and en.json.

### Tests
- New tests for disabled-tool rejection logic and LangSmith config key propagation.
- Web search tests updated for removed `web_search_enabled` field.

## 27. 8B Router → 30B Thinking Worker architecture (19. února 2026)

### Architecture
- **New routing layer**: Fast 8B model (Qwen3-VL-8B-Instruct-FP8) classifies queries, delegates complex ones to 30B thinking model (Qwen3-VL-30B-A3B-Thinking) with controllable thinking budget.
- **Tool-based delegation**: Router gets a single virtual tool `delegate_to_thinking_agent` with `task_description`, `complexity` (simple/medium/complex/expert), and `thinking_budget` (low/medium/high/maximum → 1K/4K/16K/32K tokens).
- **Graceful fallback**: If 8B router fails, falls back to 30B directly — zero user impact.
- Simple queries (greetings, meta-questions) → 8B answers directly (~0.1s TTFT).
- Document/regulatory queries → 8B delegates to 30B with full RAG tool loop.

### Backend
- `src/single_agent/routing_runner.py` — `RoutingAgentRunner` wraps `SingleAgentRunner` via composition. `__getattr__` delegates `initialize()`, `get_tool_health()`, etc. transparently.
- `src/single_agent/runner.py` — Added `extra_llm_kwargs` parameter to `run_query()` for thinking budget passthrough.
- `src/agent/providers/factory.py` — Added `local_llm_8b` provider type (port 18082 via `LOCAL_LLM_8B_BASE_URL`).
- `backend/agent_adapter.py` — Conditional `RoutingAgentRunner` initialization when `routing.enabled=true`. New `routing` SSE event.
- `config.json` — New `routing` section (enabled, router_model, worker_model, thinking_budgets). New `qwen3-vl-8b-local` model in registry.
- `prompts/agents/router.txt` — Router system prompt for query classification.
- vLLM Qwen3 thinking control: `extra_body.chat_template_kwargs.enable_thinking` (disabled for router, enabled with budget for worker).

### Frontend
- `useChat.ts` — Handles `routing` SSE event (classifying/delegate/direct progress messages).
- `types/index.ts` — Added `routing` to SSE event union type.
- i18n strings for routing progress (cs/en).

### Configuration
- `routing.enabled`: Toggle routing on/off (currently `true` in config.json).
- `routing.thinking_budgets`: Token budgets per complexity level.
- `.env` — Added `LOCAL_LLM_8B_BASE_URL=http://localhost:8082/v1`.
- Backend deploy: Add `-e LOCAL_LLM_8B_BASE_URL=http://host.docker.internal:18082/v1`.

## 26. Deploy Qwen3-VL-8B-Instruct on gx10-fa34 (19. února 2026)

### Infrastructure
- **New vLLM container** (`vllm-qwen3vl-8b`) on gx10-fa34, coexisting with embedding server
- Model: Qwen/Qwen3-VL-8B-Instruct-FP8 (official FP8 quant, served as `Qwen/Qwen3-VL-8B-Instruct`), port 8082/18082
- Flags: `--max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 16 --gpu-memory-utilization 0.60 --enable-auto-tool-choice --tool-call-parser hermes`
- GPU: model 10.5 GiB + embedding 17 GiB = ~78 GiB / 119 GiB (~41 GiB headroom)
- SSH tunnel + socat bridge (18082) + UFW rule for Docker access

### FP8 benchmark results (5 iterations, 2 warmup)
- **Decode**: 20-23 tok/s (FP8 halves weight reads per token vs BF16's 11-13 tok/s → +73%)
- **TTFT**: 0.09-0.38s (prefix caching helps; BF16 was 0.15-0.66s)
- **KV cache**: 405K tokens, 12.35x concurrency at 32K context
- **E2E**: text_only ~2.1s, rag_5pages ~10.5s, tool_call ~1.9s
- **Think ratio**: 73-100% (Qwen3 Instruct still activates hybrid thinking)
- Quality: "nearly identical to BF16" per official Qwen benchmarks

### Benchmark script enhancements (`scripts/vllm_benchmark.py`)
- Added `--model` CLI argument (default: Qwen3-VL-30B) — configurable model name per endpoint
- Added `--compare-model` argument — different models on different endpoints for A/B comparison
- Labels now include model name for clarity in output tables

## 25. Upload modal with drag-and-drop + document access level (19. února 2026)

### New UploadModal component
- Full-screen modal with drag-and-drop file upload, replaces inline upload in DocumentBrowser
- Batch file selection (multiple files at once)
- Per-file category selector (legislation / documentation) with amber/blue toggle buttons
- New access level selector (secret / public) with lock/globe icons
- Progress bar with stage labels during upload
- i18n: full CZ/EN translations for all new labels

### Backend: access_level support
- `access_level` column added to `vectors.documents` (CHECK constraint: `public`/`secret`, default `public`)
- Upload endpoint accepts `access_level` form parameter
- Admin PATCH endpoint accepts optional `access_level` alongside `category`
- COALESCE-safe UPSERT — prevents NULL constraint violations when only one field is provided

### DocumentBrowser improvements
- Inline category toggle: click the category icon (Scale/BookOpen) to switch a document between legislation and documentation instantly
- Simplified to pure document browser — upload UI moved to UploadModal
- Click-outside handler respects open UploadModal (won't close browser while modal is open)

### API fixes
- `Accept: application/json` header added to all API calls — required for nginx admin route map
- New `updateDocumentCategory()` method for PATCH requests

### Cleanup
- Removed QPP-based search confidence scoring (disabled, not production-ready)
- Simplified search tool and tests (removed confidence bands, QPP feature extraction)
- Removed unused `postgres_adapter` search confidence methods
- Deleted `tests/agent/test_search_confidence.py`

## 24. Fix empty response with local_llm tool calling + 30B model restoration (19. února 2026)

### Bug fix: "Model returned empty response" when agent uses tools (local variant)
- **Root cause**: `ThinkTagStreamParser` emitted `text_delta: '\n\n'` between `</think>` and tool calls. This whitespace artifact prematurely cleared `agentProgress.currentAgent` in the frontend, triggering the "empty response" error display for the entire duration of tool execution (~2-3 minutes).
- **Backend fix** (`runner.py`): Added `text_delta_pending` buffer in `_stream_llm_iteration()`. Leading whitespace after `</think>` is buffered and only emitted when substantive (non-whitespace) content arrives. Whitespace-only text before tool calls is silently discarded.
- **Frontend fix** (`useChat.ts`, `ChatMessage.tsx`): `text_delta` handler no longer sets `isStreaming=false` — deferred to `done` event. Empty response check now also considers `agentProgress.isStreaming` and active tools.
- **Missing event** (`agent_adapter.py`): Added `tool_calls_summary` SSE event (was documented but never implemented). Frontend `hasToolCalls` now correctly reflects tool usage.

### 30B model restoration
- gx10-eb6e was running francji1's 235B AWQ model (stuck at NCCL init, needed TP=2 across 2 nodes). Restored Qwen3-VL-30B-A3B-Thinking with optimized flags + `--enable-auto-tool-choice --tool-call-parser hermes` for Hermes-style tool calling.
- Config reverted: `local` variant → `qwen3-vl-30b-local` (was briefly pointing to 235B).

### Re-embedding with local Qwen3-VL-Embedding-8B
- Completed full re-embedding of 517 VL pages with local embedder (4096-dim, gx10-fa34).
- Embedding server fixes: correct model class (`AutoModelForImageTextToText`), system instruction, last-token pooling, BF16→float32 conversion.
- Search quality verified: same-section 0.60-0.80, same-document 0.40-0.60, cross-document 0.25-0.45.

## 23. vLLM Throughput Optimization — 55% decode speedup (19. února 2026)

### Production vLLM (gx10-eb6e) flag changes
- `--max-model-len`: 262144 → **131072** (doubles KV cache from ~256K to 512K tokens, concurrency 1.97x → 3.91x)
- `--max-num-batched-tokens`: 2048 (default) → **8192** (fewer prefill chunks for long RAG prompts)
- `--max-num-seqs`: 4 → **8** (more scheduling flexibility)
- `--gpu-memory-utilization`: unchanged at 0.92 (0.95 exceeds free memory on unified arch)

### Benchmark results (baseline → optimized)
- **Decode throughput**: 14.0 → 21.7 tok/s (+55%)
- **TTFT**: 0.25 → 0.21s (−14%)
- **E2E (text_only)**: 52.4 → 26.6s (−49%)
- **E2E (rag_5pages)**: 101.4 → 42.8s (−58%)
- **Think ratio**: ~33% of tokens wasted on `<think>` blocks (unchanged, Qwen3 behavior)

### New scripts
- **`scripts/vllm_benchmark.py`**: Streaming benchmark measuring TTFT, decode throughput, e2e latency. 4 profiles (text_only, rag_5pages, rag_8pages, tool_call). Supports A/B comparison between endpoints.

## 22. Local Models on GB10 (DGX Spark) — zero API cost inference (18. února 2026)

### LLM Provider: local_llm
- **`DeepInfraProvider`**: Added `base_url` parameter for custom OpenAI-compatible endpoints (llama.cpp, vLLM). API key not required in local mode.
- **Provider factory**: New `local_llm` provider branch routes to `LOCAL_LLM_BASE_URL` env var (default: `http://localhost:18080/v1`).
- **Model registry**: Added `local_llm` to `LLM_PROVIDERS` frozenset. New `qwen3-vl-235b-local` model entry with `provider: local_llm`.
- **Config**: `"local"` variant now points to `qwen3-vl-235b-local` (GB10 llama.cpp server) instead of DeepInfra cloud.

### Local Embedding: LocalVLEmbedder
- **`src/vl/local_embedder.py`**: Drop-in replacement for `JinaClient`. Same interface (`embed_query`, `embed_image`, `embed_pages`, `close`). Calls local vLLM `/v1/embeddings` endpoint via httpx.
- **`create_vl_components()`**: New `"embedder"` config key — `"jina"` (default) or `"local"`. Embedder selection via `_create_embedder()` factory.
- **Config**: `vl.embedder = "local"`, `vl.local_embedding_url`, `vl.local_embedding_model` keys added.

### Infrastructure (GB10 DGX Spark)
- 2x NVIDIA GB10 (120 GB unified memory each, aarch64, CUDA 13.0)
- Qwen3-VL-235B-A22B-Instruct Q4_K_M (~142 GB) split across 2 nodes via llama.cpp RPC
- Qwen3-VL-Embedding-8B served via vLLM on second node
- SSH tunnels: `localhost:18080` (LLM), `localhost:18888` (embedding)

### Scripts
- **`scripts/reembed_vl_pages.py`**: Re-embed all VL pages with local or Jina embedder. Supports `--doc-id`, `--limit`, `--dry-run`, `--embedder local|jina`.

## 21. Codebase Audit — SSOT enforcement, dead code removal, PR review fixes (PR #33, 17. února 2026)

### Dead code removal (~16 800 řádků smazáno)
- **`src/multi_agent/`** celý adresář smazán (10 081 řádků) — legacy LangGraph multi-agent systém, nahrazen SingleAgentRunner
- **`src/benchmark/`** celý adresář smazán (1 691 řádků) — nepoužívaný benchmark modul
- **`src/utils/`** 5 mrtvých modulů: `api_clients.py`, `batch_api.py`, `metadata.py`, `persistence.py`, `statistics.py` (1 927 řádků)
- **`src/agent/rag_confidence.py`** (549 řádků), `src/agent/providers/tool_translator.py` — nahrazeny sdílenými moduly
- **`frontend/src/components/chat/AgentProgress.tsx`** — nepoužívaný komponent
- Mrtvé testy: `tests/multi_agent/`, `tests/test_benchmark_*`, `tests/test_evaluation_infrastructure.py`, `tests/utils/test_persistence.py`, `tests/agent/test_rag_confidence.py`

### SSOT konsolidace — nové sdílené moduly
- **`backend/deps.py`** — centralizovaný dependency injection (auth, postgres adapter, VL/graph komponenty, token extraction, cookie helper, login logika, PDF cache). Všechny route moduly importují odtud.
- **`src/agent/providers/openai_compat.py`** — sdílené Anthropic↔OpenAI konverzní helpery (`STOP_REASON_MAP`, `convert_tools_to_openai`, `convert_response_to_anthropic`, `convert_system_to_string`, `convert_assistant_blocks_to_openai`). Používá OpenAIProvider i DeepInfraProvider.
- **`src/agent/tools/adapter.py`** — ToolAdapter extrahován z odstraněného `multi_agent/tools/adapter.py`
- **`src/agent/tools/models.py`** — ToolExecution, ToolStats, ToolUsageMetrics extrahovány z `multi_agent/core/state.py`
- **`src/agent/observability.py`** — `setup_langsmith()` + `LangSmithIntegration` extrahováno z `multi_agent/observability/`
- **`src/utils/async_helpers.py`** — sdílený `run_async_safe()` (dříve duplikován v graph/storage.py a postgres_adapter.py) + `vec_to_pgvector()`
- **`src/utils/text_helpers.py`** — sdílený `strip_code_fences()` (dříve duplikován ve 3 graph modulech)
- **`src/storage/conversation_mixin.py`** — `ConversationStorageMixin` deduplikuje 170 řádků conversation CRUD (6 metod)
- **`src/graph/types.py`** — ENTITY_TYPES a RELATIONSHIP_TYPES konstanty (SSOT pro validaci)
- **`frontend/src/config.ts`** — jediný zdroj `API_BASE_URL` (nahrazuje 8+ inline definic)
- **`frontend/src/components/chat/AttachmentChip.tsx`** — sdílený attachment chip komponent

### Bug fixy a vylepšení
- **OpenAI provider**: Kritický fix `STOP_REASON_MAP` — `"tool_calls"` → `"tool_use"` (agent nedostával tool results)
- **OpenAI provider**: Fix `_is_o_series()` — regex s word boundaries místo substring match (false-positive prevence)
- **OpenAI compat**: Malformované tool calls se zachovávají s `{}` inputem místo tichého zahazování (agent může retryovat)
- **Gemini provider**: Unikátní tool call IDs (UUID), model caching přes `_get_model()`, extrahovaný `_prepare_request()`
- **Backend main.py**: Smazán neautentizovaný DELETE endpoint, `sanitize_error()` v SSE, title lock fix, DB transakce
- **Backend middleware**: `AuthMiddleware` odstraněn, `middleware/auth.py` deleguje na `deps.py` (SSOT)
- **Exchange rate**: Fix exception hierarchy (`TimeoutException` před `HTTPError`)
- **Jina client**: Persistentní `httpx.Client` s `close()` metodou (connection pooling)
- **VL page store**: Instance dict cache místo `lru_cache` na instančních metodách
- **Graph storage**: `asyncio.Lock` v `_ensure_pool()` (thread-safe double-check locking), batch entity ID resolution
- **Adapter**: `get_tool_schema()` — zúžený exception catch na `(AttributeError, TypeError, KeyError)`

### Frontend
- **VITE_API_URL → VITE_API_BASE_URL**: Fix tichého nesouladu v PDFSidePanel a CitationContext
- **ErrorBoundary, ToolCallDisplay**: i18n překlady (cs.json + en.json)
- **API service**: `credentials:'include'` na deleteMessage/streamClarification, nové getAgentVariant/setAgentVariant metody

### Verifikace
- 403 testů pass, 0 failures
- Frontend TypeScript: čistý build, žádné chyby
- PR review (4 agenti): code-reviewer, silent-failure-hunter, test-analyzer, comment-analyzer — všechny kritické nálezy opraveny

## 20. Web Search Tool — Gemini Google Search grounding (17. února 2026)
- **New `web_search` tool**: Internet search via Gemini's native Google Search grounding. Last-resort tool for questions requiring current/external information not in the document corpus.
- **Backend**: `src/agent/tools/web_search.py` — `WebSearchTool` with `@register_tool`, Pydantic input validation, grounding metadata extraction (sources with URLs + titles)
- **Config**: `ToolConfig.web_search_enabled` / `web_search_model` fields, wired from `config.json` → `agent_tools.web_search`
- **System prompt** (`prompts/agents/unified.txt`): Tool added to table + "Web search (last resort)" guidance section
- **Frontend citations**: `\webcite{url}{title}` syntax → `<webcite>` HTML tags → `WebCitationLink` component (blue badge with `ExternalLink` icon, opens URL in new tab)
- **Dependency**: `google-genai>=1.0` added to `pyproject.toml` (coexists with `google-generativeai`)
- **Tests**: `tests/agent/test_web_search.py` — 12 tests covering input validation, disabled state, missing API key, successful search, no sources, API errors, citation format, config defaults

## 19. KG extrakce & deduplikace — benchmark-driven vylepšení (17. února 2026)
- **Manuální GT benchmark** (2×5 stran z atomového zákona): Claude jako anotátor vytvořil ground-truth KG z obrázků stránek, porovnání s pipeline výstupem
- **7 systematických problémů** identifikováno v extrakci: překlepy, verbose názvy, typ v názvu, špatné entity typy, špatné relationship typy, chybějící entity, vágní CONTROL entity
- **Extraction prompt** (`prompts/graph_entity_extraction.txt`) — 3 nová pravidla:
  - "Extract the CORE concept, not qualified variants" — zabraňuje kontextuální přespecifikaci (e.g., "fyzikální spouštění" místo "fyzikální spouštění podle vnitřních předpisů")
  - Příklady WRONG/RIGHT pro concise naming, relationship vyjádření kvalifikátorů
  - Konzistentní s `prompts/graph_gt_text_extraction.txt`
- **Benchmark výsledky:** OLD prompt: 53 issues → NEW prompt: 0 issues (100% eliminace na 2. vzorku)
- **Dedup pipeline** (`scripts/graph_normalize_dedup.py`) — 2 nové fáze:
  - **Phase 1b** (substring containment): merguje entity kde kratší název ⊂ delší název (same type+doc), s LLM potvrzením. Filtruje: min 2 slova, min 40% délky, skip SANCTION/DEADLINE/SECTION/REQUIREMENT
  - **Phase 2** rozšířen o **word-overlap trigger**: kromě trigram sim > 0.75 také hledá páry s word Jaccard overlap > 0.6 (trigram sim 0.3–0.75), pokrývá případy kde trigram selhává kvůli rozdílu délek
  - Phase 2 filtruje substring páry (handled v Phase 1b) aby neduplikoval práci
- **Dedup LLM prompt** (`prompts/graph_entity_dedup.txt`) — 4 nová pravidla:
  - Core concept vs qualified variant = duplicáty (keep shorter)
  - Compound aktivity vs constituenty = duplicáty ("údržba a opravy" = "údržba")
  - České morfologické varianty = duplicáty (genitiv vs lokativ)
  - Různé částky sankcí ≠ duplicáty
- GT data: `data/esbirka_benchmark/gt_manual.json` (v1, 5 stran), `gt_manual_v2.json` (v2, 5 různých stran)

## 18. e-Sbírka benchmark — GT dataset & KG pipeline evaluace (17. února 2026)
- Nový skript `scripts/esbirka_gt_dataset.py` — stahuje strukturovaná data z e-Sbírka REST API (`www.e-sbirka.cz/sbr-cache`), extrahuje GT entity a vztahy z metadat, fragmentů a souvislostí, stahuje PDF
- Nový skript `scripts/esbirka_pipeline_extract.py` — spouští existující VL+KG pipeline na benchmark PDF (PageStore rendering + EntityExtractor), in-memory dedup bez DB
- Nový skript `scripts/esbirka_compare.py` — 4-fázový srovnávací skript (language-agnostic):
  - Phase 1: Exact normalized match + citation number match (strukturální)
  - Phase 2a: Semantic embedding match (multilingual-e5-small, same-type, threshold 0.75)
  - Phase 2b: Cross-type semantic match (different types, threshold 0.85)
  - Phase 3: LLM judge pro borderline případy (optional `--llm-judge`)
  - Relationship matching: semantic embedding na celých triplech (source → type → target)
- Benchmark dokumenty: zákon č. 263/2016 Sb. (atomový zákon) + vyhláška č. 422/2016 Sb. (radiační ochrana) — sdílené entity (SÚJB, jaderné zařízení) pro test deduplikace
- GT dataset: 1 961 entit, 1 223 vztahů, 42 očekávaných dedup skupin
- Pipeline výsledky (Haiku 4.5, 165 stran): 2 444 entit, 1 973 vztahů
- **Výsledky srovnání:** Entity F1=71.2% (R=98.3%, P=55.8%), Relationship F1=58.3% (R=97.3%, P=41.7%), Dedup rate=52.4%
- Data v `data/esbirka_benchmark/` (GT JSON, pipeline JSON, PDF, page images)

## 17. Knowledge Graph — legislativní ontologie (16. února 2026)
- 4 nové entity typy: DEFINITION (formální právní definice), SANCTION (sankce za porušení), DEADLINE (lhůty a přechodná období), AMENDMENT (novelizace předpisu)
- 5 nových relationship typů: SUPERSEDES (zrušení předpisu), DERIVED_FROM (prováděcí předpis z nadřazeného), HAS_SANCTION (vazba povinnost→sankce), HAS_DEADLINE (vazba povinnost→lhůta), COMPLIES_WITH (důkaz→požadavek)
- Rozšířený extraction prompt s popisy nových typů a příklady
- Rozšířený dedup prompt o pravidla pro SANCTION, AMENDMENT, DEFINITION entity
- `graph_search` tool zobrazuje nové typy v entity_type filtru
- `compliance_check` tool obohacuje findings o sankce přes HAS_SANCTION vztahy
- Žádné změny DB schématu — entity_type a relationship_type jsou TEXT sloupce

## 15. PDF search — diakritika, zvýrazňování, scroll (PR #30, 16. února 2026)
- Diacritics-insensitive full-text search v PDF preview: `normalizeText()` stripuje LaTeX spacing modifiers (ˇ˘˙˚˛˜˝ U+02C7–U+02DD) s okolním whitespace, poté NFD + combining marks
- Počítání výskytů místo stránek: "1 / 12" namísto "1 / 3 pages"
- DOM-based cross-span highlighting: post-processing text layeru s position mapping (`normToOrig[]`) — funguje i pro LaTeX PDF kde je slovo rozloženo přes více TextItems
- CSS `:has(.search-hl)` override opacity text layeru (react-pdf default 0.2 → 1.0) pro viditelné oranžové zvýraznění
- Přímý `scrollTop` výpočet namísto nespolehlivého `scrollIntoView` v nested overflow kontejnerech; dvou-fázový scroll (rough page → refined highlight)
- Split efektů: `[documentId]` reset vs `[initialPage]` navigace — oprava bugu kdy kliknutí na jinou citaci neposouvalo PDF
- Odstraněn citation hover preview tooltip (`CitationPreview.tsx` smazán)

## 16. Nginx cache-control fix (PR #31, 16. února 2026)
- `Cache-Control: no-store, no-cache, must-revalidate` na SPA fallback `location /` v frontend nginx
- Prevence `Failed to fetch dynamically imported module` po rebuildu frontendu (prohlížeč cachoval starý `index.html` s neexistujícími chunk hashi)

## 0. Integrace QPP retrieval confidence do search pipeline (15. února 2026)
- `score_retrieval_general()` z `rag_confidence/` integrován do `SearchTool`
- Nová metoda `PostgresVectorStoreAdapter.get_all_vl_similarities()` — vrací cosine similarity ke VŠEM stránkám (pro QPP feature extraction)
- Nová metoda `VLRetriever.search_with_embedding()` — vrací výsledky + query embedding (bez redundantního Jina API callu)
- QPP confidence (0.0–1.0) přidáno do `result.metadata["retrieval_confidence"]` a `citations`
- Systémový prompt rozšířen o interpretaci confidence bandů (HIGH/MEDIUM/LOW/VERY_LOW)
- Image search QPP přeskakuje (není textový dotaz pro QPP features)
- Best-effort: selhání QPP neblokuje search — confidence je jen informativní
- 12 nových testů v `tests/agent/test_search_confidence.py`

## 1. Graph RAG — znalostní graf (PR #15)
- Nový modul `src/graph/` — PostgreSQL knowledge graph s Leiden komunitami (igraph)
- Extrakce entit a vztahů ze stránkových obrázků přes multimodální LLM
- 3 nové agentní nástroje: `graph_search`, `graph_context`, `graph_communities`
- Optimalizace systémového promptu pro automatické využívání graph nástrojů
- Opravy: asyncio event loop mismatch, pool leak, CancelledError propagace

## 2. Sémantické vyhledávání v grafu (PR #17)
- `GraphEmbedder` — multilingual-e5-small (384-dim) pro cross-language dotazy (EN→CZ)
- HNSW indexy na embedding sloupce v graph tabulkách
- FTS fallback (PostgreSQL full-text search) bez embedderu
- Backfill skripty pro embeddingy a FTS migraci

## 3. Compliance check — kontrola shody (PR #18)
- Nový nástroj `compliance_check` — hodnocení shody dokumentů s regulatorními požadavky
- 5 nových typů entit: OBLIGATION, PROHIBITION, PERMISSION, EVIDENCE, CONTROL
- Kategorie dokumentů (`documentation` / `legislation`) — filtrace při vyhledávání
- VL mód: načítání page images jako multimodální bloky pro LLM assessment
- Prompt šablona pro compliance assessment (MET/UNMET/PARTIAL/UNCLEAR)

## 4. Odstranění OCR architektury (PR #19)
- Kompletní refaktoring na VL-only — smazáno 40+ OCR souborů, 29 203 řádků kódu netto
- Zjednodušení storage vrstvy: odstraněn layer1/2/3, MetadataFilter, BM25, hierarchical_search
- Zjednodušení agentních nástrojů (search, expand_context, get_document_info)
- 52 nových unit testů pro VL nástroje
- Archivováno v branch `archive/ocr-implementation` a tagu `v1.0-ocr`

## 5. Paralelní indexovací pipeline + kategorie (PR #21)
- Embedding (Jina API) a summarizace (LLM) běží souběžně přes asyncio tasks
- Dialog pro výběr kategorie dokumentu (legislation/documentation) před uploadem
- Cancel tlačítko během uploadu s cleanup (soubory + DB)
- Opravy: TOCTOU race v queue drain, cleanup deduplikace, UI state management

## 6. Deduplikace entit v grafu + automatický rebuild (PR #22, #23, #25)
- Exact dedup (case-insensitive `lower(name)`) + sémantický dedup (LLM arbitr, Union-Find)
- Automatický debounced rebuild komunit po upload/delete/reindex (10s delay)
- `post_processor.py`: exact dedup → embed → semantic dedup → community detection → summarize → save
- Strukturované JSON prompty pro dedup verdict a community summary
- Kanonická jména: při merge se vybírá nejdelší jméno (plný název > zkratka)
- 35+ nových testů (parsing, transitive closure, failure handling)

## 7. Multi-formátová podpora dokumentů (PR #24)
- Upload DOCX, TXT, Markdown, HTML, LaTeX — konverze do PDF před VL pipeline
- `src/vl/document_converter.py`: LibreOffice (DOCX), pdflatex (LaTeX), PyMuPDF fitz.Story (HTML/MD/TXT)
- Chat přílohy: obrázky jako multimodální bloky, PDF → page images, text dokumenty → extrakce textu
- Frontend: attachment chips se jménem souboru, ikonou a velikostí
- Image search: `search` tool podporuje `image_attachment_index` a `image_page_id`
- Docker: přidány balíčky libreoffice-writer + texlive
- 25 nových unit testů pro document converter

## 8. RAG confidence modul (PR #26)
- Cherry-pick `rag_confidence/` z branch `dev/matyas` (autor: veselm73)
- QPP-based retrieval confidence scoring (24 language-agnostic features, MLP)
- Conformal prediction s kalibrovaným prahem (τ=0.711, 90% coverage)
- Trénovaný GeneralQPP model na ViDoRe V3 (AUROC 0.771, AUPRC 0.855)

## 9. Progresivní context compaction 
- 3-vrstvá správa kontextu: tool output pruning → LLM-summarized compaction → emergency truncation
- `ContextBudgetMonitor` — sledování token budgetu z API odpovědí
- Detekce duplicitních page images — opakované stránky nahrazeny textovou referencí
- Rozšíření adaptive retrieval window (fetch_k: 20→100, max_k: 10→20)
- Smazáno 13 obsoletních multi-agent/evaluation promptů

## 10. Adaptivní k-retrieval (PR #19)
- Nový modul `src/retrieval/adaptive_k.py` — automatické určení počtu výsledků na základě distribuce skóre
- Otsu thresholding: maximalizace inter-class variance pro nalezení přirozeného cutoffu mezi relevantními a irelevantními výsledky
- GMM (Gaussian Mixture Model): 2-komponentní Gaussovská směs, práh na váženém průměru středních hodnot
- Unimodal fallback: pokud je rozsah skóre < `score_gap_threshold` (0.05), vrací `min_k` výsledků
- Konfigurovatelné meze: `min_k` (1), `max_k` (10), `fetch_k` (20), `min_samples_for_adaptive` (3)
- Sdíleno VL search i graph search — čistá score analýza bez vazby na PostgreSQL/Jina
- 16 testů v `tests/retrieval/test_adaptive_k.py` (bimodal separation, unimodal fallback, bounds enforcement, GMM, edge cases)

## 11. Admin — správa dokumentů (PR #13)
- Kompletní admin stránka `DocumentsPage.tsx` — seznam dokumentů s metadaty (počet stran, velikost, datum indexace, kategorie)
- Upload dokumentu s výběrem kategorie (legislation/documentation) a SSE progress streaming
- Delete dokumentu — kaskádový mazání vektorů, page images, PDF, graph dat, kategorie
- Reindex — přeindexování stávajícího dokumentu se SSE progress
- Inline editace kategorie dokumentu (Select dropdown)
- Bezpečnost: regex validace `document_id`, ochrana proti path traversal, žádné raw exception v API responses
- Nginx SSE routing pro `text/event-stream` Content-Type

## 12. Admin — vizualizace znalostního grafu
- Interaktivní `GraphPage.tsx` — Sigma.js (WebGL) + graphology pro vykreslení až 500 uzlů
- Backend endpointy: `GET /admin/graph/overview` (statistiky + filtrovací seznamy), `GET /admin/graph/data` (uzly + hrany)
- Filtrování podle dokumentu, komunity a typu entity (Autocomplete multi-select)
- ForceAtlas2 physics layout (gravitace, Barnes-Hut optimalizace pro >100 uzlů)
- Velikost uzlů škálovaná podle degree (počet spojení)
- Barevná paleta pro 15+ typů entit (REGULATION, ORGANIZATION, PERSON, CONCEPT, ...)
- Vyhledávání: highlight matching uzlů, dimming ostatních, auto-focus kamery
- Tooltip s detaily při hoveru nad uzlem/hranou (jméno, typ, popis, dokument, váha)
- Sidebar s taby: seznam uzlů seskupených podle typu / seznam komunit
- GEXF export pro import do Gephi

## 13. Paralelní konverzace (per-user sessions)
- Backend plně izoluje requesty — každý SSE stream běží nezávisle
- Per-conversation streaming: `streamingConversationIds` Set + `streamingRefsMap` (Map<conversationId, StreamingState>)
- Uživatel může odesílat zprávy do více konverzací současně — guard blokuje pouze double-send do téže konverzace
- Sidebar: animovaný spinner u každé konverzace, která právě streamuje (podpora více spinnerů naráz)
- Smazání konverzace během streamování automaticky abortuje její stream
- `beforeunload`: abort VŠECH aktivních streamů při refreshi/zavření stránky
- `cancelStreaming` přijímá volitelné `conversationId` pro cílený cancel
- Architektura: SSE (Server-Sent Events), nikoliv WebSocket — stream nepřežije refresh stránky
- Design doc: `docs/plans/2026-02-15-parallel-chat-streaming-design.md`

## 14. Dokumentace a ostatní
- Kompletní přepis dokumentace — smazáno 24 zastaralých souborů (~13k řádků), napsáno 5 nových (ARCHITECTURE, DEPLOYMENT, API, GRAPH_RAG, CONFIGURATION)
- Audit hardening — error handling, resource management, data safety (PR #14)
- Benchmark criteria evaluation pipeline (`benchmark_criteria/`) — 2191 CZ otázek z jaderné bezpečnosti

---

**Statistiky:**
- 18 merged PRs (#13–#31)
- ~65 commitů
- Net: cca -29 000 řádků (hlavně díky OCR removal)
- Nové moduly: `src/graph/`, `rag_confidence/`, `src/retrieval/`, `src/vl/document_converter.py`, `benchmark_criteria/`
- Nové skripty: `scripts/esbirka_gt_dataset.py`, `scripts/esbirka_pipeline_extract.py`, `scripts/esbirka_compare.py`
