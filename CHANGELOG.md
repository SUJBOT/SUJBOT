# Changelog — 13.–16. února 2026

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
