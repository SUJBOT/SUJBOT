# Changelog — 13.–15. února 2026

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
- Adaptivní k-retrieval: Otsu/GMM automatické score thresholding
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

## 9. Progresivní context compaction (uncommitted)
- 3-vrstvá správa kontextu: tool output pruning → LLM-summarized compaction → emergency truncation
- `ContextBudgetMonitor` — sledování token budgetu z API odpovědí
- Detekce duplicitních page images — opakované stránky nahrazeny textovou referencí
- Rozšíření adaptive retrieval window (fetch_k: 20→100, max_k: 10→20)
- Smazáno 13 obsoletních multi-agent/evaluation promptů

## 10. Dokumentace a ostatní
- Kompletní přepis dokumentace — smazáno 24 zastaralých souborů (~13k řádků), napsáno 5 nových (ARCHITECTURE, DEPLOYMENT, API, GRAPH_RAG, CONFIGURATION)
- Admin document management page (list/upload/delete/reindex se SSE progress) (PR #13)
- Audit hardening — error handling, resource management, data safety (PR #14)
- Benchmark criteria evaluation pipeline (`benchmark_criteria/`) — 2191 CZ otázek z jaderné bezpečnosti
- Design doc: parallel chat streaming (per-conversation streaming state)

---

**Statistiky:**
- 14 merged PRs (#13–#26)
- ~60 commitů
- Net: cca -29 000 řádků (hlavně díky OCR removal)
- Nové moduly: `src/graph/`, `rag_confidence/`, `src/vl/document_converter.py`, `benchmark_criteria/`
