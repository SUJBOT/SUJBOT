# Compliance Check Design

**Date:** 2026-02-15
**Status:** Approved
**Approach:** Community-Based Compliance Mapping (Approach A)

## Overview

Add compliance checking capability to SUJBOT by leveraging existing Graph RAG Leiden communities as "compliance domains". A new `compliance_check` agent tool enables users to check documents against regulations, perform gap analysis, cross-document mapping, and regulatory querying.

## Architecture

```
User query ("Splňuje dokument X zákon Y?")
  → Agent selects compliance_check tool
  → Community search (find relevant compliance domains)
  → Requirement extraction (OBLIGATION/PROHIBITION/PERMISSION entities)
  → Evidence matching (vector search in target document)
  → LLM assessment (per-requirement compliance evaluation)
  → Structured compliance report
```

## 1. Entity Type Extension

Add 5 new compliance-specific entity types to the existing 10:

| Type | Description | Example |
|---|---|---|
| `OBLIGATION` | Duty imposed by regulation | "Správce musí vést záznamy o činnostech zpracování" |
| `PROHIBITION` | Ban/restriction | "Zakazuje se zpracování zvláštních kategorií údajů" |
| `PERMISSION` | Right/exception | "Zpracování je povoleno se souhlasem subjektu" |
| `EVIDENCE` | Proof of compliance in internal document | "Kapitola 5 popisuje záznamy o zpracování" |
| `CONTROL` | Mechanism/measure for compliance | "Šifrování dat v klidu i při přenosu" |

**Changes:**
- `src/graph/entity_extractor.py`: add types to `ENTITY_TYPES` set
- `prompts/graph_entity_extraction.txt`: extend with new types, examples, and compliance-aware extraction instructions

**No DB migration needed** — `entity_type` is `TEXT`, no enum constraint.

## 2. New `compliance_check` Agent Tool

### Input Schema

```python
class ComplianceCheckInput(ToolInput):
    query: str                              # "Splňuje smlouva X zákon Y?"
    document_id: Optional[str] = None       # Specific document to check
    regulation_filter: Optional[str] = None # Filter to specific regulation entity
    community_level: int = 0                # 0=detailed, 1=broader, 2=top-level
    max_requirements: int = 20              # Cap on requirements to check
```

### Execution Flow

1. **Community search** — `graph_storage.search_communities(query, level)` finds relevant compliance domains
2. **Requirement extraction** — `graph_storage.get_community_entities(id)` filtered to `OBLIGATION | PROHIBITION | PERMISSION | REQUIREMENT` types
3. **Evidence matching** — `vector_store.search(requirement.description)` filtered by `document_id` if provided
4. **LLM assessment** — Prompt-based evaluation of evidence against each requirement; returns `MET | UNMET | PARTIAL | UNCLEAR`
5. **Report generation** — Structured `ToolResult` with per-requirement findings and aggregate scores

### Output Format

```python
ToolResult(
    success=True,
    data={
        "compliance_domain": str,       # Community title
        "regulation_source": str,       # Source regulation name
        "document_checked": str,        # Target document name
        "overall_score": float,         # 0.0–1.0 aggregate
        "findings": [
            {
                "requirement": str,           # Requirement text
                "requirement_type": str,      # OBLIGATION/PROHIBITION/etc.
                "source_entity_id": int,
                "status": "MET|UNMET|PARTIAL|UNCLEAR",
                "confidence": float,          # 0.0–1.0
                "evidence": Optional[str],    # Evidence text if found
                "evidence_source": Optional[str],  # Page/chunk ID
                "gap_description": Optional[str],  # If UNMET
            },
        ],
        "summary": {
            "total_requirements": int,
            "met": int,
            "unmet": int,
            "partial": int,
            "unclear": int,
        }
    }
)
```

## 3. Communities as Compliance Domains

Leiden communities naturally cluster related entities. With compliance entity types, communities restructure into compliance domains:

**Level 0 (fine-grained):**
- "GDPR - záznamy o zpracování" (OBLIGATION + CONTROL + EVIDENCE entities)
- "GDPR - práva subjektu" (OBLIGATION + PERMISSION, possibly no EVIDENCE = gap)
- "Bezpečnost - šifrování" (REQUIREMENT + CONTROL + EVIDENCE entities)

**Level 1 (broader):**
- "Ochrana osobních údajů" (merged GDPR communities)
- "Technická bezpečnost"

**Level 2 (top-level):**
- "Regulatorní compliance"

### Gap Detection via Communities

- Community with OBLIGATION/REQUIREMENT but **no** EVIDENCE/CONTROL = **compliance gap**
- `EVIDENCE count / OBLIGATION count` ratio per community = **domain compliance score**
- Cross-document entity relationships reveal which regulations are covered by which documents

## 4. New Prompt File

`prompts/compliance_assessment.txt` — LLM prompt for evaluating whether evidence satisfies a requirement.

Input: requirement text + evidence chunks
Output: status (MET/UNMET/PARTIAL/UNCLEAR) + confidence + explanation

## 5. Implementation Components

### Files to Create
- `src/agent/tools/compliance_check.py` — New agent tool
- `prompts/compliance_assessment.txt` — LLM assessment prompt

### Files to Modify
- `src/graph/entity_extractor.py` — Add 5 entity types to `ENTITY_TYPES`
- `prompts/graph_entity_extraction.txt` — Extend extraction prompt with compliance types
- `src/agent/tools/__init__.py` — Add `_safe_import("compliance_check")`

### Files Unchanged
- `src/graph/storage.py` — Existing methods sufficient
- `src/graph/embedder.py` — Same model
- `src/graph/community_detector.py` — Leiden works on any entity types
- `src/graph/community_summarizer.py` — Auto-summarizes new types
- DB schema — No migration needed

### One-Time Re-build
After implementation:
1. `scripts/graph_rag_build.py` — Re-extract with new entity types
2. `scripts/graph_embed_backfill.py` — Re-embed new entities
3. Communities auto-restructure via Leiden

## 6. Research References

- **RAGulating Compliance** (Jomraj et al., 2025) — Multi-agent KG + RAG for regulatory QA
- **Microsoft GraphRAG** (Edge et al., 2024) — Leiden community detection + hierarchical summaries
- **PuppyGraph Compliance** — Dual-chain Intent/Implementation model
- **MIT Automated Compliance Verification** (2025) — VER-LLM framework
- **Sleimi et al. (2024)** — Legal compliance with LLMs, 81% accuracy
