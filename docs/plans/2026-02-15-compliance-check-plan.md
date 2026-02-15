# Compliance Check Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `compliance_check` agent tool that leverages Graph RAG Leiden communities as compliance domains to check documents against regulations, perform gap analysis, and produce structured compliance reports.

**Architecture:** Extend entity types (5 new compliance types), update extraction prompt, create a new agent tool that chains community search → requirement extraction → evidence matching → LLM assessment → structured report. No DB migration needed.

**Tech Stack:** Python, Pydantic, asyncpg (existing GraphStorageAdapter), pytest + anyio

**Design doc:** `docs/plans/2026-02-15-compliance-check-design.md`

---

### Task 1: Extend Entity Types in Entity Extractor

**Files:**
- Modify: `src/graph/entity_extractor.py:27-38`

**Step 1: Write the failing test**

Create test file `tests/graph/test_entity_extractor_types.py`:

```python
"""Tests for entity extractor type validation with compliance types."""

import pytest
from src.graph.entity_extractor import ENTITY_TYPES, RELATIONSHIP_TYPES


class TestEntityTypes:
    """Verify compliance entity types are registered."""

    def test_compliance_entity_types_exist(self):
        """All 5 compliance entity types must be in ENTITY_TYPES."""
        compliance_types = {"OBLIGATION", "PROHIBITION", "PERMISSION", "EVIDENCE", "CONTROL"}
        for etype in compliance_types:
            assert etype in ENTITY_TYPES, f"Missing compliance entity type: {etype}"

    def test_original_entity_types_preserved(self):
        """Original 10 types must still exist after extension."""
        original_types = {
            "REGULATION", "STANDARD", "SECTION", "ORGANIZATION", "PERSON",
            "CONCEPT", "REQUIREMENT", "FACILITY", "ROLE", "DOCUMENT",
        }
        for etype in original_types:
            assert etype in ENTITY_TYPES, f"Missing original entity type: {etype}"

    def test_total_entity_type_count(self):
        """Should have exactly 15 entity types (10 original + 5 compliance)."""
        assert len(ENTITY_TYPES) == 15

    def test_relationship_types_unchanged(self):
        """Relationship types should not be modified."""
        assert len(RELATIONSHIP_TYPES) == 9
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/graph/test_entity_extractor_types.py -v`
Expected: FAIL — `OBLIGATION` not in `ENTITY_TYPES`

**Step 3: Add compliance entity types**

In `src/graph/entity_extractor.py`, replace lines 27-38 `ENTITY_TYPES` set:

```python
ENTITY_TYPES = {
    "REGULATION",
    "STANDARD",
    "SECTION",
    "ORGANIZATION",
    "PERSON",
    "CONCEPT",
    "REQUIREMENT",
    "FACILITY",
    "ROLE",
    "DOCUMENT",
    # Compliance-specific types
    "OBLIGATION",
    "PROHIBITION",
    "PERMISSION",
    "EVIDENCE",
    "CONTROL",
}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/graph/test_entity_extractor_types.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/graph/entity_extractor.py tests/graph/test_entity_extractor_types.py
git commit -m "feat: add 5 compliance entity types to graph extractor"
```

---

### Task 2: Update Entity Extraction Prompt

**Files:**
- Modify: `prompts/graph_entity_extraction.txt`

**Step 1: Update the extraction prompt**

Replace `prompts/graph_entity_extraction.txt` with the extended version. The existing prompt is at the file root — it lists entity types (lines 1-13), relationship types (lines 15-24), rules (lines 26-33), and output format (lines 35-47).

Add 5 new entity types after the existing 10 (after line 13, before `## Relationship types`):

```
- OBLIGATION: Duties imposed by regulation — what someone MUST do (e.g., "správce musí vést záznamy", "povinnost hlášení")
- PROHIBITION: Bans or restrictions — what is FORBIDDEN (e.g., "zakazuje se zpracování zvláštních kategorií", "nesmí odmítnout")
- PERMISSION: Rights, exceptions, allowances (e.g., "zpracování je povoleno se souhlasem", "oprávnění k výkonu")
- EVIDENCE: Proof of compliance found in documents — chapters, sections, or statements that demonstrate compliance (e.g., "Kapitola 5 popisuje záznamy o zpracování")
- CONTROL: Mechanisms, measures, or procedures for compliance (e.g., "šifrování dat v klidu", "pravidelný audit", "systém řízení jakosti")
```

Also update the description of REQUIREMENT to differentiate from OBLIGATION:
```
- REQUIREMENT: Specific technical or procedural requirements — measurable criteria (e.g., "limit dávky 20 mSv/rok", "minimální tloušťka stěny 5 mm")
```

**Step 2: Verify prompt is syntactically correct**

Run: `python -c "from pathlib import Path; p = Path('prompts/graph_entity_extraction.txt'); t = p.read_text(); print(f'Prompt length: {len(t)} chars'); assert 'OBLIGATION' in t; assert 'PROHIBITION' in t; assert 'PERMISSION' in t; assert 'EVIDENCE' in t; assert 'CONTROL' in t; print('All 5 compliance types found in prompt')"`
Expected: "All 5 compliance types found in prompt"

**Step 3: Commit**

```bash
git add prompts/graph_entity_extraction.txt
git commit -m "feat: extend extraction prompt with compliance entity types"
```

---

### Task 3: Create Compliance Assessment Prompt

**Files:**
- Create: `prompts/compliance_assessment.txt`

**Step 1: Write the compliance assessment prompt**

Create `prompts/compliance_assessment.txt`:

```
You are a legal compliance assessor. Given a regulatory requirement and evidence from a document, assess whether the document satisfies the requirement.

## Requirement
Type: {requirement_type}
Text: {requirement_text}
Source: {requirement_source}

## Evidence
{evidence_text}

## Instructions
Assess whether the evidence satisfies the requirement. Consider:
1. Does the evidence directly address the requirement?
2. Is the evidence complete or only partial?
3. Are there gaps or ambiguities?

## Output format
Respond with ONLY a JSON object:

```json
{
  "status": "MET|UNMET|PARTIAL|UNCLEAR",
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation in the same language as the requirement"
}
```

Status definitions:
- MET: Evidence clearly satisfies the requirement
- UNMET: No evidence found or evidence contradicts the requirement
- PARTIAL: Evidence partially addresses the requirement but gaps exist
- UNCLEAR: Insufficient evidence to determine compliance status
```

**Step 2: Verify prompt loads correctly**

Run: `python -c "from pathlib import Path; p = Path('prompts/compliance_assessment.txt'); t = p.read_text(); assert '{requirement_type}' in t; assert '{evidence_text}' in t; print('Prompt OK')"`
Expected: "Prompt OK"

**Step 3: Commit**

```bash
git add prompts/compliance_assessment.txt
git commit -m "feat: add compliance assessment LLM prompt"
```

---

### Task 4: Create Compliance Check Tool — Input Schema and Skeleton

**Files:**
- Create: `src/agent/tools/compliance_check.py`
- Create: `tests/agent/tools/test_compliance_check.py`

**Step 1: Write failing tests for the tool skeleton**

Create `tests/agent/tools/test_compliance_check.py`:

```python
"""Tests for compliance_check agent tool."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.agent.tools.compliance_check import ComplianceCheckTool, ComplianceCheckInput
from src.agent.tools._base import ToolResult


@dataclass
class MockToolConfig:
    """Minimal ToolConfig mock for testing."""
    graph_storage: object = None
    compliance_threshold: float = 0.7


class TestComplianceCheckInput:
    """Test input schema validation."""

    def test_minimal_input(self):
        """Query is the only required field."""
        inp = ComplianceCheckInput(query="Splňuje dokument GDPR?")
        assert inp.query == "Splňuje dokument GDPR?"
        assert inp.document_id is None
        assert inp.community_level == 0
        assert inp.max_requirements == 20

    def test_full_input(self):
        """All fields can be provided."""
        inp = ComplianceCheckInput(
            query="compliance check",
            document_id="doc_123",
            regulation_filter="GDPR",
            community_level=1,
            max_requirements=10,
        )
        assert inp.document_id == "doc_123"
        assert inp.regulation_filter == "GDPR"
        assert inp.community_level == 1
        assert inp.max_requirements == 10

    def test_community_level_bounds(self):
        """Community level must be 0-2."""
        with pytest.raises(Exception):
            ComplianceCheckInput(query="test", community_level=3)
        with pytest.raises(Exception):
            ComplianceCheckInput(query="test", community_level=-1)

    def test_max_requirements_bounds(self):
        """max_requirements must be 1-50."""
        with pytest.raises(Exception):
            ComplianceCheckInput(query="test", max_requirements=0)
        with pytest.raises(Exception):
            ComplianceCheckInput(query="test", max_requirements=51)


class TestComplianceCheckToolNoGraph:
    """Test tool behavior when graph storage is not available."""

    @pytest.fixture
    def tool(self):
        """Create tool with no graph storage."""
        return ComplianceCheckTool(
            vector_store=MagicMock(),
            embedder=MagicMock(),
            config=MockToolConfig(graph_storage=None),
        )

    def test_no_graph_returns_error(self, tool):
        """Tool should return error when graph_storage is None."""
        result = tool.execute(query="test compliance")
        assert result.success is False
        assert "not available" in result.error.lower() or "not configured" in result.error.lower()


class TestComplianceCheckToolWithGraph:
    """Test tool behavior with mocked graph storage."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock graph storage."""
        graph = MagicMock()
        graph.search_communities.return_value = [
            {
                "community_id": 1,
                "level": 0,
                "title": "Ochrana osobních údajů",
                "summary": "GDPR requirements for data protection",
                "entity_ids": [10, 11, 12],
                "metadata": {},
            }
        ]
        graph.get_community_entities.return_value = [
            {
                "entity_id": 10,
                "name": "Povinnost vést záznamy",
                "entity_type": "OBLIGATION",
                "description": "Správce musí vést záznamy o činnostech zpracování",
                "document_id": "reg_gdpr",
            },
            {
                "entity_id": 11,
                "name": "Šifrování dat",
                "entity_type": "CONTROL",
                "description": "Šifrování dat v klidu i při přenosu",
                "document_id": "doc_policy",
            },
            {
                "entity_id": 12,
                "name": "Zákaz zpracování zvláštních kategorií",
                "entity_type": "PROHIBITION",
                "description": "Zakazuje se zpracování zvláštních kategorií údajů bez výjimky",
                "document_id": "reg_gdpr",
            },
        ]
        return graph

    @pytest.fixture
    def tool(self, mock_graph):
        """Create tool with mocked graph storage."""
        return ComplianceCheckTool(
            vector_store=MagicMock(),
            embedder=MagicMock(),
            llm_provider=MagicMock(),
            config=MockToolConfig(graph_storage=mock_graph),
        )

    def test_communities_searched(self, tool, mock_graph):
        """Tool should search communities for the query."""
        # Mock LLM provider to return a valid assessment
        tool.llm_provider.create_message.return_value = MagicMock(
            text='{"status": "MET", "confidence": 0.9, "explanation": "Evidence found"}'
        )
        # Mock VL search (returns empty for simplicity)
        tool.vl_retriever = None
        tool.page_store = None

        result = tool.execute(query="GDPR compliance")
        assert result.success is True
        mock_graph.search_communities.assert_called_once()

    def test_requirements_extracted_from_community(self, tool, mock_graph):
        """Tool should extract OBLIGATION/PROHIBITION/PERMISSION/REQUIREMENT entities."""
        tool.llm_provider.create_message.return_value = MagicMock(
            text='{"status": "MET", "confidence": 0.9, "explanation": "Evidence found"}'
        )
        tool.vl_retriever = None
        tool.page_store = None

        result = tool.execute(query="GDPR compliance")
        assert result.success is True
        # Should have called get_community_entities for the found community
        mock_graph.get_community_entities.assert_called_with(1)

    def test_output_structure(self, tool, mock_graph):
        """Result data should have the expected compliance report structure."""
        tool.llm_provider.create_message.return_value = MagicMock(
            text='{"status": "MET", "confidence": 0.9, "explanation": "Evidence found"}'
        )
        tool.vl_retriever = None
        tool.page_store = None

        result = tool.execute(query="GDPR compliance")
        assert result.success is True
        data = result.data
        assert "findings" in data
        assert "summary" in data
        assert "overall_score" in data
        summary = data["summary"]
        assert "total_requirements" in summary
        assert "met" in summary
        assert "unmet" in summary

    def test_no_communities_found(self, tool, mock_graph):
        """Tool should handle empty community search results."""
        mock_graph.search_communities.return_value = []
        result = tool.execute(query="nonexistent regulation")
        assert result.success is True
        assert result.data["findings"] == []
        assert result.data["summary"]["total_requirements"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/agent/tools/test_compliance_check.py -v`
Expected: FAIL — `ImportError: cannot import 'ComplianceCheckTool'`

**Step 3: Write the tool skeleton**

Create `src/agent/tools/compliance_check.py`:

```python
"""
Compliance Check Tool — community-based compliance assessment.

Leverages Graph RAG Leiden communities as compliance domains to check
documents against regulations and produce structured compliance reports.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)

_ASSESSMENT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "compliance_assessment.txt"
)

# Entity types that represent compliance requirements
_REQUIREMENT_TYPES = {"OBLIGATION", "PROHIBITION", "PERMISSION", "REQUIREMENT"}


class ComplianceCheckInput(ToolInput):
    """Input for compliance check tool."""

    query: str = Field(
        ...,
        description=(
            "Compliance question or topic to check "
            "(e.g., 'Splňuje dokument GDPR?', 'data protection requirements')"
        ),
    )
    document_id: Optional[str] = Field(
        None,
        description="Document ID to check compliance for. If omitted, checks across all documents.",
    )
    regulation_filter: Optional[str] = Field(
        None,
        description="Filter to specific regulation entity name (e.g., 'zákon č. 263/2016 Sb.')",
    )
    community_level: int = Field(
        0,
        description="Community hierarchy level (0=detailed, 1=broader, 2=top-level)",
        ge=0,
        le=2,
    )
    max_requirements: int = Field(
        20,
        description="Maximum number of requirements to check",
        ge=1,
        le=50,
    )


@register_tool
class ComplianceCheckTool(BaseTool):
    """Check document compliance against regulations using knowledge graph communities."""

    name = "compliance_check"
    description = (
        "Check document compliance against regulations — uses knowledge graph communities "
        "as compliance domains. Returns structured findings with MET/UNMET/PARTIAL/UNCLEAR "
        "status per requirement and overall compliance score."
    )
    input_schema = ComplianceCheckInput

    def execute_impl(
        self,
        query: str,
        document_id: Optional[str] = None,
        regulation_filter: Optional[str] = None,
        community_level: int = 0,
        max_requirements: int = 20,
    ) -> ToolResult:
        graph_storage = getattr(self.config, "graph_storage", None)
        if not graph_storage:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available (graph_storage not configured)",
            )

        # Step 1: Find relevant compliance communities
        try:
            communities = graph_storage.search_communities(
                query, level=community_level, limit=5
            )
        except Exception as e:
            logger.error(f"Community search failed: {e}", exc_info=True)
            return ToolResult(
                success=False, data=None,
                error=f"Community search failed: {e}",
            )

        if not communities:
            return ToolResult(
                success=True,
                data={
                    "compliance_domain": None,
                    "findings": [],
                    "overall_score": 0.0,
                    "summary": {
                        "total_requirements": 0, "met": 0,
                        "unmet": 0, "partial": 0, "unclear": 0,
                    },
                },
                metadata={"query": query, "message": "No relevant compliance communities found"},
            )

        # Step 2: Extract requirements from communities
        requirements = []
        community_titles = []
        for community in communities:
            community_titles.append(community.get("title", "Unknown"))
            try:
                entities = graph_storage.get_community_entities(community["community_id"])
            except Exception as e:
                logger.warning(f"Failed to get entities for community {community['community_id']}: {e}")
                continue

            for entity in entities:
                if entity["entity_type"] in _REQUIREMENT_TYPES:
                    # Apply regulation filter if specified
                    if regulation_filter and regulation_filter.lower() not in entity.get("name", "").lower():
                        if regulation_filter.lower() not in entity.get("description", "").lower():
                            continue
                    requirements.append(entity)

            if len(requirements) >= max_requirements:
                requirements = requirements[:max_requirements]
                break

        if not requirements:
            return ToolResult(
                success=True,
                data={
                    "compliance_domain": ", ".join(community_titles),
                    "findings": [],
                    "overall_score": 1.0,
                    "summary": {
                        "total_requirements": 0, "met": 0,
                        "unmet": 0, "partial": 0, "unclear": 0,
                    },
                },
                metadata={
                    "query": query,
                    "message": "No requirement entities found in communities",
                    "communities_searched": community_titles,
                },
            )

        # Step 3 & 4: Evidence matching + LLM assessment
        findings = self._assess_requirements(requirements, document_id)

        # Step 5: Build report
        status_counts = {"met": 0, "unmet": 0, "partial": 0, "unclear": 0}
        for f in findings:
            status_key = f["status"].lower()
            if status_key in status_counts:
                status_counts[status_key] += 1

        total = len(findings)
        overall_score = status_counts["met"] / total if total > 0 else 0.0
        if total > 0:
            # Weighted: MET=1.0, PARTIAL=0.5, UNCLEAR=0.25, UNMET=0.0
            overall_score = (
                status_counts["met"] * 1.0
                + status_counts["partial"] * 0.5
                + status_counts["unclear"] * 0.25
            ) / total

        return ToolResult(
            success=True,
            data={
                "compliance_domain": ", ".join(community_titles),
                "overall_score": round(overall_score, 3),
                "findings": findings,
                "summary": {
                    "total_requirements": total,
                    **status_counts,
                },
            },
            metadata={
                "query": query,
                "document_id": document_id,
                "communities_searched": community_titles,
                "community_level": community_level,
            },
        )

    def _assess_requirements(self, requirements, document_id):
        """Assess each requirement against document evidence."""
        findings = []
        assessment_prompt = self._load_assessment_prompt()

        for req in requirements:
            # Search for evidence
            evidence_text, evidence_source = self._find_evidence(
                req["description"] or req["name"], document_id
            )

            if not evidence_text:
                # No evidence found — mark as UNMET
                findings.append({
                    "requirement": req["name"],
                    "requirement_type": req["entity_type"],
                    "source_entity_id": req["entity_id"],
                    "status": "UNMET",
                    "confidence": 0.8,
                    "evidence": None,
                    "evidence_source": None,
                    "gap_description": f"No evidence found for: {req['description'] or req['name']}",
                })
                continue

            # LLM assessment
            assessment = self._llm_assess(
                assessment_prompt, req, evidence_text
            )
            findings.append({
                "requirement": req["name"],
                "requirement_type": req["entity_type"],
                "source_entity_id": req["entity_id"],
                "status": assessment.get("status", "UNCLEAR"),
                "confidence": assessment.get("confidence", 0.5),
                "evidence": evidence_text[:500],
                "evidence_source": evidence_source,
                "gap_description": (
                    assessment.get("explanation")
                    if assessment.get("status") in ("UNMET", "PARTIAL")
                    else None
                ),
            })

        return findings

    def _find_evidence(self, query, document_id):
        """Search for evidence using VL retriever or vector store."""
        try:
            if self._is_vl_mode():
                results = self.vl_retriever.search(
                    query=query, k=3, document_filter=document_id
                )
                if results:
                    # For VL mode, get page summaries from metadata
                    texts = []
                    source = None
                    for r in results:
                        meta = getattr(r, "metadata", {}) or {}
                        summary = meta.get("page_summary", "")
                        if summary:
                            texts.append(summary)
                        if not source:
                            source = r.page_id
                    return "\n\n".join(texts) if texts else None, source
            else:
                # OCR mode — search vector store directly
                results = self.vector_store.similarity_search(query, k=3)
                if results:
                    texts = [r.get("content", r.get("raw_content", "")) for r in results]
                    source = results[0].get("chunk_id", "unknown")
                    return "\n\n".join(t for t in texts if t), source
        except Exception as e:
            logger.warning(f"Evidence search failed for '{query[:50]}...': {e}")

        return None, None

    def _llm_assess(self, prompt_template, requirement, evidence_text):
        """Use LLM to assess whether evidence satisfies requirement."""
        if not self.llm_provider:
            # No LLM available — use heuristic
            return {"status": "UNCLEAR", "confidence": 0.3, "explanation": "No LLM available for assessment"}

        prompt = prompt_template.format(
            requirement_type=requirement["entity_type"],
            requirement_text=requirement["description"] or requirement["name"],
            requirement_source=requirement.get("document_id", "unknown"),
            evidence_text=evidence_text,
        )

        try:
            response = self.llm_provider.create_message(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                system="",
                max_tokens=500,
                temperature=0.0,
            )
            return self._parse_assessment(response.text)
        except Exception as e:
            logger.warning(f"LLM assessment failed: {e}")
            return {"status": "UNCLEAR", "confidence": 0.3, "explanation": f"Assessment failed: {e}"}

    def _parse_assessment(self, text):
        """Parse LLM assessment response."""
        if not text:
            return {"status": "UNCLEAR", "confidence": 0.3, "explanation": "Empty response"}

        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

        try:
            data = json.loads(text)
            status = data.get("status", "UNCLEAR").upper()
            if status not in ("MET", "UNMET", "PARTIAL", "UNCLEAR"):
                status = "UNCLEAR"
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            return {
                "status": status,
                "confidence": confidence,
                "explanation": data.get("explanation", ""),
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse assessment JSON: {e}. Response: {text[:200]}")
            return {"status": "UNCLEAR", "confidence": 0.3, "explanation": text[:200]}

    def _load_assessment_prompt(self):
        """Load compliance assessment prompt template."""
        if _ASSESSMENT_PROMPT_PATH.exists():
            return _ASSESSMENT_PROMPT_PATH.read_text(encoding="utf-8")
        logger.warning(f"Assessment prompt not found: {_ASSESSMENT_PROMPT_PATH}")
        return (
            "Assess whether this evidence satisfies the requirement.\n"
            "Requirement ({requirement_type}): {requirement_text}\n"
            "Source: {requirement_source}\n"
            "Evidence: {evidence_text}\n"
            "Respond with JSON: {{\"status\": \"MET|UNMET|PARTIAL|UNCLEAR\", "
            "\"confidence\": 0.0-1.0, \"explanation\": \"...\"}}"
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agent/tools/test_compliance_check.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/agent/tools/compliance_check.py tests/agent/tools/test_compliance_check.py
git commit -m "feat: add compliance_check agent tool with tests"
```

---

### Task 5: Register Tool in __init__.py

**Files:**
- Modify: `src/agent/tools/__init__.py:62-64`

**Step 1: Write failing test**

Create `tests/agent/tools/test_tool_registration.py`:

```python
"""Test that compliance_check tool is registered."""

import pytest
from src.agent.tools import get_registry


def test_compliance_check_registered():
    """compliance_check should be in the tool registry."""
    registry = get_registry()
    tool_names = list(registry._tool_classes.keys())
    assert "compliance_check" in tool_names, (
        f"compliance_check not registered. Available: {tool_names}"
    )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/tools/test_tool_registration.py -v`
Expected: FAIL — `compliance_check` not in registry

**Step 3: Add import to __init__.py**

In `src/agent/tools/__init__.py`, after the Graph RAG tools block (line 64), add:

```python
# Compliance tools
_safe_import("compliance_check")
```

Also update the module docstring (line 4) to reflect 9 tools:
```python
"""
RAG Tools

9 tools for retrieval, analysis, knowledge graph queries, and compliance checking.
All tools are registered automatically via @register_tool decorator.

Tools:
- search: Jina v4 cosine search -> page images (VL mode)
- expand_context: Adjacent page expansion
- get_document_list: List all indexed documents
- get_document_info: Document metadata/summaries
- get_stats: Corpus/index statistics
- graph_search: Entity search in knowledge graph
- graph_context: Multi-hop entity neighborhood
- graph_communities: Thematic community summaries
- compliance_check: Community-based compliance assessment
"""
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/agent/tools/test_tool_registration.py tests/agent/tools/test_compliance_check.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agent/tools/__init__.py tests/agent/tools/test_tool_registration.py
git commit -m "feat: register compliance_check tool in tool registry"
```

---

### Task 6: Update graph_search Tool Description

**Files:**
- Modify: `src/agent/tools/graph_search.py:23-26`

**Step 1: Update entity_type field description to include new types**

In `src/agent/tools/graph_search.py` line 23-26, update the `entity_type` field description:

```python
    entity_type: Optional[str] = Field(
        None,
        description=(
            "Filter by entity type: REGULATION, STANDARD, ORGANIZATION, PERSON, "
            "CONCEPT, FACILITY, ROLE, DOCUMENT, SECTION, REQUIREMENT, "
            "OBLIGATION, PROHIBITION, PERMISSION, EVIDENCE, CONTROL"
        ),
    )
```

Also update the tool description (line 36-38) to mention compliance types:

```python
    description = (
        "Search knowledge graph for entities (regulations, standards, organizations, persons, "
        "concepts, facilities, roles, documents, sections, requirements, "
        "obligations, prohibitions, permissions, evidence, controls). "
        "Returns matching entities with their relationships."
    )
```

**Step 2: Verify no test breakage**

Run: `uv run pytest tests/ -v -x --timeout=30 2>/dev/null | head -50`
Expected: No failures related to graph_search

**Step 3: Commit**

```bash
git add src/agent/tools/graph_search.py
git commit -m "feat: update graph_search description with compliance entity types"
```

---

### Task 7: Run Full Test Suite and Verify

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests pass, no regressions

**Step 2: Verify tool loads in Python**

Run:
```bash
uv run python -c "
from src.agent.tools import get_registry
registry = get_registry()
print('Registered tools:', list(registry._tool_classes.keys()))
assert 'compliance_check' in registry._tool_classes
print('compliance_check tool registered successfully')
"
```
Expected: "compliance_check tool registered successfully"

**Step 3: Final commit (if any fixes needed)**

If any fixes were needed, commit them:
```bash
git add -A
git commit -m "fix: resolve test issues from compliance check integration"
```

---

### Task 8: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/GRAPH_RAG.md` (if exists)

**Step 1: Update CLAUDE.md**

Add compliance_check to the tool list in the Architecture Overview section. Update entity type list. Add note about new compliance entity types.

**Step 2: Update docs/GRAPH_RAG.md**

Add section about compliance entity types and the compliance_check tool.

**Step 3: Commit**

```bash
git add CLAUDE.md docs/GRAPH_RAG.md
git commit -m "docs: add compliance check documentation"
```

---

### Post-Implementation: Re-build Graph (Manual Step)

After all code is deployed, run these commands to re-extract entities with the new types:

```bash
# Re-extract entities with new compliance types (full re-extraction)
uv run python scripts/graph_rag_build.py

# Re-embed new entities
uv run python scripts/graph_embed_backfill.py
```

This is a one-time operation that will:
1. Re-extract entities from all pages → new OBLIGATION/PROHIBITION/PERMISSION/EVIDENCE/CONTROL entities appear
2. Re-run Leiden community detection → communities restructure into compliance domains
3. Re-generate community summaries → summaries reflect compliance structure
4. Backfill embeddings for new entities → semantic search works for new types
