"""
Tests for compliance_check agent tool.

Covers input validation, graph storage access, requirement extraction,
evidence search (VL + OCR), LLM assessment parsing, and output structure.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agent.tools.compliance_check import ComplianceCheckInput, ComplianceCheckTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockToolConfig:
    """Minimal ToolConfig stand-in for tests."""

    graph_storage: Optional[Any] = None
    compliance_threshold: float = 0.7


def _make_tool(
    graph_storage=None,
    llm_provider=None,
    vl_retriever=None,
    page_store=None,
    vector_store=None,
    compliance_threshold: float = 0.7,
):
    """Factory for ComplianceCheckTool with mocked dependencies."""
    config = MockToolConfig(
        graph_storage=graph_storage,
        compliance_threshold=compliance_threshold,
    )
    return ComplianceCheckTool(
        vector_store=vector_store or MagicMock(),
        embedder=MagicMock(),
        reranker=None,
        context_assembler=None,
        llm_provider=llm_provider,
        config=config,
        vl_retriever=vl_retriever,
        page_store=page_store,
    )


def _make_community(community_id: int, title: str = "Test Community", level: int = 0):
    """Create a mock community dict."""
    return {
        "community_id": community_id,
        "level": level,
        "title": title,
        "summary": f"Summary for community {community_id}",
        "entity_ids": [1, 2, 3],
        "metadata": {},
    }


def _make_entity(
    entity_id: int,
    name: str,
    entity_type: str,
    description: str = "Some requirement description",
    document_id: str = "doc_1",
):
    """Create a mock entity dict."""
    return {
        "entity_id": entity_id,
        "name": name,
        "entity_type": entity_type,
        "description": description,
        "document_id": document_id,
    }


# ===========================================================================
# TestComplianceCheckInput
# ===========================================================================


class TestComplianceCheckInput:
    """Pydantic input validation."""

    def test_minimal_input(self):
        """Only query is required; defaults should be populated."""
        inp = ComplianceCheckInput(query="Is radiation shielding adequate?")
        assert inp.query == "Is radiation shielding adequate?"
        assert inp.document_id is None
        assert inp.regulation_filter is None
        assert inp.community_level == 0
        assert inp.max_requirements == 20

    def test_full_input(self):
        """All fields provided."""
        inp = ComplianceCheckInput(
            query="Check reactor safety",
            document_id="BZ_VR1",
            regulation_filter="SUJB",
            community_level=2,
            max_requirements=10,
        )
        assert inp.query == "Check reactor safety"
        assert inp.document_id == "BZ_VR1"
        assert inp.regulation_filter == "SUJB"
        assert inp.community_level == 2
        assert inp.max_requirements == 10

    def test_community_level_bounds(self):
        """community_level must be 0-2."""
        with pytest.raises(ValidationError):
            ComplianceCheckInput(query="test", community_level=3)
        with pytest.raises(ValidationError):
            ComplianceCheckInput(query="test", community_level=-1)

    def test_max_requirements_bounds(self):
        """max_requirements must be 1-50."""
        with pytest.raises(ValidationError):
            ComplianceCheckInput(query="test", max_requirements=0)
        with pytest.raises(ValidationError):
            ComplianceCheckInput(query="test", max_requirements=51)


# ===========================================================================
# TestComplianceCheckToolNoGraph
# ===========================================================================


class TestComplianceCheckToolNoGraph:
    """Tool behaviour when graph storage is unavailable."""

    def test_no_graph_returns_error(self):
        """graph_storage=None should return an error ToolResult."""
        tool = _make_tool(graph_storage=None)
        result = tool.execute_impl(query="check safety")
        assert result.success is False
        assert "graph" in result.error.lower()


# ===========================================================================
# TestComplianceCheckToolWithGraph
# ===========================================================================


class TestComplianceCheckToolWithGraph:
    """Core tool behaviour with mocked graph storage."""

    def _setup_graph(self, communities=None, entities=None):
        """Create a graph_storage mock with configurable returns."""
        gs = MagicMock()
        gs.search_communities.return_value = communities or []
        gs.get_community_entities.return_value = entities or []
        return gs

    def test_communities_searched(self):
        """search_communities is called with query and level."""
        gs = self._setup_graph(communities=[])
        tool = _make_tool(graph_storage=gs)
        tool.execute_impl(query="radiation safety", community_level=1)
        gs.search_communities.assert_called_once_with("radiation safety", level=1, limit=5)

    def test_requirements_extracted_from_community(self):
        """get_community_entities is called for each community found."""
        community = _make_community(community_id=42, title="Safety")
        gs = self._setup_graph(
            communities=[community],
            entities=[
                _make_entity(1, "Must monitor radiation", "OBLIGATION"),
            ],
        )
        tool = _make_tool(graph_storage=gs, vector_store=MagicMock())
        # vector_store.similarity_search returns empty list -> UNMET findings
        tool.vector_store.similarity_search.return_value = []
        result = tool.execute_impl(query="radiation safety")
        gs.get_community_entities.assert_called_once_with(42)
        assert result.success is True

    def test_output_structure(self):
        """Result data has findings, summary, overall_score, compliance_domain."""
        community = _make_community(community_id=1, title="Radiation Domain")
        gs = self._setup_graph(
            communities=[community],
            entities=[
                _make_entity(1, "Requirement A", "OBLIGATION"),
            ],
        )
        tool = _make_tool(graph_storage=gs, vector_store=MagicMock())
        tool.vector_store.similarity_search.return_value = []
        result = tool.execute_impl(query="test")
        assert result.success is True

        data = result.data
        assert "findings" in data
        assert "summary" in data
        assert "overall_score" in data
        assert "compliance_domain" in data

        summary = data["summary"]
        assert "total_requirements" in summary
        assert "met" in summary
        assert "unmet" in summary
        assert "partial" in summary
        assert "unclear" in summary

    def test_no_communities_found(self):
        """Empty community search returns success with empty findings."""
        gs = self._setup_graph(communities=[])
        tool = _make_tool(graph_storage=gs)
        result = tool.execute_impl(query="unknown topic")
        assert result.success is True
        assert result.data["findings"] == []
        assert result.data["summary"]["total_requirements"] == 0

    def test_only_requirement_types_extracted(self):
        """Entities of non-requirement types (CONTROL, EVIDENCE) are filtered out."""
        community = _make_community(community_id=1)
        entities = [
            _make_entity(1, "Must do X", "OBLIGATION"),
            _make_entity(2, "Fire control panel", "CONTROL"),
            _make_entity(3, "Audit report", "EVIDENCE"),
            _make_entity(4, "Cannot do Y", "PROHIBITION"),
        ]
        gs = self._setup_graph(communities=[community])
        gs.get_community_entities.return_value = entities
        tool = _make_tool(graph_storage=gs, vector_store=MagicMock())
        tool.vector_store.similarity_search.return_value = []
        result = tool.execute_impl(query="test")
        assert result.success is True

        finding_names = [f["requirement"] for f in result.data["findings"]]
        assert "Must do X" in finding_names
        assert "Cannot do Y" in finding_names
        assert "Fire control panel" not in finding_names
        assert "Audit report" not in finding_names
        assert result.data["summary"]["total_requirements"] == 2

    def test_regulation_filter(self):
        """regulation_filter restricts requirements to matching entities only."""
        community = _make_community(community_id=1)
        entities = [
            _make_entity(1, "SUJB regulation requirement", "REQUIREMENT"),
            _make_entity(2, "EU standard requirement", "REQUIREMENT"),
            _make_entity(
                3,
                "SUJB safety obligation",
                "OBLIGATION",
                description="SUJB mandates yearly inspection",
            ),
        ]
        gs = self._setup_graph(communities=[community])
        gs.get_community_entities.return_value = entities
        tool = _make_tool(graph_storage=gs, vector_store=MagicMock())
        tool.vector_store.similarity_search.return_value = []
        result = tool.execute_impl(query="test", regulation_filter="SUJB")
        assert result.success is True

        finding_names = [f["requirement"] for f in result.data["findings"]]
        assert "SUJB regulation requirement" in finding_names
        assert "SUJB safety obligation" in finding_names
        assert "EU standard requirement" not in finding_names

    def test_max_requirements_cap(self):
        """Number of findings is capped at max_requirements."""
        community = _make_community(community_id=1)
        # Create 10 requirement entities
        entities = [_make_entity(i, f"Req {i}", "OBLIGATION") for i in range(10)]
        gs = self._setup_graph(communities=[community])
        gs.get_community_entities.return_value = entities
        tool = _make_tool(graph_storage=gs, vector_store=MagicMock())
        tool.vector_store.similarity_search.return_value = []
        result = tool.execute_impl(query="test", max_requirements=3)
        assert result.success is True
        assert len(result.data["findings"]) == 3
        assert result.data["summary"]["total_requirements"] == 3


# ===========================================================================
# TestComplianceAssessment
# ===========================================================================


class TestComplianceAssessment:
    """LLM-based assessment parsing and fallback behaviour."""

    def _setup_with_evidence(self, llm_response_text=None, llm_provider=None):
        """
        Create tool with one community, one requirement, and evidence.

        If llm_response_text is provided, creates a mock LLM provider that
        returns that text.  If llm_provider is explicitly None, no LLM is set.
        """
        community = _make_community(community_id=1)
        entities = [_make_entity(1, "Monitor radiation", "OBLIGATION")]

        gs = MagicMock()
        gs.search_communities.return_value = [community]
        gs.get_community_entities.return_value = entities

        vs = MagicMock()
        # similarity_search returns dicts like real chunks
        vs.similarity_search.return_value = [
            {
                "content": "Radiation monitoring is performed quarterly.",
                "chunk_id": "doc_1_L3_c5",
                "document_id": "doc_1",
                "score": 0.85,
            }
        ]

        if llm_response_text is not None:
            llm_provider = MagicMock()
            mock_response = MagicMock()
            # The LLM provider's create_message returns an object with .content
            # which is a list of content blocks
            mock_block = MagicMock()
            mock_block.text = llm_response_text
            mock_response.content = [mock_block]
            llm_provider.create_message.return_value = mock_response

        return _make_tool(
            graph_storage=gs,
            llm_provider=llm_provider,
            vector_store=vs,
        )

    def test_parse_valid_assessment(self):
        """Valid JSON from LLM is parsed correctly."""
        llm_json = json.dumps(
            {
                "status": "MET",
                "confidence": 0.92,
                "explanation": "Evidence clearly shows quarterly monitoring.",
            }
        )
        tool = self._setup_with_evidence(llm_response_text=llm_json)
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        assert finding["status"] == "MET"
        assert finding["confidence"] == 0.92
        assert result.data["overall_score"] == 1.0  # MET = 1.0

    def test_parse_invalid_json(self):
        """Malformed JSON falls back to UNCLEAR."""
        tool = self._setup_with_evidence(llm_response_text="This is not JSON at all!")
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        assert finding["status"] == "UNCLEAR"

    def test_parse_empty_response(self):
        """Empty LLM response falls back to UNCLEAR."""
        tool = self._setup_with_evidence(llm_response_text="")
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        assert finding["status"] == "UNCLEAR"

    def test_no_llm_provider(self):
        """Tool works without LLM — uses heuristic (evidence found -> UNCLEAR)."""
        tool = self._setup_with_evidence(llm_provider=None)
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        # Without LLM, evidence exists but can't be assessed -> UNCLEAR
        assert finding["status"] == "UNCLEAR"
        assert finding["evidence"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
