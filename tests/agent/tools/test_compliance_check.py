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

    def test_empty_query_rejected(self):
        """Empty query string must be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            ComplianceCheckInput(query="")


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
        """Tool works without LLM — falls back to UNCLEAR when evidence is found."""
        tool = self._setup_with_evidence(llm_provider=None)
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        # Without LLM, evidence exists but can't be assessed -> UNCLEAR
        assert finding["status"] == "UNCLEAR"
        assert finding["evidence"] is not None


# ===========================================================================
# TestComplianceVLMode
# ===========================================================================


class TestComplianceVLMode:
    """VL mode: evidence from page images (multimodal) instead of text chunks."""

    def _setup_vl_tool(self, llm_response_text=None, search_results=None):
        """Create a VL-mode compliance tool with mocked VL components."""
        community = _make_community(community_id=1)
        entities = [_make_entity(1, "Monitor radiation", "OBLIGATION")]

        gs = MagicMock()
        gs.search_communities.return_value = [community]
        gs.get_community_entities.return_value = entities

        vl_retriever = MagicMock()
        if search_results is None:
            search_results = [
                {
                    "page_id": "doc_1_page_3",
                    "document_id": "doc_1",
                    "page_number": 3,
                    "score": 0.88,
                },
                {
                    "page_id": "doc_1_page_7",
                    "document_id": "doc_1",
                    "page_number": 7,
                    "score": 0.75,
                },
            ]
        vl_retriever.search.return_value = search_results

        page_store = MagicMock()
        page_store.get_image_base64.side_effect = lambda pid: f"base64_data_for_{pid}"

        llm_provider = None
        if llm_response_text is not None:
            llm_provider = MagicMock()
            mock_block = MagicMock()
            mock_block.text = llm_response_text
            mock_response = MagicMock()
            mock_response.content = [mock_block]
            llm_provider.create_message.return_value = mock_response

        return _make_tool(
            graph_storage=gs,
            llm_provider=llm_provider,
            vl_retriever=vl_retriever,
            page_store=page_store,
        )

    def test_vl_auto_filters_to_documentation_when_no_doc_id(self):
        """When document_id is None, evidence search filters to 'documentation' category."""
        tool = self._setup_vl_tool()
        tool._search_evidence_vl("radiation", document_id=None)
        call_kwargs = tool.vl_retriever.search.call_args
        assert call_kwargs.kwargs.get("category_filter") == "documentation"

    def test_vl_no_category_filter_when_doc_id_specified(self):
        """When document_id is specified, no category_filter should be applied."""
        tool = self._setup_vl_tool()
        tool._search_evidence_vl("radiation", document_id="BZ_VR1")
        call_kwargs = tool.vl_retriever.search.call_args
        assert call_kwargs.kwargs.get("category_filter") is None

    def test_vl_evidence_search_failure_returns_none(self):
        """VL evidence search failure returns (None, error_msg) not empty list."""
        tool = self._setup_vl_tool()
        tool.vl_retriever.search.side_effect = RuntimeError("connection lost")
        evidence, source = tool._search_evidence_vl("radiation", document_id=None)
        assert evidence is None
        assert "failed" in source.lower()

    def test_vl_evidence_returns_page_images(self):
        """VL mode evidence search returns page image dicts, not text."""
        tool = self._setup_vl_tool()
        evidence, source = tool._search_evidence_vl("radiation", document_id=None)
        assert isinstance(evidence, list)
        assert len(evidence) == 2
        assert evidence[0]["page_id"] == "doc_1_page_3"
        assert evidence[0]["base64_data"] == "base64_data_for_doc_1_page_3"
        assert evidence[0]["document_id"] == "doc_1"
        assert evidence[0]["page_number"] == 3
        assert source == "doc_1_page_3"

    def test_vl_evidence_empty_results(self):
        """VL mode returns empty list when no pages found."""
        tool = self._setup_vl_tool(search_results=[])
        evidence, source = tool._search_evidence_vl("unknown", document_id=None)
        assert evidence == []
        assert source is None

    def test_vl_evidence_image_load_failure(self):
        """VL mode skips pages whose images fail to load."""
        tool = self._setup_vl_tool()
        # First page loads, second raises
        tool.page_store.get_image_base64.side_effect = [
            "base64_ok",
            Exception("disk error"),
        ]
        evidence, source = tool._search_evidence_vl("radiation", document_id=None)
        assert len(evidence) == 1
        assert evidence[0]["base64_data"] == "base64_ok"

    def test_vl_finding_evidence_display(self):
        """In VL mode, finding evidence shows page references not raw text."""
        llm_json = json.dumps({"status": "MET", "confidence": 0.9, "explanation": "ok"})
        tool = self._setup_vl_tool(llm_response_text=llm_json)
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        assert "doc_1 p.3" in finding["evidence"]
        assert "doc_1 p.7" in finding["evidence"]

    def test_vl_llm_receives_image_content_blocks(self):
        """LLM provider receives multimodal content (image blocks + text)."""
        llm_json = json.dumps({"status": "MET", "confidence": 0.85, "explanation": "ok"})
        tool = self._setup_vl_tool(llm_response_text=llm_json)
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True

        # Verify LLM was called with image content blocks
        call_args = tool.llm_provider.create_message.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        content = messages[0]["content"]
        assert isinstance(content, list)

        # Should have 2 image blocks + 1 text block
        image_blocks = [b for b in content if b.get("type") == "image"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(image_blocks) == 2
        assert len(text_blocks) == 1
        assert image_blocks[0]["source"]["type"] == "base64"
        assert image_blocks[0]["source"]["media_type"] == "image/png"

    def test_vl_no_llm_falls_back_to_unclear(self):
        """VL mode without LLM provider returns UNCLEAR with image evidence."""
        tool = self._setup_vl_tool(llm_response_text=None)
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        assert finding["status"] == "UNCLEAR"
        assert "doc_1 p.3" in finding["evidence"]

    def test_vl_no_evidence_returns_unmet(self):
        """VL mode with no page images returns UNMET."""
        tool = self._setup_vl_tool(search_results=[])
        result = tool.execute_impl(query="unknown topic")
        assert result.success is True
        finding = result.data["findings"][0]
        assert finding["status"] == "UNMET"
        assert finding["evidence"] is None

    def test_vl_search_failure_returns_unclear(self):
        """VL evidence search failure results in UNCLEAR, not false UNMET."""
        tool = self._setup_vl_tool()
        tool.vl_retriever.search.side_effect = RuntimeError("DB down")
        result = tool.execute_impl(query="radiation monitoring")
        assert result.success is True
        finding = result.data["findings"][0]
        assert finding["status"] == "UNCLEAR"
        assert "failed" in finding["gap_description"].lower()

    def test_parse_markdown_wrapped_json(self):
        """Markdown-wrapped JSON from LLM is parsed correctly."""
        tool = self._setup_vl_tool()
        status, conf, explanation = tool._parse_assessment_response(
            '```json\n{"status": "MET", "confidence": 0.9, "explanation": "ok"}\n```'
        )
        assert status == "MET"
        assert conf == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
