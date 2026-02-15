"""
Compliance Check Tool — regulatory requirement assessment.

Searches knowledge graph communities for compliance requirements
(OBLIGATION, PROHIBITION, PERMISSION, REQUIREMENT), gathers evidence
from the document corpus (VL or OCR mode), and optionally calls an LLM
to assess each requirement's compliance status.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)

# Requirement entity types to extract from communities
_REQUIREMENT_TYPES = frozenset({"OBLIGATION", "PROHIBITION", "PERMISSION", "REQUIREMENT"})

# Status weights for overall score calculation
_STATUS_WEIGHTS: Dict[str, float] = {
    "MET": 1.0,
    "PARTIAL": 0.5,
    "UNCLEAR": 0.25,
    "UNMET": 0.0,
}

# Max evidence text length per finding
_MAX_EVIDENCE_CHARS = 500

# Assessment prompt template path (relative to project root)
_ASSESSMENT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "compliance_assessment.txt"
)


class ComplianceCheckInput(ToolInput):
    """Input for compliance check tool."""

    query: str = Field(..., description="Compliance question or topic to check")
    document_id: Optional[str] = Field(
        None,
        description="Document ID to check compliance against (if omitted, searches all documents)",
    )
    regulation_filter: Optional[str] = Field(
        None,
        description="Filter to a specific regulation (matches entity name or description)",
    )
    community_level: int = Field(
        0,
        description="Community hierarchy level (0=finest, 1=broader, 2=top-level)",
        ge=0,
        le=2,
    )
    max_requirements: int = Field(
        20,
        description="Maximum number of requirements to assess",
        ge=1,
        le=50,
    )


@register_tool
class ComplianceCheckTool(BaseTool):
    """Assess document compliance against regulatory requirements from the knowledge graph."""

    name = "compliance_check"
    description = (
        "Check document compliance against regulatory requirements extracted from the knowledge graph. "
        "Searches communities for obligations/prohibitions/permissions/requirements, "
        "gathers evidence, and assesses compliance status (MET/UNMET/PARTIAL/UNCLEAR)."
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
        # 1. Check graph_storage availability
        graph_storage = getattr(self.config, "graph_storage", None)
        if not graph_storage:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available (graph_storage not configured)",
            )

        # 2. Search communities
        try:
            communities = graph_storage.search_communities(query, level=community_level, limit=5)
        except Exception as e:
            logger.error(f"Community search failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Community search failed: {e}",
            )

        if not communities:
            return ToolResult(
                success=True,
                data={
                    "compliance_domain": "",
                    "overall_score": 0.0,
                    "findings": [],
                    "summary": {
                        "total_requirements": 0,
                        "met": 0,
                        "unmet": 0,
                        "partial": 0,
                        "unclear": 0,
                    },
                },
                metadata={"query": query, "community_level": community_level},
            )

        # 3. Extract requirements from communities
        compliance_domain = ", ".join(c.get("title", "") for c in communities if c.get("title"))
        requirements = self._extract_requirements(graph_storage, communities, regulation_filter)

        # 4. Cap at max_requirements
        requirements = requirements[:max_requirements]

        # 5. Load assessment prompt template
        assessment_prompt = self._load_assessment_prompt()

        # 6. Assess each requirement
        findings: List[Dict[str, Any]] = []
        for req in requirements:
            finding = self._assess_requirement(req, document_id, assessment_prompt)
            findings.append(finding)

        # 7. Build summary
        summary = self._build_summary(findings)
        overall_score = self._compute_overall_score(findings)

        return ToolResult(
            success=True,
            data={
                "compliance_domain": compliance_domain,
                "overall_score": round(overall_score, 3),
                "findings": findings,
                "summary": summary,
            },
            metadata={
                "query": query,
                "community_level": community_level,
                "document_id": document_id,
                "regulation_filter": regulation_filter,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_requirements(
        self,
        graph_storage: Any,
        communities: List[Dict],
        regulation_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Extract requirement-type entities from communities."""
        requirements: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for community in communities:
            community_id = community.get("community_id")
            if community_id is None:
                continue

            try:
                entities = graph_storage.get_community_entities(community_id)
            except Exception as e:
                logger.warning(
                    f"Failed to get entities for community {community_id}: {e}",
                    exc_info=True,
                )
                continue

            for entity in entities:
                entity_type = entity.get("entity_type", "")
                if entity_type not in _REQUIREMENT_TYPES:
                    continue

                entity_id = entity.get("entity_id")
                if entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)

                # Apply regulation filter
                if regulation_filter:
                    name = entity.get("name", "")
                    description = entity.get("description", "")
                    filter_lower = regulation_filter.lower()
                    if filter_lower not in name.lower() and filter_lower not in description.lower():
                        continue

                requirements.append(entity)

        return requirements

    def _assess_requirement(
        self,
        requirement: Dict[str, Any],
        document_id: Optional[str],
        assessment_prompt: Optional[str],
    ) -> Dict[str, Any]:
        """Assess a single requirement by searching for evidence and optionally using LLM."""
        req_name = requirement.get("name", "Unknown")
        req_type = requirement.get("entity_type", "REQUIREMENT")
        req_description = requirement.get("description", req_name)
        req_entity_id = requirement.get("entity_id", 0)

        # Search for evidence (returns images list in VL mode, text string in OCR mode)
        evidence, evidence_source = self._search_evidence(req_description, document_id)

        # No evidence found
        if not evidence:
            return {
                "requirement": req_name,
                "requirement_type": req_type,
                "source_entity_id": req_entity_id,
                "status": "UNMET",
                "confidence": 0.0,
                "evidence": None,
                "evidence_source": None,
                "gap_description": f"No evidence found for: {req_name}",
            }

        # Assess with LLM (if available)
        status, confidence, explanation = self._run_assessment(
            requirement, evidence, evidence_source, assessment_prompt
        )

        # Build finding — evidence field is page refs for VL, truncated text for OCR
        if isinstance(evidence, list):
            evidence_display = ", ".join(
                f"{img['document_id']} p.{img['page_number']}" for img in evidence
            )
        else:
            evidence_display = evidence[:_MAX_EVIDENCE_CHARS]

        finding: Dict[str, Any] = {
            "requirement": req_name,
            "requirement_type": req_type,
            "source_entity_id": req_entity_id,
            "status": status,
            "confidence": confidence,
            "evidence": evidence_display,
            "evidence_source": evidence_source,
        }

        # Add gap_description for UNMET/PARTIAL
        if status in ("UNMET", "PARTIAL"):
            finding["gap_description"] = explanation or f"Gaps identified for: {req_name}"
        else:
            finding["gap_description"] = None

        return finding

    def _search_evidence(self, query: str, document_id: Optional[str]) -> tuple:
        """Search for evidence. Returns (page_images_list, source) in VL mode, (text, source) in OCR."""
        if self._is_vl_mode():
            return self._search_evidence_vl(query, document_id)
        return self._search_evidence_ocr(query, document_id)

    def _search_evidence_vl(self, query: str, document_id: Optional[str]) -> tuple:
        """VL mode: search page embeddings and load page images for multimodal LLM.

        When no document_id is specified, automatically filters to 'documentation'
        category — requirements already come from legislation (via the knowledge graph),
        so evidence should come from internal documentation.
        """
        try:
            # Auto-filter to documentation when searching broadly for evidence
            category_filter = None if document_id else "documentation"
            results = self.vl_retriever.search(
                query=query, k=3, document_filter=document_id, category_filter=category_filter,
            )
            if not results:
                return ([], None)

            page_images: List[Dict[str, Any]] = []
            source = None
            for r in results:
                page_id = getattr(r, "page_id", None) or r.get("page_id", "")
                if source is None:
                    source = page_id
                try:
                    b64_data = self.page_store.get_image_base64(page_id)
                    page_images.append({
                        "page_id": page_id,
                        "base64_data": b64_data,
                        "document_id": getattr(r, "document_id", "") or r.get("document_id", ""),
                        "page_number": getattr(r, "page_number", 0) or r.get("page_number", 0),
                    })
                except Exception as e:
                    logger.warning(f"Failed to load image for {page_id}: {e}")

            return (page_images, source)
        except Exception as e:
            logger.warning(f"VL evidence search failed: {e}", exc_info=True)
            return ([], None)

    def _search_evidence_ocr(self, query: str, document_id: Optional[str]) -> tuple:
        """OCR mode: search text chunks via vector store."""
        try:
            results = self.vector_store.similarity_search(query, k=3)
            if not results:
                return ("", None)

            # Filter by document_id if specified
            if document_id:
                results = [
                    r
                    for r in results
                    if (
                        r.get("document_id", "")
                        if isinstance(r, dict)
                        else getattr(r, "document_id", "")
                    )
                    == document_id
                ]

            if not results:
                return ("", None)

            # Extract text content
            texts = []
            source = None
            for r in results:
                if isinstance(r, dict):
                    content = r.get("content", r.get("raw_content", ""))
                    chunk_id = r.get("chunk_id", "")
                else:
                    content = getattr(r, "page_content", getattr(r, "content", ""))
                    chunk_id = getattr(r, "chunk_id", "")

                if content:
                    texts.append(content)
                if source is None and chunk_id:
                    source = chunk_id

            evidence = "\n\n".join(texts) if texts else ""
            return (evidence, source)
        except Exception as e:
            logger.warning(f"OCR evidence search failed: {e}", exc_info=True)
            return ("", None)

    def _run_assessment(
        self,
        requirement: Dict[str, Any],
        evidence: Any,
        evidence_source: Optional[str],
        assessment_prompt: Optional[str],
    ) -> tuple:
        """Run LLM assessment or fall back to heuristic. Evidence is images (list) or text (str)."""
        if not self.llm_provider or not assessment_prompt:
            return ("UNCLEAR", 0.25, "LLM provider not available for assessment")

        req_name = requirement.get("name", "Unknown")
        req_type = requirement.get("entity_type", "REQUIREMENT")
        req_description = requirement.get("description", req_name)
        req_source = requirement.get("document_id", "Unknown")

        # Build prompt text (evidence placeholder differs by mode)
        if isinstance(evidence, list):
            evidence_note = f"[See the {len(evidence)} page image(s) above]"
        else:
            evidence_note = evidence[:_MAX_EVIDENCE_CHARS]

        prompt_text = (
            assessment_prompt.replace("{requirement_type}", req_type)
            .replace("{requirement_text}", req_description)
            .replace("{requirement_source}", req_source)
            .replace("{evidence_text}", evidence_note)
        )

        # Build message content — multimodal for VL, text-only for OCR
        if isinstance(evidence, list) and evidence:
            content: List[Dict[str, Any]] = []
            for img in evidence:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img["base64_data"],
                    },
                })
            content.append({"type": "text", "text": prompt_text})
        else:
            content = prompt_text  # type: ignore[assignment]

        try:
            response = self.llm_provider.create_message(
                messages=[{"role": "user", "content": content}],
                tools=[],
                system="",
                max_tokens=512,
                temperature=0.0,
            )

            # Extract text from response
            response_text = ""
            if hasattr(response, "content") and response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        response_text += block.text

            return self._parse_assessment_response(response_text)

        except Exception as e:
            logger.warning(f"LLM assessment failed for '{req_name}': {e}", exc_info=True)
            return ("UNCLEAR", 0.25, f"Assessment failed: {e}")

    def _parse_assessment_response(self, response_text: str) -> tuple:
        """Parse LLM JSON response. Falls back to UNCLEAR on failure."""
        if not response_text or not response_text.strip():
            return ("UNCLEAR", 0.25, "Empty assessment response")

        try:
            # Try to extract JSON from response (handle potential markdown wrapping)
            text = response_text.strip()
            if text.startswith("```"):
                # Strip markdown code block
                lines = text.split("\n")
                text = "\n".join(line for line in lines if not line.strip().startswith("```"))

            data = json.loads(text)
            status = data.get("status", "UNCLEAR").upper()
            if status not in _STATUS_WEIGHTS:
                status = "UNCLEAR"

            confidence = float(data.get("confidence", 0.25))
            confidence = max(0.0, min(1.0, confidence))

            explanation = data.get("explanation", "")
            return (status, confidence, explanation)

        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(f"Failed to parse assessment JSON: {response_text[:200]}")
            return ("UNCLEAR", 0.25, "Could not parse assessment response")

    def _load_assessment_prompt(self) -> Optional[str]:
        """Load assessment prompt template from file."""
        try:
            if _ASSESSMENT_PROMPT_PATH.exists():
                return _ASSESSMENT_PROMPT_PATH.read_text(encoding="utf-8")
            logger.warning(f"Assessment prompt not found: {_ASSESSMENT_PROMPT_PATH}")
            return None
        except Exception as e:
            logger.error(f"Failed to load assessment prompt: {e}")
            return None

    @staticmethod
    def _build_summary(findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Build summary counts from findings."""
        counts = {"met": 0, "unmet": 0, "partial": 0, "unclear": 0}
        for f in findings:
            status = f.get("status", "UNCLEAR").lower()
            if status in counts:
                counts[status] += 1
        counts["total_requirements"] = len(findings)
        return counts

    @staticmethod
    def _compute_overall_score(findings: List[Dict[str, Any]]) -> float:
        """Compute weighted overall compliance score (0.0 - 1.0)."""
        if not findings:
            return 0.0
        total = sum(_STATUS_WEIGHTS.get(f.get("status", "UNCLEAR"), 0.0) for f in findings)
        return total / len(findings)
