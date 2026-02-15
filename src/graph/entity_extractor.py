"""
Entity Extractor — multimodal extraction from page images.

Sends page image + prompt to an LLM provider, parses structured JSON output
containing entities and relationships.
"""

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from ..exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..agent.providers.base import BaseProvider
    from ..vl.page_store import PageStore

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "graph_entity_extraction.txt"
)

# Valid entity and relationship types (for validation)
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
    "OBLIGATION",
    "PROHIBITION",
    "PERMISSION",
    "EVIDENCE",
    "CONTROL",
}
RELATIONSHIP_TYPES = {
    "DEFINES",
    "REFERENCES",
    "AMENDS",
    "REQUIRES",
    "REGULATES",
    "PART_OF",
    "APPLIES_TO",
    "SUPERVISES",
    "AUTHORED_BY",
}


class EntityExtractor:
    """
    Extracts entities and relationships from page images via multimodal LLM.

    Uses the same provider interface as VLIndexingPipeline summary generation.
    """

    def __init__(self, provider: "BaseProvider"):
        self.provider = provider
        if not _PROMPT_PATH.exists():
            raise ConfigurationError(
                f"Entity extraction prompt not found: {_PROMPT_PATH}",
                details={"prompt_path": str(_PROMPT_PATH)},
            )
        self._prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def extract_from_page(self, page_id: str, page_store: "PageStore") -> Dict[str, List[Dict]]:
        """
        Extract entities and relationships from a page image.

        Args:
            page_id: Page identifier (e.g., "BZ_VR1_p001")
            page_store: PageStore instance for loading images

        Returns:
            Dict with 'entities' and 'relationships' lists, or empty if extraction fails
        """

        try:
            image_b64 = page_store.get_image_base64(page_id)
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Page image not found for {page_id}, skipping extraction: {e}")
            return {"entities": [], "relationships": []}

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{page_store.image_format}",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

        try:
            response = self.provider.create_message(
                messages=messages,
                tools=[],
                system="",
                max_tokens=8000,
                temperature=0.0,
            )
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as e:
            logger.error(
                f"LLM call failed for entity extraction on {page_id}: {e}",
                exc_info=True,
            )
            return {"entities": [], "relationships": []}

        return self._parse_response(response.text, page_id)

    def _parse_response(self, text: str, page_id: str) -> Dict[str, List[Dict]]:
        """Parse LLM response into structured entities and relationships."""
        if not text:
            logger.warning(f"Empty extraction response for {page_id}")
            return {"entities": [], "relationships": []}

        # Strip markdown code fences if present
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON from extraction for {page_id}: {e}. "
                f"Response preview: {text[:200]}"
            )
            return {"entities": [], "relationships": []}

        # Validate and filter entities
        raw_entities = data.get("entities", [])
        entities = []
        for ent in raw_entities:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name", "").strip()
            etype = ent.get("type", "").strip().upper()
            if not name or etype not in ENTITY_TYPES:
                continue
            entities.append(
                {
                    "name": name,
                    "type": etype,
                    "description": ent.get("description", ""),
                }
            )

        filtered_entities = len(raw_entities) - len(entities)
        if filtered_entities > 0:
            logger.debug(
                f"{page_id}: filtered out {filtered_entities}/{len(raw_entities)} "
                f"entities (invalid name or type)"
            )

        # Validate and filter relationships
        entity_names = {ent["name"] for ent in entities}
        raw_relationships = data.get("relationships", [])
        relationships = []
        for rel in raw_relationships:
            if not isinstance(rel, dict):
                continue
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rtype = rel.get("type", "").strip().upper()
            if not source or not target or rtype not in RELATIONSHIP_TYPES:
                continue
            if source not in entity_names or target not in entity_names:
                continue
            relationships.append(
                {
                    "source": source,
                    "target": target,
                    "type": rtype,
                    "description": rel.get("description", ""),
                }
            )

        filtered_rels = len(raw_relationships) - len(relationships)
        if filtered_rels > 0:
            logger.debug(
                f"{page_id}: filtered out {filtered_rels}/{len(raw_relationships)} "
                f"relationships (invalid type or missing entities)"
            )

        logger.debug(
            f"{page_id}: extracted {len(entities)} entities, {len(relationships)} relationships"
        )
        return {"entities": entities, "relationships": relationships}
