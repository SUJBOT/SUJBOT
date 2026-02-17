"""
Community Summarizer — LLM summaries for entity communities.

Generates structured title + description for clusters of related entities
using their names, types, descriptions, and relationships.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..exceptions import ConfigurationError
from ..utils.text_helpers import strip_code_fences

if TYPE_CHECKING:
    from ..agent.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "graph_community_summary.txt"


class CommunitySummarizer:
    """Generate structured title + description for entity communities via LLM."""

    def __init__(self, provider: "BaseProvider"):
        self.provider = provider
        if not _PROMPT_PATH.exists():
            raise ConfigurationError(
                f"Community summary prompt not found: {_PROMPT_PATH}",
                details={"prompt_path": str(_PROMPT_PATH)},
            )
        self._prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def summarize(
        self,
        community_entities: List[Dict],
        community_relationships: List[Dict],
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a structured title and description for a community of entities.

        Args:
            community_entities: List of entity dicts (name, entity_type, description)
            community_relationships: List of relationship dicts between community entities

        Returns:
            Tuple of (title, description), or None if generation fails
        """
        if not community_entities:
            return None

        # Build text representation
        parts = ["Entities in this community:"]
        for e in community_entities:
            desc = f" — {e['description']}" if e.get("description") else ""
            parts.append(f"- [{e.get('entity_type', '?')}] {e['name']}{desc}")

        if community_relationships:
            parts.append("\nRelationships:")
            for r in community_relationships[:20]:  # Cap at 20 for token efficiency
                parts.append(f"- {r.get('source', '?')} —[{r.get('type', '?')}]→ {r.get('target', '?')}")

        context = "\n".join(parts)
        user_message = f"{self._prompt}\n\n{context}"

        try:
            response = self.provider.create_message(
                messages=[{"role": "user", "content": user_message}],
                tools=[],
                system="",
                max_tokens=300,
                temperature=0.0,
            )
            return self._parse_response(response.text)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as e:
            logger.warning(f"Community summarization failed: {e}", exc_info=True)
            return None

    def _parse_response(self, text: str) -> Optional[Tuple[str, str]]:
        """Parse structured JSON response into (title, description) tuple."""
        if not text:
            logger.warning("Empty response from community summarizer LLM")
            return None

        text = strip_code_fences(text.strip())

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from community summary: {e}. Preview: {text[:200]}")
            return None

        if not isinstance(data, dict):
            logger.warning(f"Community summary JSON is not an object: {type(data).__name__}. Preview: {text[:200]}")
            return None

        title = (data.get("title") or "").strip()
        description = (data.get("description") or "").strip()

        if not title or not description:
            logger.warning(f"Missing title or description in community summary: {data}")
            return None

        return (title[:100], description)
