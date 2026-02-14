"""
Community Summarizer — LLM summaries for entity communities.

Generates text summaries for clusters of related entities
using their names, types, descriptions, and relationships.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from ..exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..agent.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "graph_community_summary.txt"


class CommunitySummarizer:
    """Generate text summaries for entity communities via LLM."""

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
    ) -> Optional[str]:
        """
        Generate a text summary for a community of entities.

        Args:
            community_entities: List of entity dicts (name, entity_type, description)
            community_relationships: List of relationship dicts between community entities

        Returns:
            Summary text, or None if generation fails
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
        user_message = f"{self._prompt}\n\n{context}\n\nSummary:"

        try:
            response = self.provider.create_message(
                messages=[{"role": "user", "content": user_message}],
                tools=[],
                system="",
                max_tokens=200,
                temperature=0.0,
            )
            return response.text.strip() if response.text else None
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except Exception as e:
            logger.warning(f"Community summarization failed: {e}")
            return None
