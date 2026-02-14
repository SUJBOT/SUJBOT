"""
Graph Communities Tool — thematic cluster search.

Returns community summaries from Leiden community detection,
enabling global queries about document themes and cross-document topics.
"""

import logging
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GraphCommunitiesInput(ToolInput):
    """Input for community search."""

    query: Optional[str] = Field(
        None,
        description="Topic to search for in community summaries. If omitted, returns all communities.",
    )
    level: int = Field(
        0,
        description="Hierarchy level (0=finest detail, 1=broader categories, 2=top-level themes)",
        ge=0,
        le=2,
    )


@register_tool
class GraphCommunitiesTool(BaseTool):
    """Get thematic community summaries from the knowledge graph."""

    name = "graph_communities"
    description = (
        "Get thematic community summaries — clusters of related entities detected by Leiden algorithm. "
        "Use for global questions like 'what topics do these documents cover?' or "
        "'which regulations relate to radiation protection?'"
    )
    input_schema = GraphCommunitiesInput

    def execute_impl(
        self,
        query: Optional[str] = None,
        level: int = 0,
    ) -> ToolResult:
        graph_storage = getattr(self.config, "graph_storage", None)
        if not graph_storage:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available (graph_storage not configured)",
            )

        try:
            communities = graph_storage.get_communities(level=level)
        except Exception as e:
            logger.error(f"Graph communities query failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Graph communities query failed: {e}",
            )

        if not communities:
            return ToolResult(
                success=True,
                data={"communities": [], "message": f"No communities found at level {level}"},
                metadata={"level": level},
            )

        # If query provided, filter communities by relevance
        if query:
            query_lower = query.lower()
            scored = []
            for c in communities:
                score = 0
                title = (c.get("title") or "").lower()
                summary = (c.get("summary") or "").lower()
                if query_lower in title:
                    score += 2
                if query_lower in summary:
                    score += 1
                # Check individual query words
                for word in query_lower.split():
                    if len(word) > 2:
                        if word in title:
                            score += 1
                        if word in summary:
                            score += 0.5
                if score > 0:
                    c["relevance_score"] = score
                    scored.append(c)

            scored.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            communities = scored[:10]

        # Enrich top communities with entity details
        for c in communities[:5]:
            try:
                entities = graph_storage.get_community_entities(c["community_id"])
                c["entities"] = [
                    {"name": e["name"], "type": e["entity_type"]}
                    for e in entities[:15]
                ]
            except Exception as e:
                logger.warning(f"Failed to get entities for community {c['community_id']}: {e}")
                c["entities"] = []

        return ToolResult(
            success=True,
            data={
                "communities": communities,
                "count": len(communities),
                "level": level,
            },
            metadata={"query": query, "level": level},
        )
