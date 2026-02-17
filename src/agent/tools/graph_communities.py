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
        graph_storage, err = self._get_graph_storage()
        if err:
            return err

        try:
            if query:
                communities = graph_storage.search_communities(query, level=level, limit=10)
            else:
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

        # Enrich top communities with entity details
        for c in communities[:5]:
            try:
                entities = graph_storage.get_community_entities(c["community_id"])
                c["entities"] = [
                    {"name": e["name"], "type": e["entity_type"]}
                    for e in entities[:15]
                ]
            except Exception as e:
                logger.warning(
                    f"Failed to get entities for community {c['community_id']}: {e}",
                    exc_info=True,
                )
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
