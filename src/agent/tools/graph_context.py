"""
Graph Context Tool — multi-hop entity neighborhood.

Gets the N-hop neighborhood of an entity in the knowledge graph,
enabling cross-document reasoning and relationship discovery.
"""

import logging

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GraphContextInput(ToolInput):
    """Input for graph context retrieval."""

    entity_name: str = Field(..., description="Entity name to explore")
    depth: int = Field(2, description="Number of hops to traverse (1-3)", ge=1, le=3)


@register_tool
class GraphContextTool(BaseTool):
    """Get N-hop neighborhood of an entity — connected entities and relationships."""

    name = "graph_context"
    description = (
        "Get multi-hop context for an entity in the knowledge graph. "
        "Returns connected entities and relationships within N hops."
    )
    input_schema = GraphContextInput

    def execute_impl(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> ToolResult:
        graph_storage = getattr(self.config, "graph_storage", None)
        if not graph_storage:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available (graph_storage not configured)",
            )

        # Find the entity first
        entities = graph_storage.search_entities(query=entity_name, limit=1)
        if not entities:
            return ToolResult(
                success=True,
                data={"message": f"Entity '{entity_name}' not found in knowledge graph"},
                metadata={"entity_name": entity_name},
            )

        entity = entities[0]
        result = graph_storage.get_entity_relationships(entity["entity_id"], depth=depth)

        return ToolResult(
            success=True,
            data={
                "entity": result["entity"],
                "relationships": result["relationships"],
                "connected_entities": result["connected_entities"],
                "depth": depth,
                "relationship_count": len(result["relationships"]),
                "connected_count": len(result["connected_entities"]),
            },
            metadata={"entity_name": entity_name, "depth": depth},
        )
