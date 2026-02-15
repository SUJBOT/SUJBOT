"""
Graph Search Tool — entity search in knowledge graph.

Searches entities by semantic embedding similarity (multilingual-e5-small),
returns matching entities with their direct relationships.
"""

import logging
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GraphSearchInput(ToolInput):
    """Input for graph entity search."""

    query: str = Field(..., description="Entity name or search term (cross-language: English queries find Czech entities)")
    entity_type: Optional[str] = Field(
        None,
        description="Filter by entity type: REGULATION, STANDARD, ORGANIZATION, PERSON, CONCEPT, FACILITY, ROLE, DOCUMENT, SECTION, REQUIREMENT",
    )
    limit: int = Field(10, description="Maximum results", ge=1, le=50)


@register_tool
class GraphSearchTool(BaseTool):
    """Search the knowledge graph for entities by semantic similarity."""

    name = "graph_search"
    description = (
        "Search knowledge graph for entities (regulations, standards, organizations, persons, "
        "concepts, facilities, roles, documents, sections, requirements). "
        "Returns matching entities with their relationships."
    )
    input_schema = GraphSearchInput

    def execute_impl(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> ToolResult:
        graph_storage = getattr(self.config, "graph_storage", None)
        if not graph_storage:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available (graph_storage not configured)",
            )

        try:
            entities = graph_storage.search_entities(
                query=query,
                entity_type=entity_type,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Graph search failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Graph search failed: {e}",
            )

        if not entities:
            return ToolResult(
                success=True,
                data={"entities": [], "message": f"No entities found for '{query}'"},
                metadata={"query": query, "entity_type": entity_type},
            )

        # For top results, fetch their direct relationships
        for entity in entities[:5]:
            try:
                result = graph_storage.get_entity_relationships(entity["entity_id"], depth=1)
                entity["relationships"] = result.get("relationships", [])[:10]
            except Exception as e:
                logger.warning(
                    f"Failed to get relationships for {entity['name']}: {e}",
                    exc_info=True,
                )
                entity["relationships"] = []

        return ToolResult(
            success=True,
            data={"entities": entities, "count": len(entities)},
            metadata={"query": query, "entity_type": entity_type},
        )
