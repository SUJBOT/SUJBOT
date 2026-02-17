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
        description=(
            "Filter by entity type: REGULATION, STANDARD, ORGANIZATION, PERSON, "
            "CONCEPT, FACILITY, ROLE, DOCUMENT, SECTION, REQUIREMENT, "
            "OBLIGATION, PROHIBITION, PERMISSION, EVIDENCE, CONTROL, "
            "DEFINITION, SANCTION, DEADLINE, AMENDMENT"
        ),
    )
    limit: int = Field(10, description="Maximum results", ge=1, le=50)


@register_tool
class GraphSearchTool(BaseTool):
    """Search the knowledge graph for entities by semantic similarity."""

    name = "graph_search"
    description = (
        "Search knowledge graph for entities (regulations, standards, organizations, persons, "
        "concepts, facilities, roles, documents, sections, requirements, "
        "obligations, prohibitions, permissions, evidence, controls, "
        "definitions, sanctions, deadlines, amendments). "
        "Returns matching entities with their relationships."
    )
    input_schema = GraphSearchInput

    def execute_impl(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> ToolResult:
        graph_storage, err = self._get_graph_storage()
        if err:
            return err

        # Determine fetch size: if adaptive-k is enabled and graph uses
        # embedding search, fetch a larger candidate pool for thresholding.
        adaptive_config = getattr(self.config, "adaptive_retrieval", None)
        uses_embeddings = hasattr(graph_storage, "_embedder") and graph_storage._embedder
        if adaptive_config and adaptive_config.enabled and uses_embeddings:
            fetch_limit = max(adaptive_config.fetch_k, limit)
        else:
            fetch_limit = limit

        try:
            entities = graph_storage.search_entities(
                query=query,
                entity_type=entity_type,
                limit=fetch_limit,
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

        # Apply adaptive-k filtering for embedding search results
        adaptive_meta = {}
        if (
            adaptive_config
            and adaptive_config.enabled
            and uses_embeddings
            and len(entities) >= adaptive_config.min_samples_for_adaptive
        ):
            try:
                from src.retrieval.adaptive_k import adaptive_k_filter, AdaptiveKConfig

                scores = [e.get("score", 0.0) for e in entities]
                effective_config = AdaptiveKConfig(
                    enabled=True,
                    method=adaptive_config.method,
                    fetch_k=adaptive_config.fetch_k,
                    min_k=adaptive_config.min_k,
                    max_k=min(adaptive_config.max_k, limit),
                    score_gap_threshold=adaptive_config.score_gap_threshold,
                    min_samples_for_adaptive=adaptive_config.min_samples_for_adaptive,
                )
                result = adaptive_k_filter(entities, scores, effective_config)
                entities = result.items

                adaptive_meta = {
                    "adaptive_k": {
                        "threshold": result.threshold,
                        "method": result.method_used,
                        "original_count": result.original_count,
                        "filtered_count": result.filtered_count,
                        "score_range": result.score_range,
                    }
                }

                logger.info(
                    "Graph adaptive-k: %d -> %d entities (threshold=%.3f, method=%s)",
                    result.original_count,
                    result.filtered_count,
                    result.threshold,
                    result.method_used,
                )
            except Exception as e:
                logger.warning(f"Graph adaptive-k failed, using raw results: {e}")

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

        metadata = {"query": query, "entity_type": entity_type}
        metadata.update(adaptive_meta)

        return ToolResult(
            success=True,
            data={"entities": entities, "count": len(entities)},
            metadata=metadata,
        )
