# Knowledge Graph Architecture (Graph RAG)

The Graph RAG subsystem extracts structured knowledge from document pages and stores it as a PostgreSQL-based knowledge graph. It enables cross-document reasoning and thematic queries that pure vector search cannot handle well.

## Purpose

Standard vector search retrieves relevant passages but struggles with:
- **Cross-document queries**: "Which regulations reference standard X?" requires understanding relationships across documents.
- **Global/thematic queries**: "What are the main topics in the corpus?" requires aggregation, not retrieval.
- **Entity-centric queries**: "What does SUJB regulate?" requires traversing relationships from a specific entity.

Graph RAG addresses these by maintaining a knowledge graph of entities and their relationships, organized into hierarchical communities with LLM-generated summaries.

## Architecture

```
PDF Pages (images)
  --> Multimodal LLM (entity extraction)
  --> PostgreSQL graph schema
       - graph.entities
       - graph.relationships
  --> igraph Leiden community detection
  --> LLM community summarization
       - graph.communities
  --> Agent tools (graph_search, graph_context, graph_communities)
```

### Pipeline Steps

1. **Entity extraction**: Each page image is sent to a multimodal LLM with a structured extraction prompt. The LLM identifies entities and relationships on the page.
2. **Storage**: Entities and relationships are stored in PostgreSQL with deduplication (UPSERT on `name + entity_type + document_id`).
3. **Community detection**: The Leiden algorithm (via igraph) groups densely connected entities into communities at multiple hierarchy levels.
4. **Community summarization**: Each community gets an LLM-generated title and summary describing its theme and key entities.

## Entity Types (10)

| Type | Description | Example |
|------|-------------|---------|
| `REGULATION` | Legal regulations | Zakon c. 263/2016 Sb. |
| `STANDARD` | Technical standards | ISO 9001 |
| `SECTION` | Document sections | §5 odst. 2 |
| `ORGANIZATION` | Organizations | SUJB, CEZ |
| `PERSON` | Named persons | — |
| `CONCEPT` | Abstract concepts | jaderna bezpecnost |
| `REQUIREMENT` | Requirements/obligations | — |
| `FACILITY` | Physical facilities | JE Dukovany |
| `ROLE` | Organizational roles | — |
| `DOCUMENT` | Referenced documents | — |

## Relationship Types (9)

| Type | Description |
|------|-------------|
| `DEFINES` | Entity A defines entity B |
| `REFERENCES` | Entity A references entity B |
| `AMENDS` | Entity A amends entity B |
| `REQUIRES` | Entity A requires entity B |
| `REGULATES` | Entity A regulates entity B |
| `PART_OF` | Entity A is part of entity B |
| `APPLIES_TO` | Entity A applies to entity B |
| `SUPERVISES` | Entity A supervises entity B |
| `AUTHORED_BY` | Entity A is authored by entity B |

## Database Schema

All tables are in the `graph` schema.

### `graph.entities`

```sql
entity_id       SERIAL PRIMARY KEY
name            TEXT NOT NULL
entity_type     TEXT NOT NULL        -- one of the 10 types above
description     TEXT
source_page_id  TEXT
document_id     TEXT NOT NULL
metadata        JSONB
created_at      TIMESTAMP DEFAULT NOW()

UNIQUE (name, entity_type, document_id)
-- Trigram index on name for similarity search:
CREATE INDEX idx_entities_name_trgm ON graph.entities
  USING gin (name gin_trgm_ops);
```

### `graph.relationships`

```sql
relationship_id    SERIAL PRIMARY KEY
source_entity_id   INT NOT NULL REFERENCES graph.entities(entity_id) ON DELETE CASCADE
target_entity_id   INT NOT NULL REFERENCES graph.entities(entity_id) ON DELETE CASCADE
relationship_type  TEXT NOT NULL     -- one of the 9 types above
description        TEXT
weight             REAL DEFAULT 1.0
source_page_id     TEXT
metadata           JSONB
created_at         TIMESTAMP DEFAULT NOW()

CREATE INDEX idx_rel_source ON graph.relationships(source_entity_id);
CREATE INDEX idx_rel_target ON graph.relationships(target_entity_id);
```

### `graph.communities`

```sql
community_id   SERIAL PRIMARY KEY
level          INT NOT NULL          -- Leiden hierarchy level (0, 1, 2...)
title          TEXT
summary        TEXT
summary_model  TEXT
entity_ids     INT[] NOT NULL        -- Array of entity IDs in this community
metadata       JSONB
created_at     TIMESTAMP DEFAULT NOW()

CREATE INDEX idx_communities_level ON graph.communities(level);
```

## Community Detection

Communities are detected using the **Leiden algorithm** (via the `igraph` library), which partitions the entity-relationship graph into clusters of densely connected nodes.

- **Algorithm**: Leiden (improvement over Louvain, guarantees connected communities)
- **Hierarchical**: Multiple resolution levels (0 = finest, higher = coarser)
- **Resolution parameter**: Controls granularity of communities
- **Implementation**: `src/graph/community_detector.py`

Each community is then summarized by an LLM (`src/graph/community_summarizer.py`) to produce a human-readable title and description of the community's theme.

## Agent Tools

The agent accesses the graph via three tools:

### `graph_search`

Searches entities by name using PostgreSQL trigram similarity (`pg_trgm`).

**Input:** `{"query": "SUJB", "limit": 10}`
**Returns:** Matching entities with type, description, and document source.
**Use case:** Find specific entities by name.

### `graph_context`

Gets the relationship neighborhood of an entity using recursive CTE traversal (N-hop).

**Input:** `{"entity_id": 42, "max_hops": 2}`
**Returns:** Entity details + all connected entities and relationships within N hops.
**Use case:** Explore what an entity is connected to.

### `graph_communities`

Searches community summaries for thematic queries.

**Input:** `{"query": "nuclear safety", "level": 0, "limit": 5}`
**Returns:** Matching community titles, summaries, and member entity counts.
**Use case:** Answer broad/thematic questions about the corpus.

## Backfill Script

To build the graph for already-indexed documents:

```bash
uv run python scripts/graph_rag_build.py
```

The script:
1. Reads all page images from `data/vl_pages/`
2. Extracts entities and relationships via multimodal LLM
3. Stores them in the `graph` schema
4. Runs Leiden community detection
5. Generates community summaries

## Pipeline Integration

When uploading a new document via the web UI (`POST /documents/upload`), entity extraction runs automatically as the final stage (`graph_extraction`). It uses the same multimodal LLM as the backfill script.

If entity extraction fails for a page, it logs a warning and continues with the remaining pages. After 3 consecutive failures, extraction is aborted for that document (the rest of the indexing succeeds).

## Implementation

Key files:

| File | Purpose |
|------|---------|
| `src/graph/storage.py` | `GraphStorageAdapter` — PostgreSQL CRUD for entities, relationships, communities |
| `src/graph/entity_extractor.py` | Page image → entity/relationship extraction via multimodal LLM |
| `src/graph/community_detector.py` | Leiden algorithm via igraph |
| `src/graph/community_summarizer.py` | LLM-generated community titles and summaries |
| `src/agent/tools/graph_search.py` | Agent tool: trigram entity search |
| `src/agent/tools/graph_context.py` | Agent tool: N-hop relationship traversal |
| `src/agent/tools/graph_communities.py` | Agent tool: community summary search |
| `scripts/graph_rag_build.py` | Backfill script for existing documents |
