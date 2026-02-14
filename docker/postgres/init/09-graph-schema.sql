-- Graph RAG Schema
-- Knowledge graph for cross-document entity/relationship reasoning.
-- Uses graph schema, separate from vectors schema.

CREATE SCHEMA IF NOT EXISTS graph;

-- Ensure pg_trgm extension for fuzzy text search on entity names
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Entities extracted from page images via multimodal LLM
CREATE TABLE graph.entities (
    entity_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- REGULATION, SECTION, ORGANIZATION, CONCEPT, etc.
    description TEXT,
    source_page_id TEXT REFERENCES vectors.vl_pages(page_id) ON DELETE SET NULL,
    document_id TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(name, entity_type, document_id)
);

-- Relationships between entities
CREATE TABLE graph.relationships (
    relationship_id SERIAL PRIMARY KEY,
    source_entity_id INT REFERENCES graph.entities(entity_id) ON DELETE CASCADE,
    target_entity_id INT REFERENCES graph.entities(entity_id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,  -- DEFINES, REFERENCES, AMENDS, REQUIRES, etc.
    description TEXT,
    weight FLOAT DEFAULT 1.0,
    source_page_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Leiden community detection results
CREATE TABLE graph.communities (
    community_id SERIAL PRIMARY KEY,
    level INT NOT NULL,           -- Leiden hierarchy level (0 = finest)
    title TEXT,
    summary TEXT,
    summary_model TEXT,
    entity_ids INT[] NOT NULL,    -- Array of entity_id
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_entities_type ON graph.entities(entity_type);
CREATE INDEX idx_entities_document ON graph.entities(document_id);
CREATE INDEX idx_entities_name_trgm ON graph.entities USING gin(name gin_trgm_ops);
CREATE INDEX idx_relationships_source ON graph.relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON graph.relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON graph.relationships(relationship_type);
CREATE INDEX idx_communities_level ON graph.communities(level);
