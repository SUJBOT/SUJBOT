-- Graph RAG Schema
-- Knowledge graph for cross-document entity/relationship reasoning.
-- Uses graph schema, separate from vectors schema.

CREATE SCHEMA IF NOT EXISTS graph;

-- pg_trgm for trigram index on entity names (fast prefix/substring matching)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- unaccent for diacritics-insensitive search (bezpecnost → bezpečnost)
CREATE EXTENSION IF NOT EXISTS unaccent;

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
    search_tsv TSVECTOR,
    search_embedding vector(384),  -- multilingual-e5-small
    UNIQUE(name, entity_type, document_id)
);

-- Entity aliases: tracks alternative names, abbreviations, translations
-- Prevents re-creation of merged entities under non-canonical names
CREATE TABLE graph.entity_aliases (
    alias_id    SERIAL PRIMARY KEY,
    entity_id   INT NOT NULL REFERENCES graph.entities(entity_id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    alias_type  TEXT NOT NULL DEFAULT 'variant',
        -- 'variant', 'abbreviation', 'translation', 'acronym', 'former_name'
    language    TEXT,            -- NULL = same as source, 'en', 'cs'
    source      TEXT,           -- 'exact_dedup', 'semantic_dedup', 'extraction', 'manual'
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX idx_entity_aliases_unique ON graph.entity_aliases (lower(alias), entity_id);
CREATE INDEX idx_entity_aliases_lookup ON graph.entity_aliases (lower(alias));
CREATE INDEX idx_entity_aliases_entity ON graph.entity_aliases (entity_id);

-- Relationships between entities
CREATE TABLE graph.relationships (
    relationship_id SERIAL PRIMARY KEY,
    source_entity_id INT REFERENCES graph.entities(entity_id) ON DELETE CASCADE,
    target_entity_id INT REFERENCES graph.entities(entity_id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,  -- DEFINES, REFERENCES, AMENDS, REQUIRES, etc.
    description TEXT,
    weight FLOAT DEFAULT 1.0,
    source_page_id TEXT,
    search_embedding vector(384),  -- multilingual-e5-small
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
    search_tsv TSVECTOR,
    search_embedding vector(384),  -- multilingual-e5-small
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Auto-populate search_tsv on entity INSERT/UPDATE
CREATE OR REPLACE FUNCTION graph.entities_search_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_tsv := to_tsvector('simple', unaccent(
        coalesce(NEW.name, '') || ' ' || coalesce(NEW.description, '')
    ));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_entities_search_tsv
    BEFORE INSERT OR UPDATE OF name, description ON graph.entities
    FOR EACH ROW EXECUTE FUNCTION graph.entities_search_tsv_trigger();

-- Auto-populate search_tsv on community INSERT/UPDATE
CREATE OR REPLACE FUNCTION graph.communities_search_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_tsv := to_tsvector('simple', unaccent(
        coalesce(NEW.title, '') || ' ' || coalesce(NEW.summary, '')
    ));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_communities_search_tsv
    BEFORE INSERT OR UPDATE OF title, summary ON graph.communities
    FOR EACH ROW EXECUTE FUNCTION graph.communities_search_tsv_trigger();

-- Indexes
CREATE INDEX idx_entities_type ON graph.entities(entity_type);
CREATE INDEX idx_entities_document ON graph.entities(document_id);
CREATE INDEX idx_entities_name_trgm ON graph.entities USING gin(name gin_trgm_ops);
CREATE INDEX idx_relationships_source ON graph.relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON graph.relationships(target_entity_id);
CREATE INDEX idx_relationships_type ON graph.relationships(relationship_type);
CREATE INDEX idx_communities_level ON graph.communities(level);
CREATE INDEX idx_entities_search_tsv ON graph.entities USING gin(search_tsv);
CREATE INDEX idx_communities_search_tsv ON graph.communities USING gin(search_tsv);
CREATE INDEX idx_entities_embedding ON graph.entities USING hnsw(search_embedding vector_cosine_ops);
CREATE INDEX idx_relationships_embedding ON graph.relationships USING hnsw(search_embedding vector_cosine_ops);
CREATE INDEX idx_communities_embedding ON graph.communities USING hnsw(search_embedding vector_cosine_ops);
