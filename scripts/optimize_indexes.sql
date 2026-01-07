-- ============================================================================
-- Tower News RAG - Index Optimization Script
-- ============================================================================
-- Run this in Supabase SQL Editor to optimize search performance.
--
-- Changes:
-- 1. Replace IVFFlat with HNSW index (better recall, auto-tuning)
-- 2. Add trigram index for fast fulltext search
-- 3. Add composite indexes for common query patterns
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- Step 1: Drop old IVFFlat index
-- ============================================================================
DROP INDEX IF EXISTS tower_knowledge_embedding_idx;

-- ============================================================================
-- Step 2: Create HNSW index for vector similarity search
-- ============================================================================
-- HNSW (Hierarchical Navigable Small World) provides:
-- - Better recall than IVFFlat
-- - No need to tune 'lists' parameter
-- - Faster queries for high-dimensional vectors
--
-- Parameters:
-- - m: Max connections per node (16 is good default)
-- - ef_construction: Build-time search width (higher = better quality, slower build)

CREATE INDEX IF NOT EXISTS tower_knowledge_embedding_hnsw_idx
ON tower_knowledge
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- Step 3: Create trigram index for fulltext search
-- ============================================================================
-- This enables fast ILIKE searches and fuzzy matching

CREATE INDEX IF NOT EXISTS tower_knowledge_content_trgm_idx
ON tower_knowledge
USING gin (content gin_trgm_ops);

-- Also index the title in metadata for faster title searches
CREATE INDEX IF NOT EXISTS tower_knowledge_metadata_title_idx
ON tower_knowledge
USING gin ((metadata->>'title') gin_trgm_ops);

-- ============================================================================
-- Step 4: Add composite indexes for common query patterns
-- ============================================================================

-- Index for filtering by post_type with score ordering
CREATE INDEX IF NOT EXISTS tower_knowledge_type_score_idx
ON tower_knowledge (post_type, score DESC);

-- Index for recent posts
CREATE INDEX IF NOT EXISTS tower_knowledge_ingested_idx
ON tower_knowledge (ingested_at DESC);

-- Index for post_id lookups (deduplication)
CREATE INDEX IF NOT EXISTS tower_knowledge_post_id_idx
ON tower_knowledge (post_id);

-- Partial index for high-score content only
CREATE INDEX IF NOT EXISTS tower_knowledge_high_score_idx
ON tower_knowledge (score DESC)
WHERE score > 50;

-- ============================================================================
-- Step 5: Create improved similarity search function
-- ============================================================================

CREATE OR REPLACE FUNCTION match_tower_knowledge_v2(
    query_embedding vector(1536),
    match_count int DEFAULT 10,
    filter_type text DEFAULT NULL,
    min_score int DEFAULT 0,
    similarity_threshold float DEFAULT 0.3
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    post_type TEXT,
    score INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tk.id,
        tk.content,
        tk.metadata,
        tk.post_type,
        tk.score,
        1 - (tk.embedding <=> query_embedding) AS similarity
    FROM tower_knowledge tk
    WHERE
        (filter_type IS NULL OR tk.post_type = filter_type)
        AND tk.score >= min_score
        AND (1 - (tk.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY tk.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- Step 6: Create fulltext search function
-- ============================================================================

CREATE OR REPLACE FUNCTION fulltext_search_tower_knowledge(
    search_query text,
    match_count int DEFAULT 10,
    filter_type text DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    post_type TEXT,
    score INTEGER,
    rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tk.id,
        tk.content,
        tk.metadata,
        tk.post_type,
        tk.score,
        similarity(tk.content, search_query) AS rank
    FROM tower_knowledge tk
    WHERE
        tk.content ILIKE '%' || search_query || '%'
        AND (filter_type IS NULL OR tk.post_type = filter_type)
    ORDER BY
        similarity(tk.content, search_query) DESC,
        tk.score DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- Step 7: Create hybrid search function (semantic + fulltext)
-- ============================================================================

CREATE OR REPLACE FUNCTION hybrid_search_tower_knowledge(
    query_embedding vector(1536),
    search_text text,
    match_count int DEFAULT 10,
    semantic_weight float DEFAULT 0.6,
    fulltext_weight float DEFAULT 0.4,
    filter_type text DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    post_type TEXT,
    score INTEGER,
    semantic_similarity FLOAT,
    fulltext_similarity FLOAT,
    combined_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH semantic AS (
        SELECT
            tk.id,
            tk.content,
            tk.metadata,
            tk.post_type,
            tk.score,
            1 - (tk.embedding <=> query_embedding) AS sem_score
        FROM tower_knowledge tk
        WHERE filter_type IS NULL OR tk.post_type = filter_type
        ORDER BY tk.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    fulltext AS (
        SELECT
            tk.id,
            similarity(tk.content, search_text) AS ft_score
        FROM tower_knowledge tk
        WHERE
            tk.content ILIKE '%' || search_text || '%'
            AND (filter_type IS NULL OR tk.post_type = filter_type)
        LIMIT match_count * 3
    )
    SELECT
        s.id,
        s.content,
        s.metadata,
        s.post_type,
        s.score,
        s.sem_score AS semantic_similarity,
        COALESCE(f.ft_score, 0) AS fulltext_similarity,
        (semantic_weight * s.sem_score + fulltext_weight * COALESCE(f.ft_score, 0)) AS combined_score
    FROM semantic s
    LEFT JOIN fulltext f ON s.id = f.id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- ============================================================================
-- Step 8: Analyze tables to update statistics
-- ============================================================================

ANALYZE tower_knowledge;

-- ============================================================================
-- Verification queries (run these to check the setup)
-- ============================================================================

-- Check indexes exist
-- SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'tower_knowledge';

-- Check extensions
-- SELECT * FROM pg_extension WHERE extname IN ('vector', 'pg_trgm');

-- Test semantic search
-- SELECT * FROM match_tower_knowledge_v2(
--     (SELECT embedding FROM tower_knowledge LIMIT 1),
--     5
-- );

-- Test fulltext search
-- SELECT * FROM fulltext_search_tower_knowledge('death wave', 5);

-- ============================================================================
-- Done!
-- ============================================================================
