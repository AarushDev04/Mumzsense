-- =============================================================================
-- MumzSense v1 — Supabase: Truncate & Prepare for Re-indexing
-- =============================================================================
-- Run this in the Supabase SQL editor BEFORE re-running embed_and_index.py
-- when the posts table was previously populated with fallback embeddings.
--
-- STEP 1: Run this file in Supabase SQL editor
-- STEP 2: Set USE_REAL_EMBEDDINGS=true in your deployment environment
-- STEP 3: Re-run: USE_REAL_EMBEDDINGS=true python data/embed_and_index.py
-- STEP 4: (Optional) Run HNSW index creation below for production performance
-- =============================================================================

-- Clear all stale fallback-embedded posts
TRUNCATE TABLE posts;

-- Verify the table is empty
SELECT COUNT(*) AS post_count FROM posts;
-- Expected: 0

-- =============================================================================
-- After re-indexing with real embeddings, create HNSW indexes for fast search
-- (PRD §5, Step 5 acceptance criterion: similarity search < 200ms)
-- =============================================================================

-- EN embedding HNSW index (bge-large-en-v1.5, 1024 dimensions)
CREATE INDEX IF NOT EXISTS posts_en_embedding_hnsw
  ON posts USING hnsw (en_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- AR embedding HNSW index (multilingual-mpnet, 768 dimensions)
CREATE INDEX IF NOT EXISTS posts_ar_embedding_hnsw
  ON posts USING hnsw (ar_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Verify HNSW indexes exist
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'posts' AND indexname LIKE '%hnsw%';
