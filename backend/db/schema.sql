-- PostgreSQL + pgvector schema for MumzSense
-- Tables: posts, feedback_log, query_cache_log

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS posts (
	post_id uuid PRIMARY KEY,
	baby_age_weeks integer,
	stage text NOT NULL,
	topic text NOT NULL,
	urgency text NOT NULL,
	situation text NOT NULL,
	advice text NOT NULL,
	outcome text,
	trust_score double precision DEFAULT 0.75,
	lang text NOT NULL DEFAULT 'en',
	en_embedding vector(1024),
	ar_embedding vector(768),
	created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback_log (
	id bigserial PRIMARY KEY,
	query_hash text NOT NULL,
	user_rating integer,
	was_helpful boolean,
	urgency_felt text,
	created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_cache_log (
	query_hash text PRIMARY KEY,
	query_normalised text NOT NULL,
	cache_hits integer NOT NULL DEFAULT 0,
	last_hit timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_posts_stage_topic ON posts(stage, topic);
CREATE INDEX IF NOT EXISTS idx_posts_urgency ON posts(urgency);

CREATE OR REPLACE FUNCTION match_posts(
	query_embedding vector,
	embedding_col text,
	match_count int DEFAULT 10,
	filter_stage text DEFAULT NULL,
	filter_topic text DEFAULT NULL
)
RETURNS TABLE (
	post_id uuid,
	baby_age_weeks integer,
	stage text,
	topic text,
	urgency text,
	situation text,
	advice text,
	outcome text,
	trust_score double precision,
	lang text,
	distance double precision
)
LANGUAGE plpgsql
AS $$
BEGIN
	RETURN QUERY EXECUTE format(
		'SELECT
			post_id,
			baby_age_weeks,
			stage,
			topic,
			urgency,
			situation,
			advice,
			outcome,
			trust_score,
			lang,
			(%I <=> $1)::double precision AS distance
		 FROM posts
		 WHERE ($2 IS NULL OR stage = $2)
		   AND ($3 IS NULL OR topic = $3)
		 ORDER BY %I <=> $1
		 LIMIT $4',
		embedding_col,
		embedding_col
	)
	USING query_embedding, filter_stage, filter_topic, match_count;
END;
$$;

-- ── HNSW indexes for sub-200ms pgvector similarity search (PRD §5, Step 5) ──
-- Run after populating the posts table with real semantic embeddings.
-- These indexes are optional for correctness but required for production latency targets.
CREATE INDEX IF NOT EXISTS posts_en_embedding_hnsw
  ON posts USING hnsw (en_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS posts_ar_embedding_hnsw
  ON posts USING hnsw (ar_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
