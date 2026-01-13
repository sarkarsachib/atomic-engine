-- Atomic Knowledge System schema (pgvector + metrics)
-- Apply with PostgreSQL 14+ and pgvector installed.

CREATE EXTENSION IF NOT EXISTS vector;

-- Core vector store (projects, templates, patterns, etc.)
CREATE TABLE IF NOT EXISTS atomic_knowledge_vectors (
  namespace text NOT NULL,
  id text NOT NULL,
  text text NOT NULL,
  embedding vector(384) NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (namespace, id)
);

-- Similarity index for fast ANN search
CREATE INDEX IF NOT EXISTS atomic_knowledge_vectors_embedding_idx
  ON atomic_knowledge_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Project lineage / duplicate evolution (optional)
CREATE TABLE IF NOT EXISTS atomic_project_versions (
  project_id text PRIMARY KEY,
  parent_project_id text,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Template usage metrics
CREATE TABLE IF NOT EXISTS atomic_template_metrics (
  template_id text PRIMARY KEY,
  selected_count bigint NOT NULL DEFAULT 0,
  success_count bigint NOT NULL DEFAULT 0,
  fail_count bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Token/cost metrics per generation (optional)
CREATE TABLE IF NOT EXISTS atomic_generation_metrics (
  id bigserial PRIMARY KEY,
  project_id text,
  template_id text,
  is_rerun boolean NOT NULL DEFAULT false,
  tokens_used bigint NOT NULL DEFAULT 0,
  estimated_cost_usd numeric(12,6) NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);
