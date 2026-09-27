-- App-level PostgreSQL initialisation script.
-- This runs ONCE when the pgvector container is first created.
-- Hindsight has its own separate container/DB — do NOT add Hindsight tables here.

-- Enable the pgvector extension for future semantic-search features
-- (e.g. storing patent embeddings alongside the cache).
CREATE EXTENSION IF NOT EXISTS vector;

-- The ORM creates patent_cache and request_log automatically via SQLAlchemy.
-- This script exists as a belt-and-suspenders safeguard and for readability.
