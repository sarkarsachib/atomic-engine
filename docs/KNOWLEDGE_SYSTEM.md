# Atomic Knowledge System (RAG)

This repository includes a first-pass knowledge layer that learns from previous generations and uses retrieval to improve future output.

## What gets stored

Each `AtomicBrain.process()` call indexes a **project record** containing:

- the original idea + parsed structure
- selected modules
- a compact embedding "fingerprint" built from the idea + spec + prototype output
- detected patterns (architecture + stack + anti-pattern hints)
- template usage (if Forge selected a template)
- lineage (optional `parentProjectId` when forcing a re-run)

## Vector database

The knowledge layer supports two modes:

- **pgvector (preferred)**: enable by setting `DATABASE_URL` and (optionally) `ATOMIC_KNOWLEDGE_AUTO_MIGRATE=1`.
- **in-memory fallback**: used automatically if the pg client is unavailable or the database is down.

Similarity search is always available (pgvector when healthy, otherwise in-memory cosine similarity).

## Redis cache

If `REDIS_URL` is set, retrieved RAG context is cached in Redis. Otherwise an in-process TTL cache is used.

## Templates

Forge selects from a small built-in template catalog (`src/knowledge/templates.ts`).

Templates are embedded and indexed under the `templates` namespace and the selection is tracked in `knowledgeSystem.metrics`.

## Pattern recognition

On indexing, the system detects:

- architecture patterns (MVC, Microservices, Serverless, EventDriven, CLI, Monolith)
- common tech stack signals
- a small set of anti-pattern warnings

A lightweight in-memory knowledge graph captures relationships between projects and patterns.

## Optional HTTP endpoints

A minimal Node HTTP wrapper exists in `src/knowledge/http_api.ts` to support async indexing endpoints without committing to a framework.

## Key env vars

- `DATABASE_URL`: enables pgvector storage
- `ATOMIC_KNOWLEDGE_AUTO_MIGRATE=1`: auto-creates tables/indexes on startup
- `REDIS_URL`: enables Redis cache
- `ATOMIC_EMBED_DIM`: embedding dimensions (must match pgvector column dimension)
