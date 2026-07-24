# TalentMatch AI — Phase 1 Documentation

## Overview

Phase 1 implements the core ingestion + matching pipeline without LangGraph agents. It provides a working retrieve-then-rerank matching system with FastAPI endpoints and a Gradio admin UI.

**Status**: Complete and tested end-to-end.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  FastAPI    │────▶│  Ingestion   │────▶│   Qdrant     │
│  Endpoints  │     │  Worker      │     │  (embeddings)│
│             │     │              │     └──────────────┘
│             │────▶│  Matching    │           │
│             │     │  Core        │     ┌──────────────┐
└──────┬──────┘     └──────────────┘     │   MongoDB    │
       │                                  │  (metadata)  │
┌──────▼──────┐                           └──────────────┘
│  Gradio UI  │
│  (Admin)    │
└─────────────┘
```

### Services
- **API** — FastAPI app exposing REST endpoints
- **Worker** — async ingestion (parse, embed, persist)
- **Qdrant** — vector store for embeddings (local mode)
- **MongoDB** — document store for metadata/scores (Atlas free tier)
- **UI** — Gradio admin app (port 7860)

---

## Project Structure

```
src/talentmatch/
├── config.py              # pydantic-settings (env vars)
├── main.py                # FastAPI app factory + lifespan
├── db/
│   ├── mongo.py           # Beanie ODM init
│   └── qdrant.py          # Qdrant client (singleton)
├── models/
│   ├── enums.py           # EntityType, BatchStatus, ItemStatus, MatchDirection
│   ├── candidate.py       # Candidate document
│   ├── jd.py              # JD document
│   ├── batch_job.py       # BatchJob + BatchItem
│   ├── match.py           # Match document
│   └── embedding.py       # EmbeddingIndex document
├── ingestion/
│   ├── extractor.py       # PDF/DOCX/TXT text extraction
│   ├── parser.py          # LLM structured parsing (Groq)
│   ├── chunker.py         # Section-based chunking
│   ├── embedder.py        # sentence-transformers (local) or LiteLLM (remote)
│   └── worker.py          # Batch orchestration
├── matching/
│   ├── retriever.py       # Qdrant retrieve + hybrid search
│   └── reranker.py        # LLM rerank with fallback
├── routers/
│   ├── health.py          # /health, /metrics
│   ├── ingestion.py       # /ingest/batch, /ingest/batch/{id}/status
│   ├── matching.py        # /match/jd-to-candidates, /match/resume-to-jds
│   └── search.py          # /search/candidates, /search/jds
├── utils/
│   ├── logging.py         # Structured logging + trace IDs
│   └── llm.py             # LLM retry wrapper (tenacity)
└── ui/
    ├── app.py             # Gradio app entry point
    └── admin.py           # Admin tab (upload, browse)
```

---

## Data Models (Beanie ODM)

### Candidate
| Field | Type | Description |
|---|---|---|
| `name` | str | Candidate name |
| `email` | str | Email address |
| `resume_raw_text` | str | Raw extracted text |
| `resume_file_path` | str | Original filename |
| `parsed_json` | dict | LLM-parsed structured data |
| `file_hash` | str | SHA-256 of file bytes (idempotency) |
| `created_at` | datetime | Creation timestamp |

### JD
| Field | Type | Description |
|---|---|---|
| `title` | str | Job title |
| `company` | str | Company name |
| `jd_raw_text` | str | Raw extracted text |
| `jd_file_path` | str | Original filename |
| `parsed_json` | dict | LLM-parsed structured data |
| `file_hash` | str | SHA-256 of file bytes (idempotency) |
| `created_at` | datetime | Creation timestamp |

### BatchJob
| Field | Type | Description |
|---|---|---|
| `entity_type` | EntityType? | None (mixed batch) |
| `status` | BatchStatus | queued/processing/completed/failed |
| `items` | list[BatchItem] | Per-file status |
| `total_items` | int | Total files |
| `completed_items` | int | Files processed |
| `created_at` | datetime | Creation timestamp |
| `updated_at` | datetime | Last update |

### Match
| Field | Type | Description |
|---|---|---|
| `jd_id` | str? | Reference to JD |
| `candidate_id` | str? | Reference to Candidate |
| `query_text` | str | Input text used for matching |
| `expanded_query_terms` | list[str] | Query expansion terms |
| `score` | float | Match score (0-100) |
| `rationale` | str | LLM explanation |
| `highlights` | list[str] | Notable facts |
| `matched_skills` | list[str] | Skills that matched |
| `missing_skills` | list[str] | Skills missing |
| `direction` | MatchDirection | Match direction |
| `created_at` | datetime | Creation timestamp |

### EmbeddingIndex
| Field | Type | Description |
|---|---|---|
| `entity_type` | EntityType | candidate/jd |
| `entity_id` | str | Reference to entity |
| `qdrant_point_id` | str | Qdrant point ID |
| `chunk_text` | str | Chunk text |
| `created_at` | datetime | Creation timestamp |

---

## Ingestion Pipeline

### Flow
```
File Upload → Extract Text → LLM Parse → Chunk by Section → Embed → Bulk Write
```

### Three Levels of Batching
1. **Request-level**: N files → 1 `BatchJob` document
2. **Embedding-call**: All chunks embedded in `EMBEDDING_BATCH_SIZE` groups
3. **Write-level**: Bulk upsert to Qdrant + `insert_many` to MongoDB

### Idempotency
- File hash (SHA-256) computed on upload
- Before insert, checks for existing record with same `file_hash`
- If found: deletes old Qdrant points + EmbeddingIndex + entity doc
- Then inserts new record (delete-old + insert-new pattern)

### Failure Isolation
- Per-file try/except in batch processing
- Failed items marked `ItemStatus.failed` with error message
- Rest of batch continues

### Chunking Strategy
**Resumes:**
- `skills` → single chunk with all skills
- `experience` → one chunk per job (title + company + description)
- `education` → one chunk per degree

**JDs:**
- `required_skills` → single chunk
- `nice_to_have` → single chunk
- `responsibilities` → one chunk per responsibility

---

## Matching Core

### Retrieve-then-Rerank Pattern
1. **Retrieve**: Embed query sections (skills, experience), search Qdrant top-N per section
2. **Aggregate**: Max-pooling across sections per entity
3. **Rerank**: LLM scores top ~20 candidates with structured prompt
4. **Return**: Top 5 by score

### Rerank Output Schema
```json
{
  "entity_id": "string",
  "score": 0-100,
  "matched_skills": ["skill1"],
  "missing_skills": ["skill1"],
  "highlights": ["6 yrs backend Java"],
  "rationale": "one-sentence explanation"
}
```

### LLM Fallback
- If LLM rerank fails: returns similarity-only scores with `rerank_skipped: True`
- Allows system to degrade gracefully when LLM is down

### Query Expansion (Free-Text Search)
- LLM expands raw query into related terms
- Example: "Java" → ["Java", "Spring Boot", "Hibernate", "Microservices"]
- Combined query used for hybrid search

### Hybrid Search
- Vector search on expanded query
- Optional filter on `skills` section
- Deduplication by entity_id, keep highest score

---

## API Endpoints

### Health & Metrics
```
GET /health          → {"status": "ok"}
GET /metrics         → Prometheus metrics (text/plain)
```

### Ingestion
```
POST /ingest/batch              (multipart: resumes[] + jds[])
  → {"batch_id": "...", "total_files": N}

GET  /ingest/batch/{batch_id}/status
  → {"batch_id", "status", "total_items", "completed_items", "items[]"}
```

### Matching
```
POST /match/jd-to-candidates    { entity_id | entity_text, top_k=5, notify=false }
  → {"matches": [{"entity_id", "score", "rationale", ...}]}

POST /match/resume-to-jds       { entity_id | entity_text, top_k=5, notify=false }
  → {"matches": [{"entity_id", "score", "rationale", ...}]}
```

### Search (Free-Text)
```
POST /search/candidates         { query_text, top_k=5 }
  → {"matches": [...]}

POST /search/jds                { query_text, top_k=5 }
  → {"matches": [...]}
```

### Documents
```
GET  /candidates                → [{"id", "name", "email", "created_at"}]
GET  /candidates/{id}           → full Candidate document
GET  /jds                       → [{"id", "title", "company", "created_at"}]
GET  /jds/{id}                  → full JD document
```

---

## Configuration (.env)

```bash
# LLM
GROQ_API_KEY=gsk_...
LLM_MODEL=groq/llama-3.3-70b-versatile

# Embedding (local, no API key)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=100

# MongoDB
MONGODB_URI=mongodb+srv://... (Atlas) or mongodb://localhost:27017/talentmatch (local)

# Qdrant
QDRANT_MODE=local              # "local" for dev, "remote" for Docker
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_CANDIDATE=candidate_chunks
QDRANT_COLLECTION_JD=jd_chunks

# Server
HOST=0.0.0.0
PORT=8000
```

---

## Observability

### Structured Logging
- Format: `TIMESTAMP | LEVEL | LOGGER | trace=TRACE_ID | MESSAGE`
- Trace IDs propagated via `X-Trace-ID` header (auto-generated if not provided)
- `TraceIDMiddleware` sets trace ID per request, stored in `ContextVar`

### Key Log Events
- Batch start/complete
- Per-file processed/failed
- LLM retries (warning level)
- LLM fallback triggered (warning level)

---

## Running

### Local Development
```bash
# Install dependencies
uv pip install -e .

# Start API server
uv run uvicorn talentmatch.main:app --reload --port 8000

# Start UI (separate terminal)
uv run python -m talentmatch.ui.app
```

### Docker
```bash
docker-compose up --build
```

---

## Dependencies

| Package | Purpose |
|---|---|
| fastapi | API framework |
| uvicorn | ASGI server |
| beanie | MongoDB ODM |
| motor | Async MongoDB driver |
| qdrant-client | Vector store |
| litellm | LLM gateway (Groq) |
| sentence-transformers | Local embeddings |
| pypdf | PDF extraction |
| python-docx | DOCX extraction |
| tenacity | Retry/backoff |
| prometheus-client | Metrics |
| gradio | Admin UI |
| httpx | HTTP client (UI → API) |

---

## Known Limitations (Phase 1)

1. **No conversational layer** — single-shot endpoints, no sessions/follow-ups (Phase 5)
2. **No LangGraph agents** — plain function calls, no agent orchestration (Phase 2)
3. **No MCP server** — tools not exposed via MCP (Phase 7)
4. **No RAGAS evaluation** — no automated quality metrics (Phase 8)
5. **No auth** — API is open (Phase 10 scope)
6. **Sequential ingestion** — files processed one at a time within a batch
7. **Fixed rerank count** — always returns top 5, not configurable per request

---

## Testing Checklist

- [ ] Upload single resume → verify parsed_json in MongoDB
- [ ] Upload single JD → verify parsed_json in MongoDB
- [ ] Upload batch (mix of resumes + JDs) → verify all items processed
- [ ] Re-upload same file → verify idempotency (no duplicate)
- [ ] `POST /match/jd-to-candidates` with entity_id → verify matches returned
- [ ] `POST /match/resume-to-jds` with entity_id → verify matches returned
- [ ] `POST /search/candidates` with query → verify results
- [ ] `POST /search/jds` with query → verify results
- [ ] Check `matches` collection in MongoDB → records persisted
- [ ] Admin UI → upload files, check status, browse documents
- [ ] Kill LLM (invalid API key) → verify fallback returns similarity-only scores
