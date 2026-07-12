# TalentMatch AI

A bidirectional Resume &harr; Job Description matching platform built with RAG (Retrieval-Augmented Generation), powered by FastAPI, Qdrant, MongoDB, and LLM-based reranking.

---

## Overview

TalentMatch AI solves two mirrored problems through a **single shared matching engine**:

- **Case A (Recruiter):** Given a Job Description, find and rank the best candidates from an ingested resume pool.
- **Case B (Candidate):** Given a resume, find and rank the best-matching job descriptions from an ingested JD pool.
- **Free-Text Search:** Keyword/skill-based search across either pool with no source document required.

This repository implements **Phase 1** of the system design &mdash; the core ingestion pipeline, vector storage, retrieve-then-rerank matching engine, and REST API. No LangGraph agents, conversational sessions, or UI are included yet.

---

## Architecture

```
                        +-------------------------------------------+
                        |           FastAPI (REST API)               |
                        |  /ingest/*  /match/*  /search/*           |
                        +-----------+---------------+---------------+
                                    |               |
                      +-------------v---+   +-------v-------------+
                      | Ingestion Pipeline|   |   Matching Engine    |
                      |                   |   |                     |
                      | Extract (PDF/DOCX)|   | Retrieve (Qdrant)   |
                      | Parse (LLM)       |   | Rerank (LLM)        |
                      | Chunk (section)   |   | Score + Rationale   |
                      | Embed (ST/remote) |   |                     |
                      | Persist (bulk)    |   |                     |
                      +---------+---------+   +---------+-----------+
                                |                       |
                      +---------v-----------------------v-----------+
                      |              Data Layer                      |
                      |  Qdrant (vectors) | MongoDB (documents)     |
                      +---------------------------------------------+
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Web Framework** | FastAPI | REST API with async support |
| **Vector Database** | Qdrant | Embedding storage and similarity search |
| **Document Store** | MongoDB + Beanie ODM | Metadata, parsed data, batch jobs, match records |
| **LLM** | LiteLLM (Groq / any provider) | Structured parsing, reranking, query expansion |
| **Embeddings** | sentence-transformers (local) or remote (Gemini, OpenAI) | Vector embeddings for chunks and queries |
| **PDF Parsing** | pypdf | PDF text extraction |
| **DOCX Parsing** | python-docx | Word document text extraction |
| **Metrics** | prometheus-client | Basic Prometheus observability |
| **Containerization** | Docker + docker-compose | Development environment |

---

## Project Structure

```
talentmatch/
├── src/talentmatch/
│   ├── main.py                 # FastAPI app, lifespan, startup logic
│   ├── config.py               # Settings via pydantic-settings (.env)
│   │
│   ├── db/
│   │   ├── mongo.py            # Beanie/Motor initialization
│   │   └── qdrant.py           # Qdrant client + collection setup
│   │
│   ├── models/
│   │   ├── enums.py            # EntityType, BatchStatus, ItemStatus, MatchDirection
│   │   ├── candidate.py        # Candidate document (name, email, parsed_json, ...)
│   │   ├── jd.py               # JD document (title, company, parsed_json, ...)
│   │   ├── match.py            # Match document (score, rationale, highlights, ...)
│   │   ├── embedding.py        # EmbeddingIndex document (entity to Qdrant point map)
│   │   └── batch_job.py        # BatchItem + BatchJob documents
│   │
│   ├── ingestion/
│   │   ├── extractor.py        # PDF/DOCX/TXT text extraction + SHA-256 hashing
│   │   ├── parser.py           # LLM-based structured parsing (resume + JD schemas)
│   │   ├── chunker.py          # Section-based chunking (skills, experience, education)
│   │   ├── embedder.py         # Local sentence-transformers or remote embedding API
│   │   └── worker.py           # Batch orchestrator: extract -> parse -> chunk -> embed -> persist
│   │
│   ├── matching/
│   │   ├── embedder.py         # Re-export of embed_text for convenience
│   │   ├── retriever.py        # Vector retrieval from Qdrant (section-based + hybrid)
│   │   └── reranker.py         # LLM-based re-ranking with structured scoring
│   │
│   └── routers/
│       ├── health.py           # GET /health, GET /metrics
│       ├── ingestion.py        # POST /ingest/batch, GET /ingest/batch/{id}/status
│       ├── matching.py         # POST /match/jd-to-candidates, POST /match/resume-to-jds
│       └── search.py           # POST /search/candidates, POST /search/jds
│
├── docker-compose.yml          # MongoDB + Qdrant + API containers
├── Dockerfile                  # Multi-stage Python 3.13 build
├── pyproject.toml              # Dependencies and project metadata
├── .env.example                # Environment variable template
└── talentmatch-ai-system-design.md  # Full system design specification
```

---

## What Phase 1 Implements

Phase 1 delivers the **core ingestion + matching engine** with no agent orchestration &mdash; the foundation all subsequent phases build on.

### Ingestion Pipeline

A batch-capable pipeline that processes uploaded resume and JD files end-to-end:

1. **Text Extraction** (`ingestion/extractor.py`) &mdash; Reads PDF (pypdf), DOCX (python-docx), and plain text files. Computes SHA-256 hash of each file for future idempotency checks.

2. **LLM Structured Parsing** (`ingestion/parser.py`) &mdash; Sends extracted text to an LLM with a structured extraction prompt. Returns typed JSON:
   - Resumes: `{name, email, skills[], experience[], education[], years_experience, raw_sections}`
   - JDs: `{title, company, required_skills[], nice_to_have[], years_experience_required, responsibilities[], raw_sections}`

3. **Section-Based Chunking** (`ingestion/chunker.py`) &mdash; Splits parsed documents by semantic section (skills, experience entries, education, responsibilities) rather than fixed-length windows. Sections retrieve better because they map to distinct query intents.

4. **Embedding** (`ingestion/embedder.py`) &mdash; Generates vector embeddings with automatic dispatch:
   - **Local:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) &mdash; zero API cost, runs in-process.
   - **Remote:** Any LiteLLM-supported embedding model (e.g., `gemini/gemini-embedding-001`) &mdash; higher quality, requires API key.

5. **Bulk Persistence** (`ingestion/worker.py`) &mdash; Writes all results in bulk operations:
   - Qdrant: bulk `upsert()` of all chunk vectors with payload metadata (`entity_id`, `section`, `text`).
   - MongoDB: `insert_many()` for `Candidate`/`JD` documents and `EmbeddingIndex` records.
   - Per-item status tracking through `BatchJob` &mdash; each file progresses through `queued -> extracted -> parsed -> embedded -> persisted` independently.

**Failure isolation:** One file's parse/embed failure marks that item `failed` with an error message; the rest of the batch continues unaffected.

### Matching Engine

A two-stage **retrieve-then-rerank** pipeline shared by all three query modes:

1. **Vector Retrieval** (`matching/retriever.py`) &mdash; Embeds query sections (skills, experience) separately, runs Qdrant similarity search per section, then aggregates results by entity using max-score pooling. Returns the top ~20 candidates/JDs.

2. **LLM Re-ranking** (`matching/reranker.py`) &mdash; Passes the retrieved entities and query text to an LLM with a structured scoring prompt. Returns for each entity:
   - `score` (0-100)
   - `matched_skills[]` / `missing_skills[]`
   - `highlights[]` (notable facts like "6 yrs backend Java", "AWS certified")
   - `rationale` (explainable match reasoning)

   The reranker returns the **top 5** results sorted by score.

### Three Query Modes

| Mode | Entry Point | How It Works |
|---|---|---|
| **JD -> Candidates** (Case A) | `POST /match/jd-to-candidates` | Pass a JD (by ID or raw text). Sections are retrieved, then LLM-reranked against the candidate pool. |
| **Resume -> JDs** (Case B) | `POST /match/resume-to-jds` | Pass a resume (by ID or raw text). Skills and experience are retrieved, then LLM-reranked against the JD pool. |
| **Free-Text Search** | `POST /search/candidates` or `POST /search/jds` | Pass a raw query string. An LLM expands the query into related terms, then hybrid search retrieves, followed by the same LLM reranking step. |

---

## API Endpoints

### Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/batch` | Upload multiple resume/JD files. Returns `batch_id` for status polling. |
| `GET` | `/ingest/batch/{batch_id}/status` | Per-item and overall ingestion progress. |

### Matching

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/match/jd-to-candidates` | Match a JD against the candidate pool. Accepts `entity_id` or `entity_text`. |
| `POST` | `/match/resume-to-jds` | Match a resume against the JD pool. Accepts `entity_id` or `entity_text`. |

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search/candidates` | Free-text skill search across candidates. |
| `POST` | `/search/jds` | Free-text skill search across job descriptions. |

### Data Access

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/candidates/{id}` | Fetch a candidate by ID. |
| `GET` | `/jds/{id}` | Fetch a JD by ID. |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check. |
| `GET` | `/metrics` | Prometheus metrics. |

---

## Data Model (MongoDB via Beanie)

| Collection | Key Fields | Purpose |
|---|---|---|
| `candidates` | `name`, `email`, `resume_raw_text`, `parsed_json` | Ingested candidate profiles |
| `jds` | `title`, `company`, `jd_raw_text`, `parsed_json` | Ingested job descriptions |
| `embeddings_index` | `entity_type`, `entity_id`, `qdrant_point_id`, `chunk_text` | Maps MongoDB entities to Qdrant vector points |
| `matches` | `jd_id`, `candidate_id`, `score`, `rationale`, `highlights[]`, `matched_skills[]`, `missing_skills[]`, `direction` | Match results (model defined, persistence pending) |
| `batch_jobs` | `status`, `items[]`, `total_items`, `completed_items` | Batch ingestion tracking with per-item status |

**Qdrant Collections:**
- `candidate_chunks` &mdash; Resume chunks with payload: `{candidate_id, section, text}`
- `jd_chunks` &mdash; JD chunks with payload: `{jd_id, section, text}`

---

## Getting Started

### Prerequisites

- Python 3.13+
- Docker and Docker Compose
- A Groq API key (or any LiteLLM-supported LLM provider)

### 1. Clone and configure

```bash
git clone <repo-url>
cd talentmatch
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Start infrastructure + API

```bash
docker-compose up --build
```

This starts:
- **MongoDB** on `localhost:27017`
- **Qdrant** on `localhost:6333`
- **API** on `localhost:8000`

### 3. Verify

```bash
curl http://localhost:8000/health
```

---

## Usage Examples

### Ingest files

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -F "resumes=@resume1.pdf" \
  -F "resumes=@resume2.docx" \
  -F "jds=@job_posting1.pdf"
```

Check status:
```bash
curl http://localhost:8000/ingest/batch/<batch_id>/status
```

### Match JD to candidates

```bash
curl -X POST http://localhost:8000/match/jd-to-candidates \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "<jd_mongo_id>"
  }'
```

Or with raw text:
```bash
curl -X POST http://localhost:8000/match/jd-to-candidates \
  -H "Content-Type: application/json" \
  -d '{
    "entity_text": "Senior Backend Engineer with 5+ years Python, FastAPI, PostgreSQL, AWS"
  }'
```

### Match resume to JDs

```bash
curl -X POST http://localhost:8000/match/resume-to-jds \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "<candidate_mongo_id>"
  }'
```

### Free-text skill search

```bash
curl -X POST http://localhost:8000/search/candidates \
  -H "Content-Type: application/json" \
  -d '{"query_text": "Java Spring Boot microservices"}'
```

---

## Configuration

All settings are configured via environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | &mdash; | API key for Groq LLM provider |
| `LLM_MODEL` | `groq/llama-3.3-70b-versatile` | LiteLLM model string for parsing/reranking |
| `EMBEDDING_MODEL` | `gemini/gemini-embedding-001` | Embedding model (local name or remote `provider/model`) |
| `EMBEDDING_DIMENSION` | `768` | Vector dimension (must match model) |
| `EMBEDDING_BATCH_SIZE` | `100` | Texts per embedding API call |
| `MONGODB_URI` | `mongodb://localhost:27017/talentmatch` | MongoDB connection string |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION_CANDIDATE` | `candidate_chunks` | Qdrant collection for resume chunks |
| `QDRANT_COLLECTION_JD` | `jd_chunks` | Qdrant collection for JD chunks |
| `HOST` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |

**Local vs Remote Embeddings:** If `EMBEDDING_MODEL` contains a `/` (e.g., `gemini/gemini-embedding-001`), the system calls a remote embedding API. If it is a bare model name (e.g., `all-MiniLM-L6-v2`), it loads a local `SentenceTransformer` model &mdash; no API key needed, runs fully offline.

---

## Data Flow Diagrams

### Ingestion Flow

```
Upload (PDF/DOCX/TXT)
  |
  v
extract_text()           -- pypdf / python-docx / plain read
  |
  v
parse_resume/parse_jd()  -- LLM structured extraction (litellm)
  |
  v
chunk_resume/chunk_jd()  -- Section-based chunking
  |
  v
embed_texts()            -- sentence-transformers or remote API
  |
  +---> Qdrant bulk upsert   (vectors + payload metadata)
  |
  +---> MongoDB insert_many  (Candidate/JD + EmbeddingIndex docs)
```

### Matching Flow (Case A / Case B)

```
JD ID or raw text
  |
  v
Load from MongoDB        -- or use raw text directly
  |
  v
Extract sections         -- skills, experience, responsibilities from parsed_json
  |
  v
retrieve_candidates()    -- Qdrant similarity search per section, max-pool by entity
  |
  v
rerank()                 -- LLM scores top ~20 entities, returns top 5
  |
  v
MatchResponse            -- {score, rationale, highlights[], matched_skills[], missing_skills[]}
```

### Free-Text Search Flow

```
Raw query string
  |
  v
expand_query()           -- LLM expands into related technologies
  |
  v
hybrid_search()          -- Embed combined query, Qdrant vector search
  |
  v
rerank()                 -- LLM scores top ~20 entities, returns top 5
  |
  v
SearchResponse           -- {score, rationale, highlights[], matched_skills[], missing_skills[]}
```

---

## Development

### Run locally without Docker

```bash
# Start only infrastructure
docker-compose up -d mongo qdrant

# Run API locally
pip install -e .
uvicorn talentmatch.main:app --reload
```

### Project conventions

- **Pydantic everywhere** -- every model, request, response, and state object is a Pydantic model.
- **Beanie ODM** -- MongoDB access via Beanie `Document` classes, not raw PyMongo.
- **Typed state** -- the `MatchDirection` enum and typed `BatchItem` models ensure data flows are auditable.
- **Bulk operations** -- ingestion writes to Qdrant and MongoDB in bulk, not per-document.

---

## What Comes Next (Phase 2+)

Phase 1 provides the building blocks. Future phases add:

| Phase | Scope |
|---|---|
| **Phase 2** | LangGraph agent workflows &mdash; wrap matching pipeline in JD->Candidates (7.1) and Resume->JDs (7.2) graphs |
| **Phase 3** | Free-text search mode with query expansion + hybrid retrieval (already partially implemented in Phase 1) |
| **Phase 4** | Gap suggestion agent &mdash; concrete resume improvement advice |
| **Phase 5** | Conversational layer &mdash; sessions, intent routing, refinement, action graphs |
| **Phase 6** | Gradio UI &mdash; Admin ingestion tab, Recruiter tab, Candidate tab |
| **Phase 7** | MCP tool server &mdash; expose tools over JSON-RPC 2.0 |
| **Phase 8** | RAGAS evaluation suite + CI gate |
| **Phase 9** | Docker-compose DEV environment, Kubernetes UAT |
| **Phase 10** | PROD promotion pipeline, observability polish |

---

## License

This is a portfolio project for learning purposes.