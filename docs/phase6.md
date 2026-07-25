# TalentMatch AI — Phase 6 Documentation

## Overview

Phase 6 adds the Gradio UI (Spec 14) with three tabs: Admin (data ingestion), Recruiter (Case A), and Candidate (Case B). The UI is a thin client that calls the FastAPI backend's `/chat/*`, `/ingest/*`, and `/sessions/*` endpoints over HTTP.

**Status**: Complete.

**Depends on**: Phase 1 (ingestion), Phase 5 (conversational endpoints).

---

## Architecture

```
Gradio UI (port 7860)  ─── HTTP ───▶  FastAPI API (port 8000)
       │                                      │
   3 tabs:                             Endpoints:
   - Admin                             /ingest/batch
   - Recruiter                         /chat/recruiter
   - Candidate                         /chat/candidate
                                       /sessions/*
```

---

## UI Layout

```
TalentMatch AI
├── Tab 1: Admin - Data Ingestion
│   ├── File upload (multi-file, PDF/DOCX/TXT)
│   ├── File type selector (Resume / JD)
│   ├── Upload & Ingest button → POST /ingest/batch
│   ├── Batch status display (auto-refresh)
│   └── Document browser (candidates/JDs + parsed JSON)
│
├── Tab 2: Recruiter (Case A)
│   ├── JD/Query text input
│   ├── Search button → POST /chat/recruiter
│   ├── Results dataframe (Score, ID, Highlights, Skills)
│   ├── Chatbot for follow-ups
│   └── "Email All Active Candidates" button
│
└── Tab 3: Candidate (Case B)
    ├── Resume/Query text input
    ├── Search button → POST /chat/candidate
    ├── Results dataframe (Score, JD ID, Highlights, Skills)
    ├── Gap suggestions panel (Markdown)
    └── Chatbot for follow-ups
```

---

## Tab Details

### Tab 1: Admin (existing from Phase 1)

| Component | Function |
|-----------|----------|
| `gr.File` | Multi-file upload (PDF/DOCX/TXT) |
| `gr.Radio` | File type selector (resume/jd) |
| `gr.Button` | Upload & Ingest → `POST /ingest/batch` |
| `gr.Dataframe` | Batch status display |
| `gr.Dropdown` | Document browser (candidates/JDs) |
| `gr.JSON` | Parsed document viewer |

### Tab 2: Recruiter (Case A)

| Component | Function |
|-----------|----------|
| `gr.Textbox` | JD paste or free-text query |
| `gr.Button` | Search → `POST /chat/recruiter` |
| `gr.Dataframe` | Matched candidates (score, highlights, skills) |
| `gr.Chatbot` | Follow-up conversation |
| `gr.Button` | "Email All Active" → action-intent message |

**Flow:**
1. User pastes JD or types query → clicks "Search"
2. Backend classifies intent, invokes matching graph
3. Results appear in dataframe + chatbot shows confirmation
4. User types follow-ups: "remove candidate 2", "email remaining"

### Tab 3: Candidate (Case B)

| Component | Function |
|-----------|----------|
| `gr.Textbox` | Resume paste or free-text query |
| `gr.Button` | Search → `POST /chat/candidate` |
| `gr.Dataframe` | Matched JDs (score, highlights, skills) |
| `gr.Markdown` | Gap suggestions panel |
| `gr.Chatbot` | Follow-up conversation |

**Flow:**
1. User pastes resume or types query → clicks "Search"
2. Backend classifies intent, invokes matching + gap suggestion graphs
3. Results appear in dataframe + gap suggestions in Markdown panel
4. User types follow-ups: "remove job 3", "email me these openings"

---

## Running the UI

```bash
# Start the API first
uv run uvicorn talentmatch.main:app --host 0.0.0.0 --port 8000

# Then start the UI (separate terminal)
uv run python -m talentmatch.ui.app
```

The UI runs on `http://localhost:7860` and calls the API on `http://localhost:8000`.

---

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `API_BASE` | `http://localhost:8000` | FastAPI backend URL |

---

## Testing Checklist

- [ ] Admin tab: upload files → verify batch status updates
- [ ] Admin tab: browse candidates/JDs → verify parsed JSON displayed
- [ ] Recruiter tab: paste JD → verify candidates returned in dataframe
- [ ] Recruiter tab: type "Java developers" → verify free-text search works
- [ ] Recruiter tab: follow-up "remove candidate 1" → verify refinement
- [ ] Recruiter tab: click "Email All" → verify email action
- [ ] Candidate tab: paste resume → verify matching JDs returned
- [ ] Candidate tab: verify gap suggestions appear in Markdown panel
- [ ] Candidate tab: follow-up "email me these" → verify email action
- [ ] Verify `session_id` persists across follow-up messages

---

## What Comes Next (Phase 7+)

1. **Phase 7** — MCP tool server
2. **Phase 8** — RAGAS evaluation suite
3. **Phase 9** — Containerize (docker-compose DEV, then K8s UAT)
4. **Phase 10** — PROD promotion pipeline, observability polish
