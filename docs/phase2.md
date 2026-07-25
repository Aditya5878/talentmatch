# TalentMatch AI — Phase 2 Documentation

## Overview

Phase 2 wraps the Phase 1 matching pipeline in LangGraph agent workflows. It converts the plain function-call matching core into typed, testable graph-based pipelines with conditional branching, error handling, and notification support.

**Status**: Complete — JD→Candidates (Case A) and Resume→JDs (Case B) graphs implemented.

**Depends on**: Phase 1 (ingestion pipeline, matching core, MongoDB/Qdrant infrastructure).

---

## What Changed from Phase 1

| Area | Phase 1 | Phase 2 |
|------|---------|---------|
| Matching | Direct function calls from routers | LangGraph StateGraph with typed state |
| Error handling | Try/except in router | Conditional edges — graph routes to END on error |
| Notification | Not implemented | Dry-run email logging (live mode placeholder) |
| State management | None | Pydantic state models flowing through graph nodes |
| New dependency | — | `langgraph>=0.2.0` |

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│  FastAPI     │────▶│  LangGraph Agent                         │
│  /agent/*    │     │                                          │
│  Endpoints   │     │  parse → retrieve → rerank → persist     │
│              │     │                          ↓                │
│              │     │                    [notify] (conditional)  │
└──────┬───────┘     └──────────────────────────────────────────┘
       │                              │
       │                     ┌────────▼────────┐
       │                     │  Phase 1 Core   │
       │                     │  (retriever,    │
       │                     │   reranker,     │
       │                     │   parser)       │
       │                     └────────┬────────┘
       │                              │
┌──────▼───────┐             ┌────────▼────────┐
│  Gradio UI  │             │  Qdrant + MongoDB│
└──────────────┘             └─────────────────┘
```

### Key Design Decisions

- **LangGraph over plain functions**: Each node is a pure function with typed state — testable in isolation, composable across graphs, and ready for checkpointing (Phase 5 conversational layer).
- **Reusable nodes**: `rerank_score_node` and `persist_matches_node` are shared between both graphs — only the parse and retrieve nodes differ.
- **Error routing**: Every edge has a conditional `_check_error` branch — if any node sets `state.error`, the graph short-circuits to END instead of continuing with bad data.
- **Notification default**: Case B always sends email (resume→JDs), Case A only sends if `notify=True` (JD→candidates).

---

## Project Structure (Phase 2 additions)

```
src/talentmatch/
├── agents/                        # NEW — LangGraph agent workflows
│   ├── __init__.py                # Exports graph singletons
│   ├── state.py                   # Pydantic state models for LangGraph
│   ├── nodes.py                   # Reusable node functions (pure functions)
│   └── graphs.py                  # Graph definitions + compiled singletons
├── models/
│   └── email_log.py               # NEW — EmailLog Beanie document
├── notification.py                # NEW — dry_run email sending
├── routers/
│   └── agent.py                   # NEW — /agent/* API endpoints
└── config.py                      # MODIFIED — added email_mode setting
```

---

## LangGraph State Models

### BaseGraphState (shared)

Both graphs inherit from `BaseGraphState`, which carries data through every node:

| Field | Type | Populated By | Used By |
|-------|------|-------------|---------|
| `entity_id` | str? | API request | parse, persist, notify |
| `entity_text` | str? | API request | parse |
| `top_k` | int | API request | (future: retrieve) |
| `notify` | bool | API request | notify branching |
| `parsed_json` | dict | parse node | retrieve, persist |
| `raw_text` | str | parse node | rerank, persist |
| `skills_text` | str | parse node | retrieve |
| `experience_texts` | list[str] | parse node | retrieve |
| `retrieved_entities` | list[dict] | retrieve node | rerank |
| `reranked_results` | list[MatchResult] | rerank node | persist, notify |
| `persisted_match_ids` | list[str] | persist node | (logging) |
| `email_logs` | list[EmailLogEntry] | notify node | (logging) |
| `error` | str? | any node | error routing |

### MatchResult

```python
class MatchResult(BaseModel):
    entity_id: str
    score: float = 0
    rationale: str = ""
    highlights: list[str] = []
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    rerank_skipped: bool = False
```

### EmailLogEntry

```python
class EmailLogEntry(BaseModel):
    recipient: str
    subject: str
    body: str
    mode: Literal["dry_run", "live"] = "dry_run"
    status: str = "pending"
```

---

## Graph Definitions

### Case A: JD → Candidates Graph

```
parse_jd ──(no error)──▶ retrieve_candidates ──(no error)──▶ rerank_score
    │                        │                                    │
    │ (error)                │ (error)                            │ (error)
    ▼                        ▼                                    ▼
   END                     END                                  END
                                                                    │ (no error)
                                                                    ▼
                                                              persist_matches
                                                                    │
                                                    ┌───────────────┴───────────────┐
                                                    │ (notify=True)                 │ (notify=False)
                                                    ▼                               ▼
                                              notify_candidates                    END
                                                    │
                                                    ▼
                                                   END
```

**Nodes:**
1. `parse_jd` — Load JD from MongoDB (by entity_id) or parse raw text via LLM
2. `retrieve_candidates` — Multi-query search against candidate collection in Qdrant
3. `rerank_score` — LLM-based scoring with structured output
4. `persist_matches` — Save Match documents to MongoDB
5. `notify_candidates` — (conditional) Send emails to matched candidates

### Case B: Resume → JDs Graph

```
parse_resume ──(no error)──▶ retrieve_jds ──(no error)──▶ rerank_score
    │                          │                              │
    │ (error)                  │ (error)                      │ (error)
    ▼                          ▼                              ▼
   END                       END                            END
                                                              │ (no error)
                                                              ▼
                                                        persist_matches
                                                              │ (no error)
                                                              ▼
                                                       notify_candidate
                                                              │
                                                              ▼
                                                             END
```

**Nodes:**
1. `parse_resume` — Load Candidate from MongoDB or parse raw text via LLM
2. `retrieve_jds` — Multi-query search against JD collection in Qdrant
3. `rerank_score` — LLM-based scoring (shared with Case A)
4. `persist_matches` — Save Match documents (shared with Case A)
5. `notify_candidate` — (always) Email candidate their matching JDs

---

## Node Functions (Reusable)

All nodes follow the pattern: `async def node(state: TypedState) -> dict`

| Node | Input State | Output State Update | Shared? |
|------|------------|-------------------|---------|
| `parse_jd_node` | JDToCandidatesState | parsed_json, raw_text, skills_text, experience_texts | No |
| `parse_resume_node` | ResumeToJDsState | parsed_json, raw_text, skills_text, experience_texts | No |
| `retrieve_candidates_node` | JDToCandidatesState | retrieved_entities | No |
| `retrieve_jds_node` | ResumeToJDsState | retrieved_entities | No |
| `rerank_score_node` | BaseGraphState | reranked_results | Yes |
| `persist_matches_node` | BaseGraphState | persisted_match_ids | Yes |
| `notify_candidates_node` | JDToCandidatesState | email_logs | No |
| `notify_candidate_node` | ResumeToJDsState | email_logs | No |

**Error propagation**: Every node checks `state.error` at entry — if set by a previous node, returns `{}` (no-op) instead of executing.

---

## API Endpoints (Phase 2 additions)

```
POST /agent/jd-to-candidates    { entity_id | entity_text, top_k=5, notify=false }
  → {"matches": [...], "email_logs": [...], "graph_steps": [...]}

POST /agent/resume-to-jds       { entity_id | entity_text, top_k=5, notify=false }
  → {"matches": [...], "email_logs": [...], "graph_steps": [...]}

GET  /email-logs                (planned: list email log entries)
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_id` | str? | One of entity_id/entity_text | MongoDB ID of ingested JD or Candidate |
| `entity_text` | str? | One of entity_id/entity_text | Raw text to match against |
| `top_k` | int | No (default 5) | Number of top results to return |
| `notify` | bool | No (default false) | Send email notifications |

### Response Body

| Field | Type | Description |
|-------|------|-------------|
| `matches` | list[dict] | Reranked results with score, rationale, highlights, skills |
| `email_logs` | list[dict] | Email send attempts (dry_run or live) |
| `graph_steps` | list[str] | Names of nodes executed in the graph |

---

## New Data Model: EmailLog

| Field | Type | Description |
|-------|------|-------------|
| `recipient` | str | Email address |
| `subject` | str | Email subject line |
| `body` | str | Email body text |
| `mode` | "dry_run" / "live" | Whether email was actually sent |
| `status` | str | pending / sent / failed / dry_run |
| `sent_at` | datetime | Timestamp |

**Default mode**: `dry_run` — emails are logged to MongoDB but not sent. Set `EMAIL_MODE=live` in `.env` to enable real sending (requires SMTP/API integration, not yet implemented).

---

## Configuration (.env additions)

```bash
# Email (Phase 2)
EMAIL_MODE=dry_run              # "dry_run" (default) or "live"
```

All Phase 1 configuration remains unchanged.

---

## Error Handling

### Graph-Level Error Routing

Every edge in both graphs has a conditional `_check_error` branch:

```python
graph.add_conditional_edges(
    "parse_jd",
    _check_error,
    {"continue": "retrieve_candidates", "end": END},
)
```

If `state.error` is set by any node:
- The graph routes to `END` immediately
- No subsequent nodes execute
- The API returns the error in the response

### Node-Level Error Isolation

Each node checks `state.error` at entry:
```python
async def retrieve_candidates_node(state):
    if state.error:
        return {}  # no-op, propagate error
    # ... actual logic
```

### API-Level Error Handling

The agent router wraps graph execution in try/except:
```python
try:
    final_state = await jd_to_candidates_graph.ainvoke(initial_state)
except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc))
```

---

## Notification System

### Dry-Run Mode (Default)

- `send_notification()` creates an `EmailLog` record with `mode=dry_run`, `status=dry_run`
- No actual email sent
- Logged to MongoDB `email_log` collection
- Visible in API response under `email_logs`

### Live Mode (Placeholder)

- Set `EMAIL_MODE=live` in `.env`
- Currently falls back to dry_run with a warning log
- Ready for SMTP/API integration (SendGrid, SES, etc.)

### Case A vs Case B

| | Case A (JD→Candidates) | Case B (Resume→JDs) |
|---|---|---|
| When | Only if `notify=true` | Always |
| Recipients | All matched candidates | The single candidate who uploaded resume |
| Email content | "You've been matched with a job" | "Here are your matching job openings" |

---

## Testing Checklist

- [ ] `POST /agent/jd-to-candidates` with entity_id → verify matches returned
- [ ] `POST /agent/jd-to-candidates` with entity_text → verify matches returned
- [ ] `POST /agent/resume-to-jds` with entity_id → verify matches returned
- [ ] `POST /agent/resume-to-jds` with entity_text → verify matches returned
- [ ] `POST /agent/jd-to-candidates` with notify=true → verify email_logs populated
- [ ] `POST /agent/resume-to-jds` → verify email_logs always populated (default behavior)
- [ ] `POST /agent/jd-to-candidates` with invalid entity_id → verify error response
- [ ] `POST /agent/resume-to-jds` with invalid entity_id → verify error response
- [ ] Verify matches persisted to MongoDB `matches` collection
- [ ] Verify email logs persisted to MongoDB `email_log` collection
- [ ] Verify graph_steps returned in response matches actual execution path
- [ ] Kill LLM (invalid API key) → verify graph returns error gracefully

---

## What Comes Next (Phase 3+)

1. **Phase 3** — Free-text search mode (`expand_query` + hybrid retrieval graph)
2. **Phase 4** — Gap suggestion agent (resume improvement suggestions)
3. **Phase 5** — Conversational layer (sessions + intent router + refinement/action graphs)
4. **Phase 6** — Gradio UI with chat interface
5. **Phase 7** — MCP tool server
6. **Phase 8** — RAGAS evaluation suite
7. **Phase 9** — Docker/Kubernetes deployment
8. **Phase 10** — Production pipeline + auth
