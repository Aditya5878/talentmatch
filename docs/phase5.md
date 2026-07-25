# TalentMatch AI — Phase 5 Documentation

## Overview

Phase 5 adds the conversational layer (Spec 1b, 7.0, 7.5, 7.6) — multi-turn sessions with intent routing, refinement, and action graphs. Users can now have conversations through `/chat/recruiter` and `/chat/candidate` endpoints.

**Status**: Complete.

**Depends on**: Phase 2 (matching graphs), Phase 3 (free-text search), Phase 4 (gap suggestions).

---

## Architecture

```
POST /chat/recruiter or /chat/candidate
  │
  ▼
Intent Router Graph (7.0)
  classify_intent → END (returns intent)
       │
       ├── new_search ────▶ matching graphs (7.1/7.2/7.4)
       │                     → store session_results
       │
       ├── refinement ────▶ Refinement Graph (7.5)
       │                     resolve_reference → apply_refinement → persist_session_results
       │
       ├── action ────────▶ Action Graph (7.6)
       │                     resolve_scope → send_email → log_email_results
       │
       └── follow_on ─────▶ return gap suggestions from session context
```

---

## New MongoDB Models

### sessions
| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Session ID |
| `mode` | `recruiter` / `candidate` | Which flow |
| `created_at` | datetime | Session creation |
| `updated_at` | datetime | Last update |

### session_messages
| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Message ID |
| `session_id` | str | Parent session |
| `role` | `user` / `assistant` | Message sender |
| `content` | str | Message text |
| `created_at` | datetime | Message timestamp |

### session_results
| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Result ID |
| `session_id` | str | Parent session |
| `entity_type` | `candidate` / `jd` | Entity type |
| `entity_id` | str | Reference to entity |
| `score` | float | Match score |
| `rationale` | str | Match rationale |
| `highlights` | list[str] | Notable facts |
| `matched_skills` | list[str] | Skills that matched |
| `missing_skills` | list[str] | Skills that are missing |
| `gap_suggestions` | list[str] | Resume improvement suggestions |
| `status` | `active` / `removed` | Whether result is still active |

---

## Intent Router (Spec 7.0)

The intent router classifies each user message into one of four categories:

| Intent | Trigger Examples | Dispatch |
|--------|-----------------|----------|
| `new_search` | "Find Java developers", JD text, resume text | Matching graphs (7.1/7.2/7.4) |
| `refinement` | "Remove candidate 3", "Only keep senior devs" | Refinement graph (7.5) |
| `action` | "Email these candidates", "Send me these" | Action graph (7.6) |
| `follow_on` | "Why did you suggest this?", "Tell me more" | Return gap suggestions |

---

## Refinement Graph (Spec 7.5)

Operates on the session's active result set — no retrieval or reranking needed.

```
resolve_reference → apply_refinement → persist_session_results
```

### resolve_reference
Maps user references ("candidate 3", "the senior ones") to specific `session_results` rows. Exact index matches resolve directly; fuzzy filters use LLM.

### apply_refinement
Flips `status` to `removed` for targeted results (or keeps only targeted results if action is "keep").

### persist_session_results
Updates the `session_results` documents in MongoDB.

---

## Action Graph (Spec 7.6)

Operates on the session's active result set — sends emails.

```
resolve_scope → send_email → log_email_results
```

### resolve_scope
Determines which active results to email (all or a subset).

### send_email
Sends an email to each result's recipient via `notification.py` (respects `EMAIL_MODE`).

### log_email_results
Records the email action in the session's conversation history.

---

## New API Endpoints

```
POST /chat/recruiter        { session_id?, message, top_k=5, notify=false }
  → { session_id, intent, matches, refinement_summary, email_logs, gap_suggestions, graph_steps }

POST /chat/candidate        { session_id?, message, top_k=5, notify=false }
  → { session_id, intent, matches, refinement_summary, email_logs, gap_suggestions, graph_steps }

GET  /sessions/{session_id}
  → { session_id, mode, messages, active_results, created_at, updated_at }

POST /sessions/{session_id}/reset
  → { session_id, message }
```

---

## Conversation Flow Example

**Turn 1 (new search):**
```
User: "Find me Python developers with 5+ years experience"
→ Intent: new_search
→ Invoke free_text_search_graph
→ Returns 5 candidates
→ Store as session_results (all active)
→ Response: "Found 5 candidates matching your search."
```

**Turn 2 (refinement):**
```
User: "Remove candidate 3"
→ Intent: refinement
→ Invoke refinement_graph
→ Mark result #3 as removed
→ Response: "Refined results: 4 candidates remaining."
```

**Turn 3 (action):**
```
User: "Email the remaining candidates"
→ Intent: action
→ Invoke action_graph
→ Send emails to 4 recipients
→ Response: "Sent 4 email(s)."
```

---

## Testing Checklist

- [ ] `POST /chat/recruiter` with JD text → verify 5 candidates returned
- [ ] `POST /chat/candidate` with resume text → verify matching JDs returned
- [ ] Verify `session_id` is returned and reusable for follow-up
- [ ] `POST /chat/recruiter` with "remove candidate 1" → verify refinement works
- [ ] `POST /chat/recruiter` with "email these candidates" → verify email action
- [ ] `GET /sessions/{id}` → verify messages and active_results populated
- [ ] `POST /sessions/{id}/reset` → verify active results cleared
- [ ] Verify conversation history preserved across turns
- [ ] Verify intent classification works for various message types

---

## What Comes Next (Phase 6+)

1. **Phase 6** — Gradio UI with chat interface
2. **Phase 7** — MCP tool server
3. **Phase 8** — RAGAS evaluation suite
