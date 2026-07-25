# TalentMatch AI — Phase 3 Documentation

## Overview

Phase 3 adds the free-text search mode (Spec 7.4) as a LangGraph graph. Users can now search for candidates or job openings using keyword/skill queries without uploading any document.

**Status**: Complete.

**Depends on**: Phase 2 (LangGraph agent workflows, reusable rerank/persist nodes).

---

## What Changed from Phase 2

| Area | Phase 2 | Phase 3 |
|------|---------|---------|
| Query modes | JD→Candidates, Resume→JDs | + Free-text search (no document) |
| State model | `BaseGraphState` | + `match_direction`, `expanded_query_terms`, `search_direction` fields |
| Rerank direction | `isinstance` check (2 options) | `state.match_direction` field (4 options) |
| New graph | — | `free_text_search_graph` |
| New nodes | — | `expand_query_node`, `hybrid_retrieve_node` |
| New endpoints | — | `POST /agent/search/candidates`, `POST /agent/search/jds` |

---

## Architecture

```
┌──────────────────┐
│  /agent/search/* │
│  Free-text query  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  Free-Text Search Graph (Spec 7.4)       │
│                                          │
│  expand_query → hybrid_retrieve          │
│       → rerank_score → persist_matches   │
│              (shared with Phase 2)       │
└──────────────────────────────────────────┘
```

---

## Graph Flow (Spec 7.4)

```
expand_query ──(no error)──▶ hybrid_retrieve ──(no error)──▶ rerank_score
    │                            │                              │
    │ (error)                    │ (error)                      │ (error)
    ▼                            ▼                              ▼
   END                         END                            END
                                                              │ (no error)
                                                              ▼
                                                        persist_matches
                                                              │
                                                              ▼
                                                             END
```

### Nodes

| Node | Input | Output | Source |
|------|-------|--------|--------|
| `expand_query` | `raw_text` (user query) | `expanded_query_terms`, combined `raw_text` | New |
| `hybrid_retrieve` | `raw_text` (combined), `search_direction` | `retrieved_entities` | New |
| `rerank_score` | `raw_text`, `retrieved_entities`, `match_direction` | `reranked_results` | Shared (Phase 2) |
| `persist_matches` | `reranked_results`, `match_direction` | `persisted_match_ids` | Shared (Phase 2) |

---

## State Model Changes

### New Fields on BaseGraphState

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `match_direction` | `MatchDirection` | `jd_to_candidate` | Which direction — `jd_to_candidate`, `resume_to_jd`, `keyword_to_candidate`, `keyword_to_jd` |
| `expanded_query_terms` | `list[str]` | `[]` | LLM-expanded skill terms from `expand_query_node` |
| `search_direction` | `"candidate" \| "jd"` | `"candidate"` | Which Qdrant collection to search |

### FreeTextSearchState

```python
class FreeTextSearchState(BaseGraphState):
    """Free-text search graph state — no document upload required."""
    pass
```

---

## Refactor: isinstance → state.match_direction

Phase 2 used `isinstance(state, JDToCandidatesState)` to determine match direction in `rerank_score_node` and `persist_matches_node`. Phase 3 replaces this with `state.match_direction` (a `MatchDirection` enum field).

**Before (Phase 2):**
```python
direction = (
    MatchDirection.jd_to_candidate
    if isinstance(state, JDToCandidatesState)
    else MatchDirection.resume_to_jd
)
```

**After (Phase 3):**
```python
direction = state.match_direction
```

Each parse node now sets `match_direction` in its return dict:
- `parse_jd_node` → `MatchDirection.jd_to_candidate`
- `parse_resume_node` → `MatchDirection.resume_to_jd`
- `expand_query_node` (via API) → `MatchDirection.keyword_to_candidate` or `keyword_to_jd`

---

## API Endpoints (Phase 3 additions)

```
POST /agent/search/candidates    { query_text, top_k=5 }
  → {"matches": [...], "email_logs": [], "graph_steps": [...]}

POST /agent/search/jds           { query_text, top_k=5 }
  → {"matches": [...], "email_logs": [], "graph_steps": [...]}
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query_text` | str | Yes | Search query (e.g. "Java developers with Spring Boot") |
| `top_k` | int | No (default 5) | Number of top results |

### Response

Same `AgentMatchResponse` as other agent endpoints. `email_logs` is always empty (no notification for free-text search).

---

## Query Expansion

The `expand_query_node` calls the LLM with a prompt to expand a raw skill query:

```
Input:  "Java"
Output: ["Java", "Spring Boot", "Hibernate", "Microservices", "Maven"]
```

The original query + expanded terms are combined into a single search string:
```
"Java Java Spring Boot Hibernate Microservices Maven"
```

This combined string is embedded and used for vector search against Qdrant.

**Fallback**: If the LLM expansion fails, the raw query is used as-is (graceful degradation).

---

## Testing Checklist

- [ ] `POST /agent/search/candidates` with "Java" → verify candidates returned
- [ ] `POST /agent/search/jds` with "Python backend" → verify JDs returned
- [ ] Verify `graph_steps` includes `["expand_query", "hybrid_retrieve", "rerank_score", "persist_matches"]`
- [ ] Verify `expanded_query_terms` is populated in graph state
- [ ] Verify matches persisted to MongoDB `matches` collection with `keyword_to_candidate` / `keyword_to_jd` direction
- [ ] Verify empty query returns 400 error
- [ ] Kill LLM → verify fallback uses raw query and graph completes

---

## What Comes Next (Phase 4+)

1. **Phase 4** — Gap suggestion agent (resume improvement suggestions)
2. **Phase 5** — Conversational layer (sessions + intent router + refinement/action graphs)
   - `/chat/recruiter` and `/chat/candidate` will call these graphs internally
   - Old `/search/*` endpoints deprecated, kept for CLI/testing
3. **Phase 6** — Gradio UI with chat interface
4. **Phase 7** — MCP tool server
5. **Phase 8** — RAGAS evaluation suite
