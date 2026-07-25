# TalentMatch AI — Phase 4 Documentation

## Overview

Phase 4 adds the gap suggestion sub-graph (Spec 7.3) embedded within the Case B (Resume→JDs) workflow. After reranking, the system compares the candidate's skills against each top-matched JD's required skills and uses the LLM to generate concrete, actionable resume improvement suggestions.

**Status**: Complete.

**Depends on**: Phase 2 (LangGraph agent workflows), Phase 3 (match_direction refactoring).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Resume → JDs Graph (Case B, updated)                              │
│                                                                     │
│  parse_resume → retrieve_jds → rerank_score                        │
│       → diff_skills → llm_suggest_edits → format_suggestions       │
│              → persist_matches → notify_candidate                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐           │
│  │  Gap Suggestion Sub-graph (new, Spec 7.3)            │           │
│  │  diff_skills → llm_suggest_edits → format_suggestions │           │
│  └──────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Graph Flow (Spec 7.2 + 7.3)

```
parse_resume ──▶ retrieve_jds ──▶ rerank_score ──▶ diff_skills
                                                        │
                                                   llm_suggest_edits
                                                        │
                                                  format_suggestions
                                                        │
                                                   persist_matches
                                                        │
                                                  notify_candidate
                                                        │
                                                       END
```

---

## New Nodes

### 1. `diff_skills_node` (sub-graph step 1)

Compares the candidate's skills against each top-matched JD's required_skills.

**Input**: `reranked_results` (top 3 matches), `parsed_json` (resume skills)

**Process**:
- Loads each top JD from MongoDB
- Extracts `required_skills` from JD's `parsed_json`
- Computes matched vs missing skills

**Output**: `gap_suggestions` — list of per-JD diff dicts:
```json
[
  {
    "jd_id": "...",
    "jd_title": "Backend Engineer",
    "jd_company": "Acme",
    "required_skills": ["Python", "Django", "PostgreSQL"],
    "matched_skills": ["Python"],
    "missing_skills": ["Django", "PostgreSQL"],
    "match_score": 85.0
  }
]
```

### 2. `llm_suggest_edits_node` (sub-graph step 2)

Calls the LLM with resume text + skill gap to generate 2-4 actionable suggestions per JD.

**Input**: `gap_suggestions` (from diff_skills), `raw_text` (resume), `parsed_json`

**Output**: `gap_suggestions` updated with `suggestions` field:
```json
[
  {
    "jd_id": "...",
    "jd_title": "Backend Engineer",
    "suggestions": [
      "Add a Django project section to your resume showing REST API development",
      "Include PostgreSQL experience from your data engineering work"
    ]
  }
]
```

**Fallback**: If LLM fails, generates generic suggestions based on missing skills.

### 3. `format_suggestions_node` (sub-graph step 3)

Structures the per-JD suggestions into a consistent output format with summary stats.

**Input**: `gap_suggestions` (from llm_suggest_edits)

**Output**: Final formatted `gap_suggestions`:
```json
[
  {
    "jd_id": "...",
    "jd_title": "Backend Engineer",
    "jd_company": "Acme",
    "match_score": 85.0,
    "skills_matched": 1,
    "skills_missing": 2,
    "suggestions": ["..."]
  }
]
```

---

## Data Model Changes

### Match Document (MongoDB)

New field added to `matches` collection:

| Field | Type | Description |
|-------|------|-------------|
| `gap_suggestions` | `list[str]` | Actionable resume improvement suggestions for this specific JD match |

This field is only populated for `direction: resume_to_jd` matches. For other directions, it remains an empty list.

---

## Resume → JDs Graph Comparison

| Phase | Nodes |
|-------|-------|
| Phase 2 | `parse_resume` → `retrieve_jds` → `rerank_score` → `persist_matches` → `notify_candidate` (5 nodes) |
| Phase 4 | `parse_resume` → `retrieve_jds` → `rerank_score` → `diff_skills` → `llm_suggest_edits` → `format_suggestions` → `persist_matches` → `notify_candidate` (8 nodes) |

---

## LLM Prompt Design

The gap suggestion prompt includes:
- **JD context**: title, company, required skills
- **Candidate context**: current skills, full resume text (truncated to 3000 chars)
- **Constraints**: 2-4 suggestions, concrete, actionable, job-specific

Example prompt structure:
```
You are a resume improvement advisor...
Job Title: Backend Engineer
Company: Acme
Required Skills: Python, Django, PostgreSQL
Candidate's Current Skills: Python, Java, React
Candidate Resume: [resume text]

Provide 2-4 specific, actionable suggestions...
```

---

## Testing Checklist

- [ ] `POST /agent/resume-to-jds` with a resume → verify `gap_suggestions` in response
- [ ] Verify each suggestion is concrete (not generic "improve your resume")
- [ ] Verify suggestions reference specific missing skills from each JD
- [ ] Verify `persist_matches_node` stores `gap_suggestions` in Match documents
- [ ] Kill LLM → verify fallback suggestions are generated (based on missing skills)
- [ ] Verify `diff_skills_node` handles JDs without `parsed_json` gracefully
- [ ] Verify `format_suggestions_node` includes `skills_matched` and `skills_missing` counts

---

## What Comes Next (Phase 5+)

1. **Phase 5** — Conversational layer (sessions + intent router + refinement/action graphs)
2. **Phase 6** — Gradio UI with chat interface
3. **Phase 7** — MCP tool server
4. **Phase 8** — RAGAS evaluation suite
