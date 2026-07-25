"""Pydantic models for LangGraph state objects.

Each graph (JD→Candidates, Resume→JDs) has its own typed state that flows
through every node. Nodes read from and write to this state — no side-effect
hunting, fully testable in isolation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class MatchResult(BaseModel):
    """A single reranked match result flowing through graph state."""

    entity_id: str
    score: float = 0
    rationale: str = ""
    highlights: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rerank_skipped: bool = False


class EmailLogEntry(BaseModel):
    """Record of an email sent (or dry-run logged) during graph execution."""

    recipient: str
    subject: str
    body: str
    mode: Literal["dry_run", "live"] = "dry_run"
    status: str = "pending"


class BaseGraphState(BaseModel):
    """Shared state fields common to both matching graphs.

    LangGraph passes this state through every node. Each node reads
    relevant fields and writes its outputs back to the state.
    """

    # Input
    entity_id: str | None = None
    entity_text: str | None = None
    top_k: int = 5
    notify: bool = False

    # Parsed data (populated by parse node)
    parsed_json: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""

    # Query fields (populated by parse node, used by retrieve node)
    skills_text: str = ""
    experience_texts: list[str] = Field(default_factory=list)

    # Retrieval results
    retrieved_entities: list[dict] = Field(default_factory=list)

    # Reranked results
    reranked_results: list[MatchResult] = Field(default_factory=list)

    # Persistence
    persisted_match_ids: list[str] = Field(default_factory=list)

    # Notification
    email_logs: list[EmailLogEntry] = Field(default_factory=list)

    # Error tracking
    error: str | None = None

    # Graph execution tracking
    completed_steps: list[str] = Field(default_factory=list)


class JDToCandidatesState(BaseGraphState):
    """State for the JD→Candidates graph (Case A).

    Flow: parse_jd → retrieve_candidates → rerank_score → persist_matches → [notify_candidates]
    """

    pass


class ResumeToJDsState(BaseGraphState):
    """State for the Resume→JDs graph (Case B).

    Flow: parse_resume → retrieve_jds → rerank_score → persist_matches → notify_candidate
    """

    pass
