"""Pydantic models for LangGraph state objects.

Each graph (JD→Candidates, Resume→JDs) has its own typed state that flows
through every node. Nodes read from and write to this state — no side-effect
hunting, fully testable in isolation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from talentmatch.models.enums import IntentType, MatchDirection


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

    # Match direction (set by parse nodes, read by rerank/persist nodes)
    match_direction: MatchDirection = MatchDirection.jd_to_candidate

    # Free-text search fields (populated by expand_query node)
    expanded_query_terms: list[str] = Field(default_factory=list)

    # Gap suggestion output (populated by gap_suggestion sub-graph, Case B only)
    gap_suggestions: list[dict] = Field(default_factory=list)

    # Which collection to search: "candidate" or "jd"
    search_direction: Literal["candidate", "jd"] = "candidate"


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


class FreeTextSearchState(BaseGraphState):
    """State for the free-text search graph (no document upload).

    Flow: expand_query → hybrid_retrieve → rerank_score → persist_matches

    Used for keyword/skill searches like "Java developers" or
    "Python backend openings" — no resume or JD needed as input.
    """

    pass


class RefinementResult(BaseModel):
    """A single result entry from the session's active result set."""

    result_id: str
    entity_id: str
    entity_type: str = "candidate"
    score: float = 0
    rationale: str = ""
    highlights: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    status: str = "active"


class RefinementState(BaseModel):
    """State for the refinement graph (Spec 7.5).

    Flow: resolve_reference → apply_refinement → persist_session_results

    Operates on the session's active result set — no retrieval call needed.
    """

    session_id: str = ""
    message: str = ""
    intent: IntentType = IntentType.refinement
    session_results: list[RefinementResult] = Field(default_factory=list)
    resolved_targets: list[str] = Field(default_factory=list)
    refinement_action: str = ""
    error: str | None = None
    completed_steps: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    """A single result entry targeted for action (email)."""

    result_id: str
    entity_id: str
    entity_type: str = "candidate"
    recipient: str = ""
    score: float = 0


class ActionState(BaseModel):
    """State for the action graph (Spec 7.6).

    Flow: resolve_scope → send_email → log_email_results

    Operates on the session's active result set — sends emails to
    matched candidates (or to the candidate about matched JDs).
    """

    session_id: str = ""
    message: str = ""
    intent: IntentType = IntentType.action
    mode: str = "recruiter"
    session_results: list[ActionResult] = Field(default_factory=list)
    email_logs: list[EmailLogEntry] = Field(default_factory=list)
    error: str | None = None
    completed_steps: list[str] = Field(default_factory=list)


class ChatState(BaseModel):
    """State for the intent router graph (Spec 7.0).

    The intent router classifies the user's message and dispatches to
    the appropriate sub-graph (matching, refinement, action, or follow-on).
    """

    session_id: str = ""
    message: str = ""
    mode: str = "recruiter"
    intent: IntentType = IntentType.new_search
    entity_text: str | None = None
    top_k: int = 5
    notify: bool = False
    match_results: list[MatchResult] = Field(default_factory=list)
    refinement_results: list[RefinementResult] = Field(default_factory=list)
    email_logs: list[EmailLogEntry] = Field(default_factory=list)
    gap_suggestions: list[dict] = Field(default_factory=list)
    error: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
