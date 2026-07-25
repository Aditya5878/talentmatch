from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field

from talentmatch.models.enums import ResultStatus, SessionMode


class Session(Document):
    """A conversational session for recruiter or candidate flow.

    Each session has a mode (recruiter or candidate), an active result set,
    and persisted conversation history. Sessions are created on first message
    and reused for follow-up turns.
    """

    mode: SessionMode
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "sessions"


class SessionMessage(Document):
    """A single message in a session's conversation history."""

    session_id: str
    role: str = "user"
    content: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "session_messages"


class SessionResult(Document):
    """A single result entry in a session's active result set.

    Represents one candidate or JD match that can be filtered, removed,
    or acted upon (emailed) during the conversation.
    """

    session_id: str
    entity_type: str = "candidate"
    entity_id: str = ""
    score: float = 0.0
    rationale: str = ""
    highlights: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    gap_suggestions: list[str] = Field(default_factory=list)
    status: ResultStatus = ResultStatus.active
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "session_results"
