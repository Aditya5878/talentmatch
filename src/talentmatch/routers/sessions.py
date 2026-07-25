"""Session management endpoints.

Provides GET /sessions/{session_id} for viewing session history
and POST /sessions/{session_id}/reset for clearing a session.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from talentmatch.models import Session, SessionMessage, SessionResult
from talentmatch.models.enums import ResultStatus
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.routers.sessions")

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionResponse(BaseModel):
    """Response for session detail endpoint.

    Attributes:
        session_id: The session ID.
        mode: Session mode (recruiter or candidate).
        messages: List of conversation messages.
        active_results: List of active result entries.
        created_at: Session creation timestamp.
        updated_at: Session last update timestamp.
    """

    session_id: str
    mode: str
    messages: list[dict] = []
    active_results: list[dict] = []
    created_at: str = ""
    updated_at: str = ""


class SessionResetResponse(BaseModel):
    """Response for session reset endpoint.

    Attributes:
        session_id: The session ID.
        message: Confirmation message.
    """

    session_id: str
    message: str


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session history and active result set.

    Returns all messages in the conversation and the current active
    results (candidates or JDs) for this session.

    Args:
        session_id: The session ID to look up.

    Returns:
        SessionResponse with messages and active results.

    Raises:
        HTTPException: 404 if session not found.
    """
    session = await Session.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    messages = await SessionMessage.find(
        SessionMessage.session_id == session_id
    ).sort("created_at").to_list()

    active_results = await SessionResult.find(
        SessionResult.session_id == session_id,
        SessionResult.status == ResultStatus.active,
    ).sort("score", reverse=True).to_list()

    return SessionResponse(
        session_id=str(session.id),
        mode=session.mode.value,
        messages=[
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
            }
            for msg in messages
        ],
        active_results=[
            {
                "entity_id": r.entity_id,
                "entity_type": r.entity_type,
                "score": r.score,
                "rationale": r.rationale,
                "highlights": r.highlights,
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
                "gap_suggestions": r.gap_suggestions,
                "status": r.status.value,
            }
            for r in active_results
        ],
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.post("/{session_id}/reset", response_model=SessionResetResponse)
async def reset_session(session_id: str):
    """Reset a session — mark all active results as removed.

    Clears the active result set but preserves conversation history.

    Args:
        session_id: The session ID to reset.

    Returns:
        SessionResetResponse with confirmation.

    Raises:
        HTTPException: 404 if session not found.
    """
    session = await Session.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    active_results = await SessionResult.find(
        SessionResult.session_id == session_id,
        SessionResult.status == ResultStatus.active,
    ).to_list()

    for result in active_results:
        result.status = ResultStatus.removed
        result.updated_at = datetime.utcnow()
        await result.save()

    session.updated_at = datetime.utcnow()
    await session.save()

    logger.info(
        "reset_session: %d results cleared for session %s",
        len(active_results),
        session_id,
        extra={"trace_id": get_trace_id()},
    )

    return SessionResetResponse(
        session_id=session_id,
        message=f"Session reset. {len(active_results)} results cleared.",
    )
