"""Chat endpoints for conversational matching (Spec 7.0).

Provides POST /chat/recruiter and POST /chat/candidate endpoints
that handle multi-turn conversations with intent routing.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from talentmatch.agents.graphs import (
    action_graph,
    free_text_search_graph,
    intent_router_graph,
    jd_to_candidates_graph,
    refinement_graph,
    resume_to_jds_graph,
)
from talentmatch.agents.state import (
    ActionState,
    ChatState,
    FreeTextSearchState,
    JDToCandidatesState,
    MatchResult,
    RefinementResult,
    RefinementState,
    ResumeToJDsState,
)
from talentmatch.models import Session, SessionMessage, SessionResult
from talentmatch.models.enums import IntentType, ResultStatus, SessionMode
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.routers.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for chat endpoints.

    Attributes:
        session_id: Optional session ID for follow-up messages.
        message: The user's message (JD text, resume text, query, or follow-up).
        top_k: Number of top results to return (default 5).
        notify: Whether to send email notifications (default False).
    """

    session_id: str | None = None
    message: str
    top_k: int = 5
    notify: bool = False


class ChatResponse(BaseModel):
    """Response from chat endpoints.

    Attributes:
        session_id: The session ID (new or existing).
        intent: The classified intent.
        matches: List of match results (if new search).
        refinement_summary: Summary of refinement action (if refinement).
        email_logs: List of email logs (if action).
        gap_suggestions: List of gap suggestions (if follow-on in Case B).
        graph_steps: List of graph execution steps.
    """

    session_id: str
    intent: str
    matches: list[dict[str, Any]] = []
    refinement_summary: str = ""
    email_logs: list[dict[str, Any]] = []
    gap_suggestions: list[dict[str, Any]] = []
    graph_steps: list[str] = []


async def _get_or_create_session(session_id: str | None, mode: SessionMode) -> Session:
    """Get an existing session or create a new one.

    Args:
        session_id: Optional session ID to look up.
        mode: Session mode (recruiter or candidate).

    Returns:
        Session document (existing or newly created).
    """
    if session_id:
        session = await Session.get(session_id)
        if session:
            return session

    session = Session(mode=mode)
    await session.insert()
    return session


async def _save_user_message(session_id: str, message: str) -> None:
    """Save a user message to the session's conversation history.

    Args:
        session_id: The session ID.
        message: The user's message content.
    """
    msg = SessionMessage(
        session_id=session_id,
        role="user",
        content=message,
    )
    await msg.insert()


async def _save_assistant_message(session_id: str, content: str) -> None:
    """Save an assistant message to the session's conversation history.

    Args:
        session_id: The session ID.
        content: The assistant's response content.
    """
    msg = SessionMessage(
        session_id=session_id,
        role="assistant",
        content=content,
    )
    await msg.insert()


async def _store_session_results(session_id: str, matches: list[dict], mode: str) -> None:
    """Store match results as session_results for follow-up refinement.

    Args:
        session_id: The session ID.
        matches: List of match result dicts.
        mode: Session mode (recruiter or candidate).
    """
    entity_type = "candidate" if mode == "recruiter" else "jd"

    # Clear previous active results for this session
    existing = await SessionResult.find(
        SessionResult.session_id == session_id,
        SessionResult.status == ResultStatus.active,
    ).to_list()
    for doc in existing:
        doc.status = ResultStatus.removed
        await doc.save()

    for match in matches:
        result = SessionResult(
            session_id=session_id,
            entity_type=entity_type,
            entity_id=match.get("entity_id", ""),
            score=match.get("score", 0),
            rationale=match.get("rationale", ""),
            highlights=match.get("highlights", []),
            matched_skills=match.get("matched_skills", []),
            missing_skills=match.get("missing_skills", []),
            gap_suggestions=match.get("gap_suggestions", []),
            status=ResultStatus.active,
        )
        await result.insert()


async def _get_active_results(session_id: str) -> list[RefinementResult]:
    """Get active session results for refinement.

    Args:
        session_id: The session ID.

    Returns:
        List of RefinementResult objects.
    """
    results = await SessionResult.find(
        SessionResult.session_id == session_id,
        SessionResult.status == ResultStatus.active,
    ).to_list()

    return [
        RefinementResult(
            result_id=str(r.id),
            entity_id=r.entity_id,
            entity_type=r.entity_type,
            score=r.score,
            rationale=r.rationale,
            highlights=r.highlights,
            matched_skills=r.matched_skills,
            missing_skills=r.missing_skills,
            status=r.status.value,
        )
        for r in results
    ]


async def _get_action_results(session_id: str, mode: str) -> list:
    """Get active session results with email recipients for action.

    Args:
        session_id: The session ID.
        mode: Session mode (recruiter or candidate).

    Returns:
        List of ActionResult-compatible dicts.
    """
    from talentmatch.agents.state import ActionResult

    results = await SessionResult.find(
        SessionResult.session_id == session_id,
        SessionResult.status == ResultStatus.active,
    ).to_list()

    action_results = []
    for r in results:
        recipient = ""
        if mode == "candidate":
            # Candidate mode: email is the candidate about matched JDs
            from talentmatch.models import Candidate
            candidate = await Candidate.get(r.entity_id) if r.entity_type == "candidate" else None
            if candidate and candidate.email:
                recipient = candidate.email
        else:
            # Recruiter mode: email candidates about the JD
            from talentmatch.models import Candidate
            candidate = await Candidate.get(r.entity_id) if r.entity_type == "candidate" else None
            if candidate and candidate.email:
                recipient = candidate.email

        action_results.append(ActionResult(
            result_id=str(r.id),
            entity_id=r.entity_id,
            entity_type=r.entity_type,
            recipient=recipient,
            score=r.score,
        ))

    return action_results


@router.post("/recruiter", response_model=ChatResponse)
async def chat_recruiter(req: ChatRequest):
    """Conversational endpoint for recruiter flow (Case A).

    Handles new searches (JD→candidates, free-text), refinement
    ("remove candidate 3"), and actions ("email these candidates").

    Args:
        req: ChatRequest with optional session_id and message.

    Returns:
        ChatResponse with matches, refinement summary, or email logs.

    Raises:
        HTTPException: 400 if message is empty.
        HTTPException: 500 if graph execution fails.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session = await _get_or_create_session(req.session_id, SessionMode.recruiter)
    await _save_user_message(str(session.id), req.message)

    # Step 1: Classify intent
    router_state = ChatState(
        session_id=str(session.id),
        message=req.message,
        mode="recruiter",
        top_k=req.top_k,
        notify=req.notify,
    )

    try:
        router_result = await intent_router_graph.ainvoke(router_state)
    except Exception as exc:
        logger.error(
            "Intent classification failed: %s",
            exc,
            extra={"trace_id": get_trace_id()},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    classified = ChatState(**router_result)
    intent = classified.intent

    logger.info(
        "chat_recruiter: intent=%s, session=%s",
        intent.value,
        str(session.id),
        extra={"trace_id": get_trace_id()},
    )

    # Step 2: Dispatch based on intent
    if intent == IntentType.new_search:
        return await _handle_new_search_recruiter(session, req, classified)
    elif intent == IntentType.refinement:
        return await _handle_refinement(session, req, classified)
    elif intent == IntentType.action:
        return await _handle_action(session, req, classified, "recruiter")
    else:
        return await _handle_follow_on(session, req, classified)


@router.post("/candidate", response_model=ChatResponse)
async def chat_candidate(req: ChatRequest):
    """Conversational endpoint for candidate flow (Case B).

    Handles new searches (resume→JDs, free-text), refinement
    ("remove this job"), and actions ("email me these openings").

    Args:
        req: ChatRequest with optional session_id and message.

    Returns:
        ChatResponse with matches, gap suggestions, refinement summary, or email logs.

    Raises:
        HTTPException: 400 if message is empty.
        HTTPException: 500 if graph execution fails.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session = await _get_or_create_session(req.session_id, SessionMode.candidate)
    await _save_user_message(str(session.id), req.message)

    # Step 1: Classify intent
    router_state = ChatState(
        session_id=str(session.id),
        message=req.message,
        mode="candidate",
        top_k=req.top_k,
        notify=req.notify,
    )

    try:
        router_result = await intent_router_graph.ainvoke(router_state)
    except Exception as exc:
        logger.error(
            "Intent classification failed: %s",
            exc,
            extra={"trace_id": get_trace_id()},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    classified = ChatState(**router_result)
    intent = classified.intent

    logger.info(
        "chat_candidate: intent=%s, session=%s",
        intent.value,
        str(session.id),
        extra={"trace_id": get_trace_id()},
    )

    # Step 2: Dispatch based on intent
    if intent == IntentType.new_search:
        return await _handle_new_search_candidate(session, req, classified)
    elif intent == IntentType.refinement:
        return await _handle_refinement(session, req, classified)
    elif intent == IntentType.action:
        return await _handle_action(session, req, classified, "candidate")
    else:
        return await _handle_follow_on(session, req, classified)


async def _handle_new_search_recruiter(
    session: Session, req: ChatRequest, classified: ChatState
) -> ChatResponse:
    """Handle a new search intent in recruiter mode (Case A).

    Determines if the message is a JD (jd-to-candidates) or free-text query,
    invokes the appropriate graph, and stores results for follow-up.
    """
    entity_text = classified.entity_text or req.message

    # Heuristic: if the text looks like a JD (long, multi-section), use JD graph
    if len(entity_text) > 200 and any(
        kw in entity_text.lower()
        for kw in ["requirements", "responsibilities", "qualifications", "experience", "skills"]
    ):
        state = JDToCandidatesState(
            entity_text=entity_text,
            top_k=req.top_k,
            notify=req.notify,
        )
        try:
            result = await jd_to_candidates_graph.ainvoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        final = JDToCandidatesState(**result)
        matches = [m.model_dump() for m in final.reranked_results]
    else:
        state = FreeTextSearchState(
            raw_text=entity_text,
            top_k=req.top_k,
            match_direction=MatchDirection.keyword_to_candidate,
            search_direction="candidate",
        )
        try:
            result = await free_text_search_graph.ainvoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        final = FreeTextSearchState(**result)
        matches = [m.model_dump() for m in final.reranked_results]

    if final.error:
        raise HTTPException(status_code=500, detail=final.error)

    await _store_session_results(str(session.id), matches, "recruiter")

    summary = f"Found {len(matches)} candidates matching your search."
    await _save_assistant_message(str(session.id), summary)

    return ChatResponse(
        session_id=str(session.id),
        intent=classified.intent.value,
        matches=matches,
        graph_steps=final.completed_steps,
    )


async def _handle_new_search_candidate(
    session: Session, req: ChatRequest, classified: ChatState
) -> ChatResponse:
    """Handle a new search intent in candidate mode (Case B).

    Determines if the message is a resume (resume-to-JDs) or free-text query,
    invokes the appropriate graph, and stores results for follow-up.
    """
    from talentmatch.models.enums import MatchDirection

    entity_text = classified.entity_text or req.message

    # Heuristic: if the text looks like a resume (long, has experience/education), use resume graph
    if len(entity_text) > 200 and any(
        kw in entity_text.lower()
        for kw in ["experience", "education", "skills", "work", "employment"]
    ):
        state = ResumeToJDsState(
            entity_text=entity_text,
            top_k=req.top_k,
        )
        try:
            result = await resume_to_jds_graph.ainvoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        final = ResumeToJDsState(**result)
        matches = [m.model_dump() for m in final.reranked_results]
    else:
        state = FreeTextSearchState(
            raw_text=entity_text,
            top_k=req.top_k,
            match_direction=MatchDirection.keyword_to_jd,
            search_direction="jd",
        )
        try:
            result = await free_text_search_graph.ainvoke(state)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        final = FreeTextSearchState(**result)
        matches = [m.model_dump() for m in final.reranked_results]

    if final.error:
        raise HTTPException(status_code=500, detail=final.error)

    await _store_session_results(str(session.id), matches, "candidate")

    summary = f"Found {len(matches)} matching job openings."
    await _save_assistant_message(str(session.id), summary)

    return ChatResponse(
        session_id=str(session.id),
        intent=classified.intent.value,
        matches=matches,
        gap_suggestions=classified.gap_suggestions,
        graph_steps=final.completed_steps,
    )


async def _handle_refinement(
    session: Session, req: ChatRequest, classified: ChatState
) -> ChatResponse:
    """Handle a refinement intent — filter/remove from active result set."""
    active_results = await _get_active_results(str(session.id))

    if not active_results:
        return ChatResponse(
            session_id=str(session.id),
            intent=classified.intent.value,
            refinement_summary="No active results to refine.",
        )

    refinement_state = RefinementState(
        session_id=str(session.id),
        message=req.message,
        session_results=active_results,
    )

    try:
        result = await refinement_graph.ainvoke(refinement_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    final = RefinementState(**result)

    if final.error:
        raise HTTPException(status_code=500, detail=final.error)

    active_count = sum(1 for r in final.session_results if r.status == "active")
    summary = f"Refined results: {active_count} candidates remaining."
    await _save_assistant_message(str(session.id), summary)

    return ChatResponse(
        session_id=str(session.id),
        intent=classified.intent.value,
        refinement_summary=summary,
        graph_steps=final.completed_steps,
    )


async def _handle_action(
    session: Session, req: ChatRequest, classified: ChatState, mode: str
) -> ChatResponse:
    """Handle an action intent — send emails to matched results."""
    action_results = await _get_action_results(str(session.id), mode)

    if not action_results:
        return ChatResponse(
            session_id=str(session.id),
            intent=classified.intent.value,
            refinement_summary="No active results to act on.",
        )

    action_state = ActionState(
        session_id=str(session.id),
        message=req.message,
        mode=mode,
        session_results=action_results,
    )

    try:
        result = await action_graph.ainvoke(action_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    final = ActionState(**result)

    if final.error:
        raise HTTPException(status_code=500, detail=final.error)

    email_logs = [e.model_dump() for e in final.email_logs]
    summary = f"Sent {len(email_logs)} email(s)."
    await _save_assistant_message(str(session.id), summary)

    return ChatResponse(
        session_id=str(session.id),
        intent=classified.intent.value,
        email_logs=email_logs,
        graph_steps=final.completed_steps,
    )


async def _handle_follow_on(
    session: Session, req: ChatRequest, classified: ChatState
) -> ChatResponse:
    """Handle a follow-on question about existing results or gap suggestions."""
    active_results = await SessionResult.find(
        SessionResult.session_id == str(session.id),
        SessionResult.status == ResultStatus.active,
    ).to_list()

    gap_suggestions = []
    for r in active_results:
        if r.gap_suggestions:
            gap_suggestions.append({
                "entity_id": r.entity_id,
                "suggestions": r.gap_suggestions,
            })

    summary = (
        f"You have {len(active_results)} active results. "
        f"Gap suggestions are available for {len(gap_suggestions)} matches."
    )
    await _save_assistant_message(str(session.id), summary)

    return ChatResponse(
        session_id=str(session.id),
        intent=classified.intent.value,
        gap_suggestions=gap_suggestions,
        refinement_summary=summary,
    )
