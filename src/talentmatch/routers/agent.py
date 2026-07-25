"""Agent API router — exposes LangGraph-based matching workflows.

Endpoints wrap the JD→Candidates and Resume→JDs graphs, providing
a typed API layer on top of the compiled LangGraph graphs.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from talentmatch.agents.graphs import jd_to_candidates_graph, resume_to_jds_graph
from talentmatch.agents.state import JDToCandidatesState, ResumeToJDsState
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.routers.agent")

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentMatchRequest(BaseModel):
    """Request model for agent-based matching endpoints.

    Provide either entity_id (to match against an ingested document)
    or entity_text (to match against raw text).

    Attributes:
        entity_id: MongoDB document ID of a JD or Candidate.
        entity_text: Raw text to match against (no persistence lookup).
        top_k: Number of top results to return (default 5).
        notify: Whether to send notification emails (default False).
    """

    entity_id: Optional[str] = None
    entity_text: Optional[str] = None
    top_k: int = 5
    notify: bool = False


class AgentMatchResponse(BaseModel):
    """Response model containing agent match results."""

    matches: list[dict]
    email_logs: list[dict] = []
    graph_steps: list[str] = []


@router.post("/jd-to-candidates", response_model=AgentMatchResponse)
async def agent_jd_to_candidates(req: AgentMatchRequest):
    """Find and rank top candidates for a JD using the LangGraph agent pipeline.

    Invokes the JD→Candidates graph: parse_jd → retrieve_candidates →
    rerank_score → persist_matches → [notify_candidates].

    Args:
        req: AgentMatchRequest with jd_id or jd_text.

    Returns:
        AgentMatchResponse with reranked matches, email logs, and graph steps.

    Raises:
        HTTPException: 400 if neither entity_id nor entity_text provided.
        HTTPException: 500 if graph execution fails.
    """
    if not req.entity_id and not req.entity_text:
        raise HTTPException(status_code=400, detail="Provide entity_id or entity_text")

    initial_state = JDToCandidatesState(
        entity_id=req.entity_id,
        entity_text=req.entity_text,
        top_k=req.top_k,
        notify=req.notify,
    )

    try:
        final_state = await jd_to_candidates_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error(
            "Agent jd_to_candidates failed: %s",
            exc,
            extra={"trace_id": get_trace_id()},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    state = JDToCandidatesState(**final_state)

    if state.error:
        raise HTTPException(status_code=500, detail=state.error)

    matches = [m.model_dump() for m in state.reranked_results]
    email_logs = [e.model_dump() for e in state.email_logs]

    return AgentMatchResponse(
        matches=matches,
        email_logs=email_logs,
        graph_steps=[
            "parse_jd",
            "retrieve_candidates",
            "rerank_score",
            "persist_matches",
            "notify_candidates" if req.notify else None,
        ],
    )


@router.post("/resume-to-jds", response_model=AgentMatchResponse)
async def agent_resume_to_jds(req: AgentMatchRequest):
    """Find and rank top JDs for a resume using the LangGraph agent pipeline.

    Invokes the Resume→JDs graph: parse_resume → retrieve_jds →
    rerank_score → persist_matches → notify_candidate.

    Args:
        req: AgentMatchRequest with candidate_id or resume_text.

    Returns:
        AgentMatchResponse with reranked matches, email logs, and graph steps.

    Raises:
        HTTPException: 400 if neither entity_id nor entity_text provided.
        HTTPException: 500 if graph execution fails.
    """
    if not req.entity_id and not req.entity_text:
        raise HTTPException(status_code=400, detail="Provide entity_id or entity_text")

    initial_state = ResumeToJDsState(
        entity_id=req.entity_id,
        entity_text=req.entity_text,
        top_k=req.top_k,
        notify=req.notify,
    )

    try:
        final_state = await resume_to_jds_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error(
            "Agent resume_to_jds failed: %s",
            exc,
            extra={"trace_id": get_trace_id()},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    state = ResumeToJDsState(**final_state)

    if state.error:
        raise HTTPException(status_code=500, detail=state.error)

    matches = [m.model_dump() for m in state.reranked_results]
    email_logs = [e.model_dump() for e in state.email_logs]

    return AgentMatchResponse(
        matches=matches,
        email_logs=email_logs,
        graph_steps=[
            "parse_resume",
            "retrieve_jds",
            "rerank_score",
            "persist_matches",
            "notify_candidate",
        ],
    )
