from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from talentmatch.matching.retriever import retrieve_candidates, retrieve_jds
from talentmatch.matching.reranker import rerank
from talentmatch.models import Candidate, JD, Match
from talentmatch.models.enums import MatchDirection

router = APIRouter(prefix="/match", tags=["matching"])


class MatchRequest(BaseModel):
    """Request model for matching endpoints.

    Provide either entity_id (to match against an ingested document)
    or entity_text (to match against raw text).

    Attributes:
        entity_id: MongoDB document ID of a JD or Candidate.
        entity_text: Raw text to match against (no persistence lookup).
        top_k: Number of top results to return (default 5).
        notify: Whether to send notification emails (not yet implemented).
    """

    entity_id: Optional[str] = None
    entity_text: Optional[str] = None
    top_k: int = 5
    notify: bool = False


class MatchResponse(BaseModel):
    """Response model containing reranked match results."""

    matches: list[dict]


@router.post("/jd-to-candidates")
async def jd_to_candidates(req: MatchRequest):
    """Find and rank the top candidates for a given job description.

    Extracts skills and responsibilities from the JD (or uses raw text),
    retrieves matching candidates from Qdrant, reranks with LLM scoring,
    and persists matches to MongoDB.

    Args:
        req: MatchRequest with jd_id or jd_text.

    Returns:
        MatchResponse with top 5 reranked candidate matches.

    Raises:
        HTTPException: 400 if neither entity_id nor entity_text provided.
        HTTPException: 404 if entity_id references a non-existent JD.
    """
    if req.entity_id:
        jd = await JD.get(req.entity_id)
        if not jd:
            raise HTTPException(status_code=404, detail="JD not found")
        query_text = jd.jd_raw_text or jd.title
        parsed = jd.parsed_json
    elif req.entity_text:
        query_text = req.entity_text
        parsed = {}
    else:
        raise HTTPException(status_code=400, detail="Provide entity_id or entity_text")

    skills = ", ".join(parsed.get("required_skills", []))
    responsibilities = parsed.get("responsibilities", [])

    results = await retrieve_candidates(skills_text=skills, experience_texts=responsibilities)

    if not results:
        return MatchResponse(matches=[])

    reranked = await rerank(
        query_text=query_text,
        entities=results,
        direction=MatchDirection.jd_to_candidate,
    )

    if req.entity_id:
        for item in reranked:
            match = Match(
                jd_id=req.entity_id,
                candidate_id=item.get("entity_id"),
                query_text=query_text[:500],
                score=item.get("score", 0),
                rationale=item.get("rationale", ""),
                highlights=item.get("highlights", []),
                matched_skills=item.get("matched_skills", []),
                missing_skills=item.get("missing_skills", []),
                direction=MatchDirection.jd_to_candidate,
            )
            await match.insert()

    return MatchResponse(matches=reranked)


@router.post("/resume-to-jds")
async def resume_to_jds(req: MatchRequest):
    """Find and rank the best-matching job descriptions for a given resume.

    Extracts skills and experience from the resume (or uses raw text),
    retrieves matching JDs from Qdrant, reranks with LLM scoring,
    and persists matches to MongoDB.

    Args:
        req: MatchRequest with candidate_id or resume_text.

    Returns:
        MatchResponse with top 5 reranked JD matches.

    Raises:
        HTTPException: 400 if neither entity_id nor entity_text provided.
        HTTPException: 404 if entity_id references a non-existent candidate.
    """
    if req.entity_id:
        candidate = await Candidate.get(req.entity_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        query_text = candidate.resume_raw_text or ""
        parsed = candidate.parsed_json
    elif req.entity_text:
        query_text = req.entity_text
        parsed = {}
    else:
        raise HTTPException(status_code=400, detail="Provide entity_id or entity_text")

    skills = ", ".join(parsed.get("skills", []))
    experience_texts = []
    for exp in parsed.get("experience", []):
        desc = exp.get("description", "")
        if desc:
            experience_texts.append(desc)

    results = await retrieve_jds(skills_text=skills, experience_texts=experience_texts)

    if not results:
        return MatchResponse(matches=[])

    reranked = await rerank(
        query_text=query_text,
        entities=results,
        direction=MatchDirection.resume_to_jd,
    )

    if req.entity_id:
        for item in reranked:
            match = Match(
                candidate_id=req.entity_id,
                jd_id=item.get("entity_id"),
                query_text=query_text[:500],
                score=item.get("score", 0),
                rationale=item.get("rationale", ""),
                highlights=item.get("highlights", []),
                matched_skills=item.get("matched_skills", []),
                missing_skills=item.get("missing_skills", []),
                direction=MatchDirection.resume_to_jd,
            )
            await match.insert()

    return MatchResponse(matches=reranked)
