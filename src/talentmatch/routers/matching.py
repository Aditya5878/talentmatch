from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from talentmatch.matching.retriever import retrieve_candidates, retrieve_jds
from talentmatch.matching.reranker import rerank
from talentmatch.models import Candidate, JD, Match
from talentmatch.models.enums import MatchDirection

router = APIRouter(prefix="/match", tags=["matching"])


class MatchRequest(BaseModel):
    entity_id: Optional[str] = None
    entity_text: Optional[str] = None
    top_k: int = 5
    notify: bool = False


class MatchResponse(BaseModel):
    matches: list[dict]


@router.post("/jd-to-candidates")
async def jd_to_candidates(req: MatchRequest):
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
