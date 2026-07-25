"""Reusable LangGraph node functions.

Each node is a pure function: takes state, returns a dict of state updates.
No side effects in the function body — all persistence happens through
the state object, which LangGraph handles via its checkpointer.
"""

import json
import logging

from bson import ObjectId

from talentmatch.agents.state import (
    BaseGraphState,
    EmailLogEntry,
    JDToCandidatesState,
    MatchResult,
    ResumeToJDsState,
)
from talentmatch.ingestion.parser import parse_jd, parse_resume
from talentmatch.matching.reranker import rerank
from talentmatch.matching.retriever import retrieve_candidates, retrieve_jds
from talentmatch.models import Candidate, JD, Match
from talentmatch.models.enums import MatchDirection
from talentmatch.notification import send_notification
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.agents.nodes")


async def parse_jd_node(state: JDToCandidatesState) -> dict:
    """Parse a JD document and extract query fields for retrieval.

    Loads the JD from MongoDB by entity_id (or uses raw text), calls the LLM
    parser, and extracts skills_text + experience_texts for the retrieve node.

    Args:
        state: Current graph state with entity_id or entity_text.

    Returns:
        State update dict with parsed_json, raw_text, skills_text, experience_texts.
    """
    if state.entity_id:
        jd = await JD.get(state.entity_id)
        if not jd:
            return {"error": f"JD not found: {state.entity_id}"}
        raw_text = jd.jd_raw_text or jd.title
        parsed = jd.parsed_json
    elif state.entity_text:
        raw_text = state.entity_text
        parsed = await parse_jd(raw_text)
    else:
        return {"error": "Provide entity_id or entity_text"}

    skills = ", ".join(parsed.get("required_skills", []))
    responsibilities = parsed.get("responsibilities", [])

    logger.info(
        "parse_jd_node: skills=%d, responsibilities=%d",
        len(parsed.get("required_skills", [])),
        len(responsibilities),
        extra={"trace_id": get_trace_id()},
    )

    return {
        "parsed_json": parsed,
        "raw_text": raw_text,
        "skills_text": skills,
        "experience_texts": responsibilities,
    }


async def parse_resume_node(state: ResumeToJDsState) -> dict:
    """Parse a resume and extract query fields for retrieval.

    Loads the Candidate from MongoDB by entity_id (or uses raw text), calls
    the LLM parser, and extracts skills_text + experience_texts.

    Args:
        state: Current graph state with entity_id or entity_text.

    Returns:
        State update dict with parsed_json, raw_text, skills_text, experience_texts.
    """
    if state.entity_id:
        candidate = await Candidate.get(state.entity_id)
        if not candidate:
            return {"error": f"Candidate not found: {state.entity_id}"}
        raw_text = candidate.resume_raw_text or ""
        parsed = candidate.parsed_json
    elif state.entity_text:
        raw_text = state.entity_text
        parsed = await parse_resume(raw_text)
    else:
        return {"error": "Provide entity_id or entity_text"}

    skills = ", ".join(parsed.get("skills", []))
    experience_texts = [
        exp.get("description", "")
        for exp in parsed.get("experience", [])
        if exp.get("description")
    ]

    logger.info(
        "parse_resume_node: skills=%d, experience=%d",
        len(parsed.get("skills", [])),
        len(experience_texts),
        extra={"trace_id": get_trace_id()},
    )

    return {
        "parsed_json": parsed,
        "raw_text": raw_text,
        "skills_text": skills,
        "experience_texts": experience_texts,
    }


async def retrieve_candidates_node(state: JDToCandidatesState) -> dict:
    """Retrieve top candidate chunks from Qdrant matching the JD's requirements.

    Uses the skills_text and experience_texts from state to run multi-query
    retrieval against the candidate collection.

    Args:
        state: Current graph state with skills_text and experience_texts.

    Returns:
        State update dict with retrieved_entities.
    """
    if state.error:
        return {}

    results = await retrieve_candidates(
        skills_text=state.skills_text,
        experience_texts=state.experience_texts,
    )

    logger.info(
        "retrieve_candidates_node: %d entities retrieved",
        len(results),
        extra={"trace_id": get_trace_id()},
    )

    return {"retrieved_entities": results}


async def retrieve_jds_node(state: ResumeToJDsState) -> dict:
    """Retrieve top JD chunks from Qdrant matching the resume's skills.

    Uses the skills_text and experience_texts from state to run multi-query
    retrieval against the JD collection.

    Args:
        state: Current graph state with skills_text and experience_texts.

    Returns:
        State update dict with retrieved_entities.
    """
    if state.error:
        return {}

    results = await retrieve_jds(
        skills_text=state.skills_text,
        experience_texts=state.experience_texts,
    )

    logger.info(
        "retrieve_jds_node: %d entities retrieved",
        len(results),
        extra={"trace_id": get_trace_id()},
    )

    return {"retrieved_entities": results}


async def rerank_score_node(state: BaseGraphState) -> dict:
    """Rerank retrieved entities using LLM scoring.

    Sends the raw text + retrieved entities to the LLM reranker, which returns
    structured scores, rationale, highlights, and matched/missing skills.

    Args:
        state: Current graph state with raw_text and retrieved_entities.

    Returns:
        State update dict with reranked_results.
    """
    if state.error or not state.retrieved_entities:
        return {}

    direction = (
        MatchDirection.jd_to_candidate
        if isinstance(state, JDToCandidatesState)
        else MatchDirection.resume_to_jd
    )

    raw_results = await rerank(
        query_text=state.raw_text,
        entities=state.retrieved_entities,
        direction=direction,
    )

    match_results = [MatchResult(**r) for r in raw_results]

    logger.info(
        "rerank_score_node: %d results reranked",
        len(match_results),
        extra={"trace_id": get_trace_id()},
    )

    return {"reranked_results": match_results}


async def persist_matches_node(state: BaseGraphState) -> dict:
    """Persist reranked results to MongoDB matches collection.

    Creates Match documents for each reranked result, linking the JD and
    candidate (or vice versa) with scores and rationale.

    Args:
        state: Current graph state with reranked_results and entity info.

    Returns:
        State update dict with persisted_match_ids.
    """
    if state.error or not state.reranked_results:
        return {}

    direction = (
        MatchDirection.jd_to_candidate
        if isinstance(state, JDToCandidatesState)
        else MatchDirection.resume_to_jd
    )

    match_ids = []
    for item in state.reranked_results:
        if isinstance(state, JDToCandidatesState):
            match = Match(
                jd_id=state.entity_id,
                candidate_id=item.entity_id,
                query_text=state.raw_text[:500],
                score=item.score,
                rationale=item.rationale,
                highlights=item.highlights,
                matched_skills=item.matched_skills,
                missing_skills=item.missing_skills,
                direction=direction,
            )
        else:
            match = Match(
                candidate_id=state.entity_id,
                jd_id=item.entity_id,
                query_text=state.raw_text[:500],
                score=item.score,
                rationale=item.rationale,
                highlights=item.highlights,
                matched_skills=item.matched_skills,
                missing_skills=item.missing_skills,
                direction=direction,
            )
        await match.insert()
        match_ids.append(str(match.id))

    logger.info(
        "persist_matches_node: %d matches persisted",
        len(match_ids),
        extra={"trace_id": get_trace_id()},
    )

    return {"persisted_match_ids": match_ids}


async def notify_candidates_node(state: JDToCandidatesState) -> dict:
    """Send notification emails to matched candidates (Case A, conditional).

    Only runs if state.notify is True. Sends emails to each matched candidate
    with their match score and highlights.

    Args:
        state: Current graph state with reranked_results and notify flag.

    Returns:
        State update dict with email_logs.
    """
    if state.error or not state.notify or not state.reranked_results:
        return {}

    email_logs = []
    for item in state.reranked_results:
        candidate = await Candidate.get(item.entity_id)
        if not candidate or not candidate.email:
            continue

        subject = "New job opportunity match"
        body = (
            f"Hi {candidate.name},\n\n"
            f"You've been matched with a job opportunity.\n"
            f"Score: {item.score}/100\n"
            f"Highlights: {', '.join(item.highlights)}\n\n"
            f"Best regards,\nTalentMatch AI"
        )

        log = await send_notification(
            recipient=candidate.email,
            subject=subject,
            body=body,
        )
        email_logs.append(log)

    logger.info(
        "notify_candidates_node: %d emails sent/logged",
        len(email_logs),
        extra={"trace_id": get_trace_id()},
    )

    return {"email_logs": email_logs}


async def notify_candidate_node(state: ResumeToJDsState) -> dict:
    """Send notification email to the candidate with their matching JDs (Case B, default).

    Always runs in Case B (resume→JDs) — this is the default email-on-match step.

    Args:
        state: Current graph state with reranked_results and entity_id.

    Returns:
        State update dict with email_logs.
    """
    if state.error or not state.entity_id or not state.reranked_results:
        return {}

    candidate = await Candidate.get(state.entity_id)
    if not candidate or not candidate.email:
        return {}

    matched_jds = []
    for item in state.reranked_results:
        jd = await JD.get(item.entity_id)
        if jd:
            matched_jds.append(
                f"- {jd.title} at {jd.company} (Score: {item.score}/100)"
            )

    subject = "Your matching job openings"
    body = (
        f"Hi {candidate.name},\n\n"
        f"Here are your top matching job openings:\n\n"
        f"{chr(10).join(matched_jds)}\n\n"
        f"Best regards,\nTalentMatch AI"
    )

    log = await send_notification(
        recipient=candidate.email,
        subject=subject,
        body=body,
    )

    logger.info(
        "notify_candidate_node: email sent to %s",
        candidate.email,
        extra={"trace_id": get_trace_id()},
    )

    return {"email_logs": [log]}
