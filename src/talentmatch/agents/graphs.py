"""LangGraph graph definitions for matching workflows.

Defines the JD→Candidates graph (Case A) and Resume→JDs graph (Case B) as
compiled LangGraph StateGraphs. Each graph chains nodes sequentially with
conditional branching for optional notification steps.
"""

import logging

from langgraph.graph import END, StateGraph

from talentmatch.agents.nodes import (
    notify_candidate_node,
    notify_candidates_node,
    parse_jd_node,
    parse_resume_node,
    persist_matches_node,
    rerank_score_node,
    retrieve_candidates_node,
    retrieve_jds_node,
)
from talentmatch.agents.state import JDToCandidatesState, ResumeToJDsState
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.agents.graphs")


def _check_error(state: JDToCandidatesState | ResumeToJDsState) -> str:
    """Branching function: route to END if error occurred, else continue.

    Args:
        state: Current graph state.

    Returns:
        Next node name or "end".
    """
    if state.error:
        logger.warning(
            "Graph error: %s", state.error, extra={"trace_id": get_trace_id()}
        )
        return "end"
    return "continue"


def _should_notify_jd(state: JDToCandidatesState) -> str:
    """Branching function: route to notify_candidates only if notify=True.

    Args:
        state: Current graph state.

    Returns:
        "notify_candidates" or "end".
    """
    return "notify_candidates" if state.notify else "end"


def build_jd_to_candidates_graph() -> StateGraph:
    """Build the JD→Candidates graph (Case A).

    Nodes: parse_jd → retrieve_candidates → rerank_score → persist_matches → [notify_candidates]

    The notify_candidates node only runs if state.notify is True.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(JDToCandidatesState)

    # Add nodes
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("retrieve_candidates", retrieve_candidates_node)
    graph.add_node("rerank_score", rerank_score_node)
    graph.add_node("persist_matches", persist_matches_node)
    graph.add_node("notify_candidates", notify_candidates_node)

    # Set entry point
    graph.set_entry_point("parse_jd")

    # Add edges with conditional branching
    graph.add_conditional_edges(
        "parse_jd",
        _check_error,
        {"continue": "retrieve_candidates", "end": END},
    )
    graph.add_conditional_edges(
        "retrieve_candidates",
        _check_error,
        {"continue": "rerank_score", "end": END},
    )
    graph.add_conditional_edges(
        "rerank_score",
        _check_error,
        {"continue": "persist_matches", "end": END},
    )
    graph.add_conditional_edges(
        "persist_matches",
        _should_notify_jd,
        {"notify_candidates": "notify_candidates", "end": END},
    )
    graph.add_edge("notify_candidates", END)

    return graph.compile()


def build_resume_to_jds_graph() -> StateGraph:
    """Build the Resume→JDs graph (Case B).

    Nodes: parse_resume → retrieve_jds → rerank_score → persist_matches → notify_candidate

    The notify_candidate node always runs (default email-on-match in Case B).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(ResumeToJDsState)

    # Add nodes
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("retrieve_jds", retrieve_jds_node)
    graph.add_node("rerank_score", rerank_score_node)
    graph.add_node("persist_matches", persist_matches_node)
    graph.add_node("notify_candidate", notify_candidate_node)

    # Set entry point
    graph.set_entry_point("parse_resume")

    # Add edges with conditional branching
    graph.add_conditional_edges(
        "parse_resume",
        _check_error,
        {"continue": "retrieve_jds", "end": END},
    )
    graph.add_conditional_edges(
        "retrieve_jds",
        _check_error,
        {"continue": "rerank_score", "end": END},
    )
    graph.add_conditional_edges(
        "rerank_score",
        _check_error,
        {"continue": "persist_matches", "end": END},
    )
    graph.add_conditional_edges(
        "persist_matches",
        _check_error,
        {"continue": "notify_candidate", "end": END},
    )
    graph.add_edge("notify_candidate", END)

    return graph.compile()


# Compiled graph singletons (created once at import time)
jd_to_candidates_graph = build_jd_to_candidates_graph()
resume_to_jds_graph = build_resume_to_jds_graph()
