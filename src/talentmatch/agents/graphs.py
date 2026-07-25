"""LangGraph graph definitions for matching workflows.

Defines the JD→Candidates graph (Case A), Resume→JDs graph (Case B),
and Free-Text Search graph as compiled LangGraph StateGraphs. Each graph
chains nodes sequentially with conditional branching for error handling.
"""

import logging

from langgraph.graph import END, StateGraph

from talentmatch.agents.nodes import (
    apply_refinement_node,
    classify_intent_node,
    diff_skills_node,
    expand_query_node,
    format_suggestions_node,
    hybrid_retrieve_node,
    llm_suggest_edits_node,
    log_email_results_node,
    notify_candidate_node,
    notify_candidates_node,
    parse_jd_node,
    parse_resume_node,
    persist_matches_node,
    persist_session_results_node,
    resolve_reference_node,
    resolve_scope_node,
    rerank_score_node,
    retrieve_candidates_node,
    retrieve_jds_node,
    send_email_node,
)
from talentmatch.agents.state import (
    ActionState,
    BaseGraphState,
    ChatState,
    FreeTextSearchState,
    JDToCandidatesState,
    RefinementState,
    ResumeToJDsState,
)
from talentmatch.utils.logging import get_trace_id

logger = logging.getLogger("talentmatch.agents.graphs")


def _check_error(state: BaseGraphState) -> str:
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

    Nodes: parse_resume → retrieve_jds → rerank_score → diff_skills →
           llm_suggest_edits → format_suggestions → persist_matches → notify_candidate

    The gap suggestion sub-graph (diff_skills → llm_suggest_edits → format_suggestions)
    generates concrete resume improvement suggestions per top-matched JD.
    The notify_candidate node always runs (default email-on-match in Case B).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(ResumeToJDsState)

    # Add nodes
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("retrieve_jds", retrieve_jds_node)
    graph.add_node("rerank_score", rerank_score_node)
    graph.add_node("diff_skills", diff_skills_node)
    graph.add_node("llm_suggest_edits", llm_suggest_edits_node)
    graph.add_node("format_suggestions", format_suggestions_node)
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
        {"continue": "diff_skills", "end": END},
    )
    graph.add_conditional_edges(
        "diff_skills",
        _check_error,
        {"continue": "llm_suggest_edits", "end": END},
    )
    graph.add_conditional_edges(
        "llm_suggest_edits",
        _check_error,
        {"continue": "format_suggestions", "end": END},
    )
    graph.add_conditional_edges(
        "format_suggestions",
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


def build_free_text_search_graph() -> StateGraph:
    """Build the Free-Text Search graph (Spec 7.4).

    Nodes: expand_query → hybrid_retrieve → rerank_score → persist_matches

    No parse step — the raw query is expanded by the LLM, then used for
    hybrid retrieval against either the candidate or JD collection.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(FreeTextSearchState)

    # Add nodes
    graph.add_node("expand_query", expand_query_node)
    graph.add_node("hybrid_retrieve", hybrid_retrieve_node)
    graph.add_node("rerank_score", rerank_score_node)
    graph.add_node("persist_matches", persist_matches_node)

    # Set entry point
    graph.set_entry_point("expand_query")

    # Add edges with conditional branching
    graph.add_conditional_edges(
        "expand_query",
        _check_error,
        {"continue": "hybrid_retrieve", "end": END},
    )
    graph.add_conditional_edges(
        "hybrid_retrieve",
        _check_error,
        {"continue": "rerank_score", "end": END},
    )
    graph.add_conditional_edges(
        "rerank_score",
        _check_error,
        {"continue": "persist_matches", "end": END},
    )
    graph.add_edge("persist_matches", END)

    return graph.compile()


# Compiled graph singletons (created once at import time)
jd_to_candidates_graph = build_jd_to_candidates_graph()
resume_to_jds_graph = build_resume_to_jds_graph()
free_text_search_graph = build_free_text_search_graph()


def _check_refinement_error(state: RefinementState) -> str:
    """Branching function for refinement graph: route to END if error."""
    if state.error:
        logger.warning(
            "Refinement graph error: %s", state.error, extra={"trace_id": get_trace_id()}
        )
        return "end"
    return "continue"


def _check_action_error(state: ActionState) -> str:
    """Branching function for action graph: route to END if error."""
    if state.error:
        logger.warning(
            "Action graph error: %s", state.error, extra={"trace_id": get_trace_id()}
        )
        return "end"
    return "continue"


def build_refinement_graph() -> StateGraph:
    """Build the Refinement graph (Spec 7.5).

    Nodes: resolve_reference → apply_refinement → persist_session_results

    Operates on the session's active result set — no retrieval or reranking needed.
    Maps user references to specific results and removes/keeps them.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(RefinementState)

    graph.add_node("resolve_reference", resolve_reference_node)
    graph.add_node("apply_refinement", apply_refinement_node)
    graph.add_node("persist_session_results", persist_session_results_node)

    graph.set_entry_point("resolve_reference")

    graph.add_conditional_edges(
        "resolve_reference",
        _check_refinement_error,
        {"continue": "apply_refinement", "end": END},
    )
    graph.add_conditional_edges(
        "apply_refinement",
        _check_refinement_error,
        {"continue": "persist_session_results", "end": END},
    )
    graph.add_edge("persist_session_results", END)

    return graph.compile()


def build_action_graph() -> StateGraph:
    """Build the Action graph (Spec 7.6).

    Nodes: resolve_scope → send_email → log_email_results

    Operates on the session's active result set — sends emails to
    matched recipients. Resolves scope (all or subset) then emails.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(ActionState)

    graph.add_node("resolve_scope", resolve_scope_node)
    graph.add_node("send_email", send_email_node)
    graph.add_node("log_email_results", log_email_results_node)

    graph.set_entry_point("resolve_scope")

    graph.add_conditional_edges(
        "resolve_scope",
        _check_action_error,
        {"continue": "send_email", "end": END},
    )
    graph.add_conditional_edges(
        "send_email",
        _check_action_error,
        {"continue": "log_email_results", "end": END},
    )
    graph.add_edge("log_email_results", END)

    return graph.compile()


def build_intent_router_graph() -> StateGraph:
    """Build the Intent Router graph (Spec 7.0).

    Classifies the user's intent and returns the classification.
    The API endpoint reads the intent from the final state and
    dispatches to the appropriate sub-graph (matching, refinement, action).

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(ChatState)

    graph.add_node("classify_intent", classify_intent_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", END)

    return graph.compile()


# Compiled graph singletons (created once at import time)
refinement_graph = build_refinement_graph()
action_graph = build_action_graph()
intent_router_graph = build_intent_router_graph()
