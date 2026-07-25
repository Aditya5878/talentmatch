"""Recruiter (Case A) and Candidate (Case B) chat tabs for the Gradio UI.

Builds reusable chat-based matching interfaces that call the
FastAPI backend's /chat/recruiter and /chat/candidate endpoints.
"""

import asyncio
import os

import httpx
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


async def _send_chat_message(message: str, session_id: str | None, mode: str) -> dict:
    """Send a message to the chat endpoint and return the response.

    Args:
        message: The user's message.
        session_id: Optional session ID for follow-up messages.
        mode: "recruiter" or "candidate".

    Returns:
        Response dict from the API.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {"message": message, "top_k": 5}
        if session_id:
            payload["session_id"] = session_id

        resp = await client.post(f"{API_BASE}/chat/{mode}", json=payload)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"API error: {resp.status.status_code}"}


def _format_results_table(matches: list[dict]) -> list[list]:
    """Format match results into rows for a Gradio Dataframe.

    Args:
        matches: List of match result dicts.

    Returns:
        List of rows: [score, entity_id, highlights, matched_skills, missing_skills]
    """
    rows = []
    for m in matches:
        rows.append([
            f"{m.get('score', 0):.0f}/100",
            m.get("entity_id", "?")[:8] + "...",
            ", ".join(m.get("highlights", [])[:3]),
            ", ".join(m.get("matched_skills", [])[:4]),
            ", ".join(m.get("missing_skills", [])[:4]),
        ])
    return rows


def _format_gap_suggestions(suggestions: list[dict]) -> str:
    """Format gap suggestions into readable Markdown.

    Args:
        suggestions: List of gap suggestion dicts.

    Returns:
        Markdown string with suggestions per JD.
    """
    if not suggestions:
        return ""

    lines = ["### Resume Improvement Suggestions\n"]
    for s in suggestions:
        title = s.get("jd_title", "Matched Job")
        company = s.get("jd_company", "")
        sugs = s.get("suggestions", [])
        if sugs:
            lines.append(f"**{title}** at {company}:")
            for sug in sugs:
                lines.append(f"- {sug}")
            lines.append("")

    return "\n".join(lines)


def build_recruiter_tab() -> None:
    """Build the Recruiter (Case A) chat tab.

    Features:
    - Text input for JD paste or free-text query
    - Chatbot for follow-up conversation
    - Results dataframe showing matched candidates
    - Action buttons for emailing candidates
    """
    with gr.Tab("Recruiter (Case A)"):
        gr.Markdown("## Find Candidates")
        gr.Markdown("Paste a Job Description or type a skill query (e.g. 'Java developers with Spring Boot')")

        session_state = gr.State(value=None)

        with gr.Row():
            query_input = gr.Textbox(
                label="Job Description or Query",
                placeholder="Paste your JD or type a search query...",
                lines=6,
            )

        search_btn = gr.Button("Search Candidates", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### Results")

        results_table = gr.Dataframe(
            headers=["Score", "ID", "Highlights", "Matched Skills", "Missing Skills"],
            interactive=False,
        )

        chatbot = gr.Chatbot(label="Follow-up Conversation", height=300)

        with gr.Row():
            chat_input = gr.Textbox(
                label="Follow-up message",
                placeholder="e.g. 'Remove candidate 2', 'Email the remaining candidates'",
                scale=4,
            )
            send_btn = gr.Button("Send", scale=1)

        with gr.Row():
            email_all_btn = gr.Button("Email All Active Candidates")
            email_status = gr.Markdown("")

        async def do_search(query, history):
            if not query.strip():
                return history, gr.Dataframe(value=[]), None, ""

            result = await _send_chat_message(query, None, "recruiter")
            if "error" in result:
                return history + [(query, f"Error: {result['error']}")], gr.Dataframe(value=[]), None, ""

            session_id = result.get("session_id")
            matches = result.get("matches", [])
            rows = _format_results_table(matches)
            steps = ", ".join(result.get("graph_steps", []))

            history = history + [
                (query, f"Found {len(matches)} candidates. Graph steps: {steps}")
            ]

            return history, gr.Dataframe(value=rows), session_id, ""

        async def do_chat(message, history, sid):
            if not message.strip():
                return history, sid, gr.Dataframe(value=[]), ""

            result = await _send_chat_message(message, sid, "recruiter")
            if "error" in result:
                return history + [(message, f"Error: {result['error']}")], sid, gr.Dataframe(value=[]), ""

            new_sid = result.get("session_id", sid)
            matches = result.get("matches", [])
            summary = result.get("refinement_summary", "")
            email_logs = result.get("email_logs", [])

            if matches:
                rows = _format_results_table(matches)
                response = f"Updated results: {len(matches)} candidates."
            elif summary:
                rows = gr.Dataframe.value
                response = summary
            elif email_logs:
                rows = gr.Dataframe.value
                response = f"Sent {len(email_logs)} email(s)."
            else:
                rows = gr.Dataframe.value
                response = "Done."

            return history + [(message, response)], new_sid, gr.Dataframe(value=rows), ""

        async def do_email_all(sid):
            if not sid:
                return "No active session. Search first."

            result = await _send_chat_message("Email all active candidates", sid, "recruiter")
            if "error" in result:
                return f"Error: {result['error']}"

            email_logs = result.get("email_logs", [])
            return f"Sent {len(email_logs)} email(s)."

        search_btn.click(
            fn=do_search,
            inputs=[query_input, chatbot],
            outputs=[chatbot, results_table, session_state, chat_input],
        )

        send_btn.click(
            fn=do_chat,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state, results_table, chat_input],
        )

        chat_input.submit(
            fn=do_chat,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state, results_table, chat_input],
        )

        email_all_btn.click(
            fn=do_email_all,
            inputs=[session_state],
            outputs=[email_status],
        )


def build_candidate_tab() -> None:
    """Build the Candidate (Case B) chat tab.

    Features:
    - Text input for resume paste or free-text query
    - Chatbot for follow-up conversation
    - Results dataframe showing matched JDs
    - Gap suggestions panel
    """
    with gr.Tab("Candidate (Case B)"):
        gr.Markdown("## Find Job Openings")
        gr.Markdown("Paste your resume or type a skill query (e.g. 'Python backend openings')")

        session_state = gr.State(value=None)

        with gr.Row():
            query_input = gr.Textbox(
                label="Resume or Query",
                placeholder="Paste your resume or type a search query...",
                lines=6,
            )

        search_btn = gr.Button("Search Openings", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### Results")

        results_table = gr.Dataframe(
            headers=["Score", "JD ID", "Highlights", "Matched Skills", "Missing Skills"],
            interactive=False,
        )

        gap_suggestions_display = gr.Markdown("")

        chatbot = gr.Chatbot(label="Follow-up Conversation", height=300)

        with gr.Row():
            chat_input = gr.Textbox(
                label="Follow-up message",
                placeholder="e.g. 'Remove job 2', 'Email me these openings'",
                scale=4,
            )
            send_btn = gr.Button("Send", scale=1)

        async def do_search(query, history):
            if not query.strip():
                return history, gr.Dataframe(value=[]), None, "", ""

            result = await _send_chat_message(query, None, "candidate")
            if "error" in result:
                return history + [(query, f"Error: {result['error']}")], gr.Dataframe(value=[]), None, "", ""

            session_id = result.get("session_id")
            matches = result.get("matches", [])
            rows = _format_results_table(matches)
            gap_sugs = result.get("gap_suggestions", [])
            gap_md = _format_gap_suggestions(gap_sugs)
            steps = ", ".join(result.get("graph_steps", []))

            history = history + [
                (query, f"Found {len(matches)} matching openings. Graph steps: {steps}")
            ]

            return history, gr.Dataframe(value=rows), session_id, gap_md, ""

        async def do_chat(message, history, sid):
            if not message.strip():
                return history, sid, gr.Dataframe(value=[]), "", ""

            result = await _send_chat_message(message, sid, "candidate")
            if "error" in result:
                return history + [(message, f"Error: {result['error']}")], sid, gr.Dataframe(value=[]), "", ""

            new_sid = result.get("session_id", sid)
            matches = result.get("matches", [])
            summary = result.get("refinement_summary", "")
            email_logs = result.get("email_logs", [])
            gap_sugs = result.get("gap_suggestions", [])
            gap_md = _format_gap_suggestions(gap_sugs)

            if matches:
                rows = _format_results_table(matches)
                response = f"Updated results: {len(matches)} openings."
            elif summary:
                rows = gr.Dataframe.value
                response = summary
            elif email_logs:
                rows = gr.Dataframe.value
                response = f"Sent {len(email_logs)} email(s)."
            else:
                rows = gr.Dataframe.value
                response = "Done."

            return history + [(message, response)], new_sid, gr.Dataframe(value=rows), gap_md, ""

        search_btn.click(
            fn=do_search,
            inputs=[query_input, chatbot],
            outputs=[chatbot, results_table, session_state, gap_suggestions_display, chat_input],
        )

        send_btn.click(
            fn=do_chat,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state, results_table, gap_suggestions_display, chat_input],
        )

        chat_input.submit(
            fn=do_chat,
            inputs=[chat_input, chatbot, session_state],
            outputs=[chatbot, session_state, results_table, gap_suggestions_display, chat_input],
        )
