"""Recruiter (Case A) and Candidate (Case B) chat tabs for the Gradio UI.

ChatGPT-style interfaces: single chatbot + file attach + text input.
All results shown as Markdown tables inline in chat messages.
"""

import os
import tempfile

import httpx
import gradio as gr

from talentmatch.ingestion.extractor import extract_text

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ──────────────────────────────────────────────────────────────────
# API helper
# ──────────────────────────────────────────────────────────────────

async def _send_chat(message: str, session_id: str | None, mode: str) -> dict:
    """Send a message to the chat endpoint."""
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {"message": message, "top_k": 5}
        if session_id:
            payload["session_id"] = session_id
        resp = await client.post(f"{API_BASE}/chat/{mode}", json=payload)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"API error: {resp.status_code}"}


# ──────────────────────────────────────────────────────────────────
# Markdown formatters
# ──────────────────────────────────────────────────────────────────

def _format_matches(matches: list[dict], entity_label: str) -> str:
    """Format match results as a Markdown table with suggested follow-ups."""
    if not matches:
        return "_No results found._"

    lines = [
        f"Found **{len(matches)}** {entity_label}:\n",
        "| # | Score | Highlights | Matched Skills | Missing Skills |",
        "|---|-------|-----------|----------------|----------------|",
    ]

    for i, m in enumerate(matches, 1):
        score = f"{m.get('score', 0):.0f}/100"
        highlights = ", ".join(m.get("highlights", [])[:3]) or "—"
        matched = ", ".join(m.get("matched_skills", [])[:4]) or "—"
        missing = ", ".join(m.get("missing_skills", [])[:4]) or "—"
        lines.append(f"| {i} | **{score}** | {highlights} | {matched} | {missing} |")

    lines.append("")
    lines.append("**You can:**")
    lines.append(f"- \"Remove {entity_label.split()[0]} #2\"")
    lines.append(f"- \"Email the remaining {entity_label.split()[0]}s\"")

    return "\n".join(lines)


def _format_gap_suggestions(suggestions: list[dict]) -> str:
    """Format gap suggestions as Markdown."""
    if not suggestions:
        return ""

    lines = ["\n---\n**Resume Improvement Suggestions:**\n"]
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


def _format_response(result: dict, mode: str) -> str:
    """Convert an API response dict into a single Markdown string for the chat."""
    if "error" in result:
        return f"**Error:** {result['error']}"

    matches = result.get("matches", [])
    summary = result.get("refinement_summary", "")
    email_logs = result.get("email_logs", [])
    gap_sugs = result.get("gap_suggestions", [])
    entity_label = "candidates" if mode == "recruiter" else "openings"

    parts = []

    if matches:
        parts.append(_format_matches(matches, entity_label))
        if mode == "candidate" and gap_sugs:
            parts.append(_format_gap_suggestions(gap_sugs))

    elif summary:
        parts.append(summary)

    elif email_logs:
        recipients = [e.get("recipient", "?") for e in email_logs[:10]]
        parts.append(
            f"**Sent {len(email_logs)} email(s):**\n"
            + "\n".join(f"- {r}" for r in recipients)
        )

    else:
        parts.append("_Done._")

    return "\n\n".join(parts)


def _handle_file(uploaded_file, mode: str) -> str:
    """Extract text from an uploaded file and return a labeled prefix."""
    if uploaded_file is None:
        return ""

    try:
        file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
        text = extract_text(file_path)
        filename = os.path.basename(file_path)
        label = "Job Description" if mode == "recruiter" else "Resume"
        return f"[Attached {label}: {filename}]\n{text}\n\n"
    except Exception as e:
        return f"[File attachment failed: {e}]\n\n"


# ──────────────────────────────────────────────────────────────────
# Shared builder
# ──────────────────────────────────────────────────────────────────

def _build_chat_tab(mode: str) -> None:
    """Build a ChatGPT-style tab for the given mode (recruiter or candidate).

    Args:
        mode: "recruiter" or "candidate" — determines endpoint and labels.
    """
    is_recruiter = mode == "recruiter"
    title = "Find Candidates" if is_recruiter else "Find Job Openings"
    placeholder = (
        "Paste a Job Description, attach a file, or type a skill query..."
        if is_recruiter
        else "Paste your resume, attach a file, or type a skill query..."
    )
    chat_placeholder = "Type a message..."
    file_label = "📎 Attach JD" if is_recruiter else "📎 Attach Resume"
    entity_label = "candidates" if is_recruiter else "matching openings"

    tab_name = "Recruiter" if is_recruiter else "Candidate"

    with gr.Tab(tab_name):
        session_state = gr.State(value=None)

        gr.Markdown(f"## 🤖 TalentMatch — {tab_name}")

        chatbot = gr.Chatbot(
            value=[
                {"role": "assistant", "content": (
                    f"Welcome! I can help you find {entity_label}.\n\n"
                    f"You can:\n"
                    f"- Paste a {'Job Description' if is_recruiter else 'resume'} directly\n"
                    f"- Attach a file using the 📎 button\n"
                    f"- Type a skill query like \"{'Java developers with Spring Boot' if is_recruiter else 'Python backend openings'}\""
                )},
            ],
            height=520,
        )

        with gr.Row():
            file_upload = gr.UploadButton(
                label=file_label,
                file_types=[".pdf", ".docx", ".txt"],
                scale=0,
            )
            msg_input = gr.Textbox(
                placeholder=chat_placeholder,
                show_label=False,
                scale=5,
                container=False,
            )
            send_btn = gr.Button("➤", scale=0, min_width=60)

        # ── handlers ───────────────────────────────────────────

        async def on_send(message, file, history, sid):
            if not message or not message.strip():
                return history, sid, None, ""

            file_prefix = _handle_file(file, mode)
            full_message = file_prefix + message

            result = await _send_chat(full_message, sid, mode)
            response_md = _format_response(result, mode)
            new_sid = result.get("session_id", sid)

            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_md},
            ]

            return new_history, new_sid, None, ""

        send_btn.click(
            fn=on_send,
            inputs=[msg_input, file_upload, chatbot, session_state],
            outputs=[chatbot, session_state, file_upload, msg_input],
        )

        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, file_upload, chatbot, session_state],
            outputs=[chatbot, session_state, file_upload, msg_input],
        )


# ──────────────────────────────────────────────────────────────────
# Public builders (called from app.py)
# ──────────────────────────────────────────────────────────────────

def build_recruiter_tab() -> None:
    """Build the Recruiter (Case A) chat tab."""
    _build_chat_tab("recruiter")


def build_candidate_tab() -> None:
    """Build the Candidate (Case B) chat tab."""
    _build_chat_tab("candidate")
