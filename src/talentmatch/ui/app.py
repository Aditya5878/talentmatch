import gradio as gr

from talentmatch.ui.admin import build_admin_tab


def create_ui() -> gr.Blocks:
    """Create the Gradio UI application with tabbed layout.

    Builds a Blocks-based UI with tabs for admin ingestion,
    recruiter matching, and candidate matching.

    Returns:
        The configured Gradio Blocks application.
    """
    with gr.Blocks(
        title="TalentMatch AI",
        theme=gr.themes.Soft(),
    ) as ui:
        gr.Markdown("# TalentMatch AI")
        gr.Markdown("Bidirectional Resume-JD matching platform with RAG + agentic evaluation")

        with gr.Tabs():
            build_admin_tab()

    return ui


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
