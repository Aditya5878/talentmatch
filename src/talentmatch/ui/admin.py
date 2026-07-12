import asyncio
import os

import httpx
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


def build_admin_tab() -> None:
    with gr.Tab("Admin - Data Ingestion"):
        gr.Markdown("## Data Ingestion")

        with gr.Row():
            file_input = gr.File(
                label="Upload Files (PDF, DOCX, TXT)",
                file_count="multiple",
                type="binary",
            )
            file_type = gr.Radio(
                choices=[("Resume", "resume"), ("Job Description", "jd")],
                label="File Type",
                value="resume",
            )

        upload_btn = gr.Button("Upload & Ingest", variant="primary")
        batch_id_display = gr.Textbox(label="Batch ID", interactive=False)
        status_display = gr.Dataframe(
            label="Ingestion Status",
            headers=["Filename", "Type", "Status", "Error"],
            interactive=False,
        )
        refresh_btn = gr.Button("Refresh Status")

        gr.Markdown("---")
        gr.Markdown("## Browse Ingested Documents")

        with gr.Row():
            browse_type = gr.Radio(
                choices=[("Candidates", "candidates"), ("Job Descriptions", "jds")],
                label="Document Type",
                value="candidates",
            )
            doc_dropdown = gr.Dropdown(
                label="Select Document",
                choices=[],
                interactive=True,
            )
        refresh_docs_btn = gr.Button("Refresh Document List")
        doc_json = gr.JSON(label="Parsed Document")

        async def upload_and_ingest(files, ftype):
            if not files:
                return "", gr.Dataframe(value=[])

            httpx_files = []
            for item in files:
                if isinstance(item, tuple):
                    name, data = item
                else:
                    name = item.name if hasattr(item, "name") else "file"
                    data = item if isinstance(item, bytes) else getattr(item, "read", lambda: b"")()
                field = "resumes" if ftype == "resume" else "jds"
                httpx_files.append((field, (name, data)))

            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{API_BASE}/ingest/batch",
                    files=httpx_files,
                )
                if resp.status_code != 200:
                    return f"Error: {resp.text}", gr.Dataframe(value=[])

                data = resp.json()
                batch_id = data["batch_id"]

                await asyncio.sleep(1)

                status_resp = await client.get(
                    f"{API_BASE}/ingest/batch/{batch_id}/status"
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    rows = [
                        [i["filename"], i["file_type"], i["status"], i.get("error") or ""]
                        for i in status_data.get("items", [])
                    ]
                    return batch_id, gr.Dataframe(value=rows)

                return batch_id, gr.Dataframe(value=[])

        async def refresh_status(batch_id):
            if not batch_id:
                return gr.Dataframe(value=[])
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{API_BASE}/ingest/batch/{batch_id}/status"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    rows = [
                        [i["filename"], i["file_type"], i["status"], i.get("error") or ""]
                        for i in data.get("items", [])
                    ]
                    return gr.Dataframe(value=rows)
                return gr.Dataframe(value=[])

        async def refresh_doc_list(doc_type):
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{API_BASE}/{doc_type}")
                if resp.status_code == 200:
                    docs = resp.json()
                    choices = []
                    for d in docs:
                        if doc_type == "candidates":
                            label = f"{d.get('name', '?')} ({d.get('email', '?')})"
                        else:
                            label = f"{d.get('title', '?')} @ {d.get('company', '?')}"
                        choices.append((label, d["id"]))
                    return gr.Dropdown(choices=choices)
                return gr.Dropdown(choices=[])

        async def load_document(doc_type, doc_id):
            if not doc_id:
                return {}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{API_BASE}/{doc_type}/{doc_id}")
                if resp.status_code == 200:
                    return resp.json().get("parsed_json", {})
                return {"error": f"Failed to load: {resp.text}"}

        upload_btn.click(
            fn=upload_and_ingest,
            inputs=[file_input, file_type],
            outputs=[batch_id_display, status_display],
        )

        refresh_btn.click(
            fn=refresh_status,
            inputs=[batch_id_display],
            outputs=[status_display],
        )

        refresh_docs_btn.click(
            fn=refresh_doc_list,
            inputs=[browse_type],
            outputs=[doc_dropdown],
        )

        doc_dropdown.change(
            fn=load_document,
            inputs=[browse_type, doc_dropdown],
            outputs=[doc_json],
        )
