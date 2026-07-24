import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from talentmatch.ingestion.worker import process_batch
from talentmatch.models import BatchJob
from talentmatch.models.enums import BatchStatus, EntityType

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_active_tasks: dict[str, asyncio.Task] = {}


class BatchStatusResponse(BaseModel):
    """Response model for batch ingestion status."""

    batch_id: str
    status: str
    total_items: int
    completed_items: int
    items: list[dict]


class IngestFileInfo(BaseModel):
    """Per-file ingestion status info."""

    filename: str
    file_type: EntityType
    status: str = "queued"


@router.post("/batch")
async def ingest_batch(
    resumes: list[UploadFile] = [],
    jds: list[UploadFile] = [],
):
    """Accept multiple resume and/or JD files and start async batch ingestion.

    Creates a BatchJob document, spawns a background task to process all files
    through extract → parse → chunk → embed → persist. Returns the batch_id
    immediately for status polling.

    Args:
        resumes: List of resume files (PDF, DOCX, TXT).
        jds: List of JD files (PDF, DOCX, TXT).

    Returns:
        {"batch_id": str, "total_files": int}

    Raises:
        HTTPException: 400 if no files provided.
    """
    files: list[tuple[str, bytes, str]] = []

    for f in resumes:
        content = await f.read()
        files.append((f.filename or "unknown", content, EntityType.candidate))

    for f in jds:
        content = await f.read()
        files.append((f.filename or "unknown", content, EntityType.jd))

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    batch = BatchJob(
        status=BatchStatus.queued,
        total_items=len(files),
        completed_items=0,
        items=[
            {
                "filename": fn,
                "file_type": ft,
                "file_hash": "",
                "status": "queued",
                "error": None,
            }
            for fn, _, ft in files
        ],
    )
    await batch.insert()

    task = asyncio.create_task(process_batch(batch, files))
    _active_tasks[str(batch.id)] = task
    task.add_done_callback(lambda _: _active_tasks.pop(str(batch.id), None))

    return {"batch_id": str(batch.id), "total_files": len(files)}


@router.get("/batch/{batch_id}/status")
async def batch_status(batch_id: str):
    """Get the current status and per-item progress of a batch ingestion job.

    Args:
        batch_id: The BatchJob document ID.

    Returns:
        BatchStatusResponse with overall status and per-file details.

    Raises:
        HTTPException: 404 if batch not found.
    """
    batch = await BatchJob.get(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    return BatchStatusResponse(
        batch_id=str(batch.id),
        status=batch.status,
        total_items=batch.total_items,
        completed_items=batch.completed_items,
        items=[
            {
                "filename": item.filename,
                "file_type": item.file_type,
                "status": item.status,
                "error": item.error,
            }
            for item in batch.items
        ],
    )
