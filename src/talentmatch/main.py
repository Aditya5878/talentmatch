import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from talentmatch.config import settings
from talentmatch.db.mongo import init_mongodb
from talentmatch.db.qdrant import ensure_collections, get_qdrant_client
from talentmatch.models import Candidate, JD
from talentmatch.routers import health, ingestion, matching, search
from talentmatch.utils.logging import TraceIDMiddleware, setup_logging

logger = logging.getLogger("talentmatch")


async def _wait_for_mongodb(retries: int = 10, delay: int = 3) -> None:
    """Retry MongoDB connection with exponential delay.

    Args:
        retries: Maximum number of connection attempts.
        delay: Seconds to wait between attempts.

    Raises:
        Exception: If all retry attempts fail.
    """
    for attempt in range(retries):
        try:
            await init_mongodb()
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler — initializes all services on startup.

    - Sets up structured logging
    - Connects to MongoDB with retries
    - Creates Qdrant collections if needed
    - Loads local sentence-transformers model (if not using remote embeddings)
    """
    setup_logging()
    logger.info("Starting TalentMatch API")

    await _wait_for_mongodb()
    logger.info("MongoDB connected")

    qdrant = get_qdrant_client()
    ensure_collections(qdrant)
    logger.info("Qdrant collections ensured")

    if "/" not in settings.embedding_model:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model)
        app.state.embedding_model = model
        logger.info("Local embedding model loaded: %s", settings.embedding_model)

    yield


app = FastAPI(
    title="TalentMatch AI",
    description="Bidirectional Resume-JD matching platform with RAG + agentic evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(TraceIDMiddleware)

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(matching.router)
app.include_router(search.router)


@app.get("/candidates")
async def list_candidates():
    """List all ingested candidates with id, name, email, and created_at."""
    docs = await Candidate.find_all().to_list()
    return [
        {"id": str(d.id), "name": d.name, "email": d.email, "created_at": d.created_at.isoformat()}
        for d in docs
    ]


@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Get a full candidate document by ID, including parsed_json."""
    doc = await Candidate.get(candidate_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return doc.model_dump()


@app.get("/jds")
async def list_jds():
    """List all ingested job descriptions with id, title, company, and created_at."""
    docs = await JD.find_all().to_list()
    return [
        {"id": str(d.id), "title": d.title, "company": d.company, "created_at": d.created_at.isoformat()}
        for d in docs
    ]


@app.get("/jds/{jd_id}")
async def get_jd(jd_id: str):
    """Get a full JD document by ID, including parsed_json."""
    doc = await JD.get(jd_id)
    if not doc:
        raise HTTPException(status_code=404, detail="JD not found")
    return doc.model_dump()
