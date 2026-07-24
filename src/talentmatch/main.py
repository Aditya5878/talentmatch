import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from talentmatch.config import settings
from talentmatch.db.mongo import init_mongodb
from talentmatch.db.qdrant import ensure_collections, get_qdrant_client
from talentmatch.models import Candidate, JD
from talentmatch.routers import health, ingestion, matching, search


async def _wait_for_mongodb(retries: int = 10, delay: int = 3) -> None:
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
    await _wait_for_mongodb()

    qdrant = get_qdrant_client()
    ensure_collections(qdrant)

    if "/" not in settings.embedding_model:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embedding_model)
        app.state.embedding_model = model

    yield


app = FastAPI(
    title="TalentMatch AI",
    description="Bidirectional Resume-JD matching platform with RAG + agentic evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(matching.router)
app.include_router(search.router)


@app.get("/candidates")
async def list_candidates():
    docs = await Candidate.find_all().to_list()
    return [
        {"id": str(d.id), "name": d.name, "email": d.email, "created_at": d.created_at.isoformat()}
        for d in docs
    ]


@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    doc = await Candidate.get(candidate_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return doc.model_dump()


@app.get("/jds")
async def list_jds():
    docs = await JD.find_all().to_list()
    return [
        {"id": str(d.id), "title": d.title, "company": d.company, "created_at": d.created_at.isoformat()}
        for d in docs
    ]


@app.get("/jds/{jd_id}")
async def get_jd(jd_id: str):
    doc = await JD.get(jd_id)
    if not doc:
        raise HTTPException(status_code=404, detail="JD not found")
    return doc.model_dump()
