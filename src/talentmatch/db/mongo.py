from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from talentmatch.config import settings
from talentmatch.models import (
    BatchJob,
    Candidate,
    EmailLog,
    EmbeddingIndex,
    JD,
    Match,
)


async def init_mongodb() -> None:
    """Initialize MongoDB connection and register Beanie document models.

    Creates an AsyncIOMotorClient using the configured URI and initializes
    Beanie ODM with all document models (Candidate, JD, BatchJob,
    EmbeddingIndex, Match, EmailLog).

    Raises:
        Exception: If MongoDB connection fails after retries in the caller.
    """
    client = AsyncIOMotorClient(settings.mongodb_uri)
    await init_beanie(
        database=client.get_default_database(),
        document_models=[
            Candidate,
            JD,
            BatchJob,
            EmbeddingIndex,
            Match,
            EmailLog,
        ],
    )
