from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from talentmatch.config import settings
from talentmatch.models import (
    BatchJob,
    Candidate,
    EmbeddingIndex,
    JD,
    Match,
)


async def init_mongodb() -> None:
    client = AsyncIOMotorClient(settings.mongodb_uri)
    await init_beanie(
        database=client.get_default_database(),
        document_models=[
            Candidate,
            JD,
            BatchJob,
            EmbeddingIndex,
            Match,
        ],
    )
