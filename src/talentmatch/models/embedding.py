from datetime import datetime

from beanie import Document
from pydantic import Field

from talentmatch.models.enums import EntityType


class EmbeddingIndex(Document):
    entity_type: EntityType
    entity_id: str
    qdrant_point_id: str
    chunk_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "embeddings_index"
