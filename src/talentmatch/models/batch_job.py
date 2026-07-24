from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import BaseModel, Field

from talentmatch.models.enums import BatchStatus, EntityType, ItemStatus


class BatchItem(BaseModel):
    filename: str
    file_type: EntityType
    file_hash: str = ""
    status: ItemStatus = ItemStatus.queued
    error: Optional[str] = None


class BatchJob(Document):
    entity_type: Optional[EntityType] = None
    status: BatchStatus = BatchStatus.queued
    items: list[BatchItem] = Field(default_factory=list)
    total_items: int = 0
    completed_items: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "batch_jobs"
