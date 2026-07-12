from datetime import datetime
from typing import Any

from beanie import Document
from pydantic import Field


class Candidate(Document):
    name: str = ""
    email: str = ""
    resume_raw_text: str = ""
    resume_file_path: str = ""
    parsed_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "candidates"
