from datetime import datetime
from typing import Any

from beanie import Document
from pydantic import Field


class JD(Document):
    title: str = ""
    company: str = ""
    jd_raw_text: str = ""
    jd_file_path: str = ""
    parsed_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "jds"
