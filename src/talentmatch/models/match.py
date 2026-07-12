from datetime import datetime
from typing import Optional

from beanie import Document
from pydantic import Field

from talentmatch.models.enums import MatchDirection


class Match(Document):
    jd_id: Optional[str] = None
    candidate_id: Optional[str] = None
    query_text: str = ""
    expanded_query_terms: list[str] = Field(default_factory=list)
    score: float = 0.0
    rationale: str = ""
    highlights: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    direction: MatchDirection = MatchDirection.jd_to_candidate
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "matches"
