from enum import StrEnum


class EntityType(StrEnum):
    candidate = "candidate"
    jd = "jd"


class BatchStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ItemStatus(StrEnum):
    queued = "queued"
    extracted = "extracted"
    parsed = "parsed"
    chunked = "chunked"
    embedded = "embedded"
    persisted = "persisted"
    failed = "failed"


class MatchDirection(StrEnum):
    jd_to_candidate = "jd_to_candidate"
    resume_to_jd = "resume_to_jd"
    keyword_to_candidate = "keyword_to_candidate"
    keyword_to_jd = "keyword_to_jd"
