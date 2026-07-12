from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    llm_model: str = "groq/llama-3.3-70b-versatile"

    mongodb_uri: str = "mongodb://localhost:27017/talentmatch"
    qdrant_url: str = "http://localhost:6333"

    embedding_model: str = "gemini/gemini-embedding-001"
    embedding_batch_size: int = 100
    embedding_dimension: int = 768

    qdrant_collection_candidate: str = "candidate_chunks"
    qdrant_collection_jd: str = "jd_chunks"

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
