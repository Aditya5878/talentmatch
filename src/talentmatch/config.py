from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file.

    Attributes:
        groq_api_key: API key for Groq LLM provider.
        tavily_api_key: API key for Tavily search provider.
        llm_model: LiteLLM model identifier for chat completions.
        mongodb_uri: MongoDB connection string.
        qdrant_url: Qdrant server URL (used in remote mode).
        qdrant_mode: Qdrant client mode — "local" (in-process) or "remote" (server).
        embedding_model: Sentence-transformers model name or LiteLLM remote model ID.
        embedding_batch_size: Number of texts per embedding API call.
        embedding_dimension: Vector dimension for the embedding model.
        qdrant_collection_candidate: Qdrant collection name for candidate chunks.
        qdrant_collection_jd: Qdrant collection name for JD chunks.
        host: API server bind host.
        port: API server bind port.
    """

    groq_api_key: str = ""
    tavily_api_key: str = ""

    llm_model: str = "groq/llama-3.3-70b-versatile"

    mongodb_uri: str = "mongodb://localhost:27017/talentmatch"
    qdrant_url: str = "http://localhost:6333"
    qdrant_mode: str = "local"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 100
    embedding_dimension: int = 384

    qdrant_collection_candidate: str = "candidate_chunks"
    qdrant_collection_jd: str = "jd_chunks"

    host: str = "0.0.0.0"
    port: int = 8000

    email_mode: str = "dry_run"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
