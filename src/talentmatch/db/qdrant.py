from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from talentmatch.config import settings

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client instance.

    In local mode, creates an in-process client with storage at ./qdrant_data.
    In remote mode, connects to the configured Qdrant server URL.
    The client is created on first call and reused for all subsequent calls
    to avoid local storage lock conflicts.

    Returns:
        QdrantClient: The shared Qdrant client instance.
    """
    global _client
    if _client is not None:
        return _client
    if settings.qdrant_mode == "local":
        _client = QdrantClient(path="./qdrant_data")
    else:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collections(client: QdrantClient) -> None:
    """Create Qdrant collections if they don't already exist.

    Ensures both candidate_chunks and jd_chunks collections exist with
    the configured embedding dimension and COSINE distance metric.

    Args:
        client: The Qdrant client to use for collection operations.
    """
    for collection_name in [
        settings.qdrant_collection_candidate,
        settings.qdrant_collection_jd,
    ]:
        try:
            client.get_collection(collection_name)
        except (UnexpectedResponse, ValueError):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
