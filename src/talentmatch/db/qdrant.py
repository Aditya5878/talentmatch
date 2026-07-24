from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from talentmatch.config import settings


def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_mode == "local":
        return QdrantClient(path="./qdrant_data")
    return QdrantClient(url=settings.qdrant_url)


def ensure_collections(client: QdrantClient) -> None:
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
