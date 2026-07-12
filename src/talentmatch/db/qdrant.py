from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from talentmatch.config import settings


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


async def ensure_collections(client: QdrantClient) -> None:
    collections = [
        (settings.qdrant_collection_candidate, "candidate_chunks"),
        (settings.qdrant_collection_jd, "jd_chunks"),
    ]
    for collection_name, _ in collections:
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
