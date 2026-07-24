from talentmatch.config import settings
from talentmatch.utils.llm import llm_embedding


def _is_remote_model() -> bool:
    return "/" in settings.embedding_model


def _get_local_model():
    from sentence_transformers import SentenceTransformer

    if not hasattr(_get_local_model, "_model"):
        _get_local_model._model = SentenceTransformer(settings.embedding_model)
    return _get_local_model._model


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if _is_remote_model():
        return await _remote_embed(texts)
    return _local_embed(texts)


async def embed_text(text: str) -> list[float]:
    result = await embed_texts([text])
    return result[0]


def _local_embed(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


async def _remote_embed(texts: list[str]) -> list[list[float]]:
    results: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await llm_embedding(model=settings.embedding_model, input=batch)
        for item in response.data:
            results.append(item["embedding"])
    return results
