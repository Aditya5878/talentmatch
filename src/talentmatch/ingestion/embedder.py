from talentmatch.config import settings
from talentmatch.utils.llm import llm_embedding


def _is_remote_model() -> bool:
    """Check if the configured embedding model is a remote API model.

    Remote models contain a '/' in their name (e.g. "openai/text-embedding-3-small").
    Local models are sentence-transformers names without '/' (e.g. "all-MiniLM-L6-v2").

    Returns:
        True if the model is remote, False if local.
    """
    return "/" in settings.embedding_model


def _get_local_model():
    """Load and cache the local sentence-transformers embedding model.

    The model is loaded once on first call and cached for subsequent calls.

    Returns:
        The SentenceTransformer model instance.
    """
    from sentence_transformers import SentenceTransformer

    if not hasattr(_get_local_model, "_model"):
        _get_local_model._model = SentenceTransformer(settings.embedding_model)
    return _get_local_model._model


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into vectors.

    Automatically delegates to local (sentence-transformers) or remote
    (LiteLLM API) embedding based on the configured model.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors, one per input text.
    """
    if _is_remote_model():
        return await _remote_embed(texts)
    return _local_embed(texts)


async def embed_text(text: str) -> list[float]:
    """Embed a single text into a vector.

    Convenience wrapper around embed_texts() for single-text embedding.

    Args:
        text: The text string to embed.

    Returns:
        The embedding vector as a list of floats.
    """
    result = await embed_texts([text])
    return result[0]


def _local_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using the local sentence-transformers model.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    model = _get_local_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


async def _remote_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts via a remote embedding API (LiteLLM).

    Processes texts in batches of EMBEDDING_BATCH_SIZE to respect
    API rate limits and payload size constraints.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    results: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await llm_embedding(model=settings.embedding_model, input=batch)
        for item in response.data:
            results.append(item["embedding"])
    return results
