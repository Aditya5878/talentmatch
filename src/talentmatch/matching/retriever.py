from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from talentmatch.config import settings
from talentmatch.db.qdrant import get_qdrant_client
from talentmatch.ingestion.embedder import embed_text


async def retrieve_candidates(
    skills_text: str,
    experience_texts: list[str],
    top_k_per_section: int = 10,
) -> list[dict]:
    """Retrieve top candidate chunks from Qdrant matching the given skills and experience.

    Embeds each section separately, searches Qdrant for top-N matches per section,
    then aggregates scores across sections using max-pooling per entity.

    Args:
        skills_text: Comma-separated skills to search for.
        experience_texts: List of experience descriptions to search against.
        top_k_per_section: Number of top results to retrieve per section.

    Returns:
        List of dicts sorted by score descending, each with 'entity_id',
        'score', and 'payload' keys.
    """
    return await _retrieve(
        collection=settings.qdrant_collection_candidate,
        skills_text=skills_text,
        experience_texts=experience_texts,
        top_k_per_section=top_k_per_section,
    )


async def retrieve_jds(
    skills_text: str,
    experience_texts: list[str],
    top_k_per_section: int = 10,
) -> list[dict]:
    """Retrieve top JD chunks from Qdrant matching the given skills and experience.

    Same retrieval logic as retrieve_candidates but against the JD collection.

    Args:
        skills_text: Comma-separated skills to search for.
        experience_texts: List of experience descriptions to search against.
        top_k_per_section: Number of top results to retrieve per section.

    Returns:
        List of dicts sorted by score descending, each with 'entity_id',
        'score', and 'payload' keys.
    """
    return await _retrieve(
        collection=settings.qdrant_collection_jd,
        skills_text=skills_text,
        experience_texts=experience_texts,
        top_k_per_section=top_k_per_section,
    )


async def _retrieve(
    collection: str,
    skills_text: str,
    experience_texts: list[str],
    top_k_per_section: int,
) -> list[dict]:
    """Core retrieval logic: embed sections, search Qdrant, aggregate scores.

    For each section (skills, experience entries), embeds the text and searches
    Qdrant for similar chunks. Aggregates per-entity scores using max-pooling
    across all sections that matched that entity.

    Args:
        collection: Qdrant collection name to search.
        skills_text: Comma-separated skills text.
        experience_texts: List of experience description texts.
        top_k_per_section: Top-N results per section search.

    Returns:
        List of aggregated results sorted by score descending.
    """
    client: QdrantClient = get_qdrant_client()

    section_texts = [("skills", skills_text)] if skills_text else []
    for et in experience_texts:
        section_texts.append(("experience", et))

    scores: dict[str, list[float]] = defaultdict(list)
    entities: dict[str, dict] = {}

    for section, text in section_texts:
        if not text.strip():
            continue
        vector = await embed_text(text)
        results = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=top_k_per_section,
        )
        for res in results:
            eid = res.payload.get("entity_id", "")
            scores[eid].append(res.score)
            if eid not in entities:
                entities[eid] = res.payload or {}

    aggregated = []
    for eid, score_list in scores.items():
        aggregated.append({
            "entity_id": eid,
            "score": max(score_list),
            "payload": entities.get(eid, {}),
        })

    aggregated.sort(key=lambda x: x["score"], reverse=True)
    return aggregated


async def hybrid_search_candidates(
    query_text: str,
    skills_filter: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Search for candidates using vector similarity with optional skills filter.

    Used for free-text keyword/skill searches (no JD or resume as input).
    Embeds the query and searches Qdrant, optionally filtering to skills
    section chunks only.

    Args:
        query_text: The search query (may be expanded with related terms).
        skills_filter: Optional list of skills to filter results by.
        top_k: Maximum number of results to return.

    Returns:
        List of matching entities sorted by score descending.
    """
    return await _hybrid_search(
        collection=settings.qdrant_collection_candidate,
        query_text=query_text,
        skills_filter=skills_filter,
        top_k=top_k,
    )


async def hybrid_search_jds(
    query_text: str,
    skills_filter: list[str] | None = None,
    top_k: int = 20,
) -> list[dict]:
    """Search for JDs using vector similarity with optional skills filter.

    Same search logic as hybrid_search_candidates but against the JD collection.

    Args:
        query_text: The search query (may be expanded with related terms).
        skills_filter: Optional list of skills to filter results by.
        top_k: Maximum number of results to return.

    Returns:
        List of matching entities sorted by score descending.
    """
    return await _hybrid_search(
        collection=settings.qdrant_collection_jd,
        query_text=query_text,
        skills_filter=skills_filter,
        top_k=top_k,
    )


async def _hybrid_search(
    collection: str,
    query_text: str,
    skills_filter: list[str] | None,
    top_k: int,
) -> list[dict]:
    """Core hybrid search: vector similarity with optional payload filtering.

    Embeds the query text and searches Qdrant. If skills_filter is provided,
    filters to only chunks in the 'skills' section. Deduplicates results
    by entity_id, keeping the highest score per entity.

    Args:
        collection: Qdrant collection name to search.
        query_text: The search query text.
        skills_filter: If provided, filter to skills section chunks.
        top_k: Maximum number of results.

    Returns:
        Deduplicated results sorted by score descending.
    """
    client: QdrantClient = get_qdrant_client()
    vector = await embed_text(query_text)

    query_filter: Filter | None = None
    if skills_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchValue(value="skills"),
                ),
            ],
        )

    results = client.search(
        collection_name=collection,
        query_vector=vector,
        query_filter=query_filter,
        limit=top_k,
    )

    seen: dict[str, dict] = {}
    for res in results:
        eid = res.payload.get("entity_id", "")
        if eid not in seen or res.score > seen[eid]["score"]:
            seen[eid] = {
                "entity_id": eid,
                "score": res.score,
                "payload": res.payload or {},
            }

    aggregated = list(seen.values())
    aggregated.sort(key=lambda x: x["score"], reverse=True)
    return aggregated
