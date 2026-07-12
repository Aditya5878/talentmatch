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
