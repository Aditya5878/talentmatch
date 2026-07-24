import os
import tempfile
import logging
from datetime import datetime

from bson import ObjectId

from talentmatch.config import settings
from talentmatch.db.qdrant import get_qdrant_client
from talentmatch.ingestion import embedder as embedder_mod
from talentmatch.ingestion.chunker import chunk_jd, chunk_resume
from talentmatch.ingestion.extractor import compute_file_hash, extract_text
from talentmatch.ingestion.parser import parse_jd, parse_resume
from talentmatch.models import BatchJob, Candidate, EmbeddingIndex, JD
from talentmatch.models.batch_job import BatchItem
from talentmatch.models.enums import BatchStatus, EntityType, ItemStatus
from talentmatch.utils.logging import get_trace_id
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

logger = logging.getLogger("talentmatch.worker")


async def process_batch(batch: BatchJob, files: list[tuple[str, bytes, str]]) -> None:
    batch.status = BatchStatus.processing
    batch.updated_at = datetime.utcnow()
    await batch.save()
    logger.info("Batch %s processing started, %d files", batch.id, len(files), extra={"trace_id": get_trace_id()})

    qdrant: QdrantClient = get_qdrant_client()

    candidate_points: list[PointStruct] = []
    jd_points: list[PointStruct] = []
    all_embeddings_index: list[EmbeddingIndex] = []
    all_candidates: list[Candidate] = []
    all_jds: list[JD] = []

    try:
        for idx, (filename, file_bytes, file_type) in enumerate(files):
            item = batch.items[idx] if idx < len(batch.items) else _make_item(filename, file_type)
            try:
                await _process_file(
                    filename=filename,
                    file_bytes=file_bytes,
                    file_type=EntityType(file_type),
                    item=item,
                    candidate_points=candidate_points,
                    jd_points=jd_points,
                    all_embeddings_index=all_embeddings_index,
                    all_candidates=all_candidates,
                    all_jds=all_jds,
                    qdrant=qdrant,
                )
                batch.completed_items += 1
                logger.info("File %s processed (%s)", filename, file_type, extra={"trace_id": get_trace_id()})
            except Exception as exc:
                item.status = ItemStatus.failed
                item.error = str(exc)
                logger.error("File %s failed: %s", filename, exc, extra={"trace_id": get_trace_id()})
            batch.items[idx] = item
            batch.updated_at = datetime.utcnow()
            await batch.save()

        if all_candidates:
            await Candidate.insert_many(all_candidates)
        if all_jds:
            await JD.insert_many(all_jds)
        if all_embeddings_index:
            await EmbeddingIndex.insert_many(all_embeddings_index)
        if candidate_points:
            qdrant.upsert(collection_name=settings.qdrant_collection_candidate, points=candidate_points)
        if jd_points:
            qdrant.upsert(collection_name=settings.qdrant_collection_jd, points=jd_points)

        batch.status = BatchStatus.completed
        logger.info("Batch %s completed", batch.id, extra={"trace_id": get_trace_id()})
    except Exception as exc:
        batch.status = BatchStatus.failed
        logger.error("Batch %s failed: %s", batch.id, exc, extra={"trace_id": get_trace_id()})
        for item in batch.items:
            if item.status not in (ItemStatus.persisted, ItemStatus.failed):
                item.status = ItemStatus.failed
                item.error = str(exc)
    finally:
        batch.updated_at = datetime.utcnow()
        await batch.save()


def _make_item(filename: str, file_type: str) -> BatchItem:
    return BatchItem(
        filename=filename,
        file_type=EntityType(file_type),
        status=ItemStatus.queued,
    )


async def _process_file(
    filename: str,
    file_bytes: bytes,
    file_type: EntityType,
    item: BatchItem,
    candidate_points: list[PointStruct],
    jd_points: list[PointStruct],
    all_embeddings_index: list[EmbeddingIndex],
    all_candidates: list[Candidate],
    all_jds: list[JD],
    qdrant: QdrantClient,
) -> None:
    file_hash = compute_file_hash(file_bytes)
    item.file_hash = file_hash

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    try:
        tmp.write(file_bytes)
        tmp.flush()
        path = tmp.name
    finally:
        tmp.close()

    try:
        raw_text = extract_text(path)
        item.status = ItemStatus.extracted

        entity_id = str(ObjectId())

        if file_type == EntityType.candidate:
            parsed = await parse_resume(raw_text)
            await _cleanup_existing(qdrant, file_type, file_hash)

            doc = Candidate(
                name=parsed.get("name", ""),
                email=parsed.get("email", ""),
                resume_raw_text=raw_text,
                resume_file_path=filename,
                parsed_json=parsed,
                file_hash=file_hash,
            )
            doc.id = ObjectId(entity_id)
            all_candidates.append(doc)
            chunks = chunk_resume(parsed)
            coll = settings.qdrant_collection_candidate
            points = candidate_points
        else:
            parsed = await parse_jd(raw_text)
            await _cleanup_existing(qdrant, file_type, file_hash)

            doc = JD(
                title=parsed.get("title", ""),
                company=parsed.get("company", ""),
                jd_raw_text=raw_text,
                jd_file_path=filename,
                parsed_json=parsed,
                file_hash=file_hash,
            )
            doc.id = ObjectId(entity_id)
            all_jds.append(doc)
            chunks = chunk_jd(parsed)
            coll = settings.qdrant_collection_jd
            points = jd_points

        item.status = ItemStatus.parsed

        if not chunks:
            item.status = ItemStatus.persisted
            return

        chunk_texts = [c["text"] for c in chunks]
        vectors = await embedder_mod.embed_texts(chunk_texts)
        item.status = ItemStatus.embedded

        for ci, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = f"{entity_id}_{ci}"
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "entity_id": entity_id,
                        "section": chunk["section"],
                        "text": chunk["text"],
                    },
                )
            )
            all_embeddings_index.append(
                EmbeddingIndex(
                    entity_type=file_type,
                    entity_id=entity_id,
                    qdrant_point_id=point_id,
                    chunk_text=chunk["text"],
                )
            )

        item.status = ItemStatus.persisted
    finally:
        os.unlink(path)


async def _cleanup_existing(qdrant: QdrantClient, file_type: EntityType, file_hash: str) -> None:
    collection = settings.qdrant_collection_candidate if file_type == EntityType.candidate else settings.qdrant_collection_jd
    model = Candidate if file_type == EntityType.candidate else JD

    existing = await model.find(model.file_hash == file_hash).to_list()
    for doc in existing:
        embeddings = await EmbeddingIndex.find(
            EmbeddingIndex.entity_type == file_type,
            EmbeddingIndex.entity_id == str(doc.id),
        ).to_list()

        if embeddings:
            point_ids = [e.qdrant_point_id for e in embeddings]
            try:
                qdrant.delete(
                    collection_name=collection,
                    points_selector=point_ids,
                )
            except Exception:
                pass
            await EmbeddingIndex.find(
                EmbeddingIndex.entity_id == str(doc.id),
                EmbeddingIndex.entity_type == file_type,
            ).delete()

        await doc.delete()
