from dataclasses import replace

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, ChunkDataset, Document, Embedding
from app.retrieval.models import RetrievedChunk, SearchFilters
from app.retrieval.profiles import RetrievalProfile


class RetrievalRepository:
  """Perform profile-scoped cosine search in Neon."""

  def __init__(self, session_factory: sessionmaker[Session]):
    self.session_factory = session_factory

  def search(
    self,
    query_vector: list[float],
    profile: RetrievalProfile,
    filters: SearchFilters,
    candidate_limit: int,
  ) -> list[RetrievedChunk]:
    distance = Embedding.embedding.cosine_distance(query_vector)
    statement = (
      select(
        Chunk.id.label("chunk_id"),
        Document.id.label("document_id"),
        Document.title,
        Chunk.section,
        Chunk.contents,
        Document.url,
        Document.published_at,
        distance.label("distance"),
      )
      .join(ChunkDataset, ChunkDataset.id == Chunk.chunk_dataset_id)
      .join(Document, Document.id == ChunkDataset.document_id)
      .join(Embedding, Embedding.chunk_id == Chunk.id)
      .where(
        ChunkDataset.configuration_hash
        == profile.chunk_profile.configuration_hash,
        ChunkDataset.status == "complete",
        ChunkDataset.document_hash == Document.content_hash,
        Embedding.model == profile.embedding_model,
        Embedding.model_version == profile.embedding_model_version,
        Embedding.chunk_hash == Chunk.content_hash,
      )
      .order_by(distance)
      .limit(candidate_limit)
    )
    if filters.published_after is not None:
      statement = statement.where(Document.published_at >= filters.published_after)
    if filters.published_before is not None:
      statement = statement.where(Document.published_at < filters.published_before)
    if filters.sections:
      statement = statement.where(Chunk.section.in_(filters.sections))
    if filters.source:
      statement = statement.where(Document.source == filters.source)

    with self.session_factory() as session:
      session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
      rows = session.execute(statement).all()

    return [
      RetrievedChunk(
        rank=index,
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        title=row.title,
        section=row.section,
        contents=row.contents,
        url=row.url,
        published_at=row.published_at,
        distance=float(row.distance),
      )
      for index, row in enumerate(rows, start=1)
    ]


def diversify(
  candidates: list[RetrievedChunk],
  *,
  limit: int,
  max_chunks_per_document: int,
) -> list[RetrievedChunk]:
  selected = []
  document_counts: dict[int, int] = {}
  for candidate in candidates:
    count = document_counts.get(candidate.document_id, 0)
    if count >= max_chunks_per_document:
      continue
    selected.append(replace(candidate, rank=len(selected) + 1))
    document_counts[candidate.document_id] = count + 1
    if len(selected) == limit:
      break
  return selected
