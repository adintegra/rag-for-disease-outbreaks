import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Engine, exists, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, ChunkDataset, Document, Embedding, EmbeddingRun
from app.embedding.model import EmbeddingIdentity
from app.ingestion.chunk_profiles import ChunkProfile
from app.ingestion.staging import ConcurrentIngestionError


@dataclass(frozen=True)
class EmbeddingCandidate:
  chunk_id: int
  content_hash: str
  contents: str


class EmbeddingRepository:
  def __init__(self, engine: Engine, session_factory: sessionmaker[Session]):
    self.engine = engine
    self.session_factory = session_factory

  @contextmanager
  def profile_lock(
    self, profile: ChunkProfile, identity: EmbeddingIdentity
  ) -> Iterator[None]:
    key = f"embed:{profile.configuration_hash}:{identity.model}:{identity.model_version}"
    with self.engine.connect() as connection:
      locked = connection.execute(
        text("SELECT pg_try_advisory_lock(hashtext(:key))"), {"key": key}
      ).scalar_one()
      connection.commit()
      if not locked:
        raise ConcurrentIngestionError(
          f"Another embedding run is active for profile {profile.name}"
        )
      try:
        yield
      finally:
        connection.execute(
          text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": key}
        )
        connection.commit()

  def candidates(
    self,
    profile: ChunkProfile,
    identity: EmbeddingIdentity,
    limit: int | None,
  ) -> list[EmbeddingCandidate]:
    current_embedding = exists().where(
      Embedding.chunk_id == Chunk.id,
      Embedding.model == identity.model,
      Embedding.model_version == identity.model_version,
      Embedding.chunk_hash == Chunk.content_hash,
    )
    statement = (
      select(Chunk.id, Chunk.content_hash, Chunk.contents)
      .join(ChunkDataset, ChunkDataset.id == Chunk.chunk_dataset_id)
      .join(Document, Document.id == ChunkDataset.document_id)
      .where(
        ChunkDataset.configuration_hash == profile.configuration_hash,
        ChunkDataset.status == "complete",
        ChunkDataset.document_hash == Document.content_hash,
        ~current_embedding,
      )
      .order_by(Chunk.id)
    )
    if limit is not None:
      statement = statement.limit(limit)
    with self.session_factory() as session:
      return [
        EmbeddingCandidate(row.id, row.content_hash, row.contents)
        for row in session.execute(statement)
      ]

  def create_run(
    self,
    profile: ChunkProfile,
    identity: EmbeddingIdentity,
    selected_count: int,
    batch_size: int,
    requested_limit: int | None,
  ) -> uuid.UUID:
    run_id = uuid.uuid4()
    with self.session_factory.begin() as session:
      session.add(
        EmbeddingRun(
          id=run_id,
          profile_name=profile.name,
          configuration_hash=profile.configuration_hash,
          model=identity.model,
          model_version=identity.model_version,
          dimensions=identity.dimensions,
          status="running",
          selected_count=selected_count,
          batch_size=batch_size,
          requested_limit=requested_limit,
        )
      )
    return run_id

  def store_batch(
    self,
    run_id: uuid.UUID,
    identity: EmbeddingIdentity,
    candidates: list[EmbeddingCandidate],
    vectors: list[list[float]],
  ) -> None:
    values = [
      {
        "chunk_id": candidate.chunk_id,
        "model": identity.model,
        "model_version": identity.model_version,
        "dimensions": identity.dimensions,
        "chunk_hash": candidate.content_hash,
        "embedding": vector,
      }
      for candidate, vector in zip(candidates, vectors, strict=True)
    ]
    with self.session_factory.begin() as session:
      statement = insert(Embedding).values(values)
      statement = statement.on_conflict_do_update(
        constraint="uq_embedding_model",
        set_={
          "dimensions": statement.excluded.dimensions,
          "chunk_hash": statement.excluded.chunk_hash,
          "embedding": statement.excluded.embedding,
          "created_at": datetime.now(UTC),
        },
      )
      session.execute(statement)
      run = session.get(EmbeddingRun, run_id)
      if run is None:
        raise ValueError(f"Unknown embedding run: {run_id}")
      run.embedded_count += len(values)

  def finish_run(self, run_id: uuid.UUID, elapsed_seconds: float) -> None:
    with self.session_factory.begin() as session:
      run = session.get(EmbeddingRun, run_id)
      if run is None:
        raise ValueError(f"Unknown embedding run: {run_id}")
      run.status = "succeeded"
      run.finished_at = datetime.now(UTC)
      run.elapsed_seconds = elapsed_seconds

  def fail_run(
    self, run_id: uuid.UUID, error: Exception, elapsed_seconds: float
  ) -> None:
    with self.session_factory.begin() as session:
      run = session.get(EmbeddingRun, run_id)
      if run is None:
        return
      run.status = "failed"
      run.failed_count = max(run.selected_count - run.embedded_count, 1)
      run.error_summary = str(error)[:2000]
      run.finished_at = datetime.now(UTC)
      run.elapsed_seconds = elapsed_seconds
