import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from sqlalchemy import Engine, text

from app.embedding.pipeline import EmbeddingPipeline, EmbeddingResult
from app.ingestion.chunk_datasets import ChunkDatasetRepository, ChunkDatasetResult
from app.ingestion.chunk_profiles import get_chunk_profile
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.staging import (
  ConcurrentIngestionError,
  StageResult,
  StagingRepository,
)
from app.ingestion.transformation import TransformationRepository, TransformResult


@dataclass(frozen=True)
class ProfileRunResult:
  chunking: ChunkDatasetResult
  embedding: EmbeddingResult | None


@dataclass(frozen=True)
class OrchestrationResult:
  run_id: uuid.UUID
  extraction: StageResult
  transformation: TransformResult
  profiles: list[ProfileRunResult]


class PipelineLock:
  """Hold a session-level PostgreSQL lock for the complete scheduled run."""

  def __init__(self, engine: Engine, source: str):
    self.engine = engine
    self.key = f"pipeline:{source}"

  @contextmanager
  def acquire(self) -> Iterator[None]:
    with self.engine.connect() as connection:
      locked = connection.execute(
        text("SELECT pg_try_advisory_lock(hashtext(:key))"), {"key": self.key}
      ).scalar_one()
      connection.commit()
      if not locked:
        raise ConcurrentIngestionError("Another complete ingestion run is active")
      try:
        yield
      finally:
        connection.execute(
          text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": self.key}
        )
        connection.commit()


class OrchestrationPipeline:
  """Run acquisition through incremental embeddings behind one interface."""

  def __init__(
    self,
    lock: PipelineLock,
    extraction: IngestionPipeline,
    transformation: TransformationRepository,
    chunking: ChunkDatasetRepository,
    embedding: EmbeddingPipeline,
    run_tracker: StagingRepository,
  ):
    self.lock = lock
    self.extraction = extraction
    self.transformation = transformation
    self.chunking = chunking
    self.embedding = embedding
    self.run_tracker = run_tracker

  def run(
    self,
    profile_names: list[str],
    *,
    embedding_batch_size: int,
    skip_embeddings: bool = False,
  ) -> OrchestrationResult:
    if not profile_names:
      raise ValueError("At least one chunk profile is required")

    run_id: uuid.UUID | None = None
    with self.lock.acquire():
      try:
        extraction = self.extraction.extract()
        run_id = extraction.run_id
        transformation = self.transformation.transform(run_id)
        profile_results = []
        for name in profile_names:
          profile = get_chunk_profile(name)
          chunking = self.chunking.generate(profile)
          embedding = (
            None
            if skip_embeddings
            else self.embedding.run(profile, batch_size=embedding_batch_size)
          )
          profile_results.append(ProfileRunResult(chunking, embedding))

        result = OrchestrationResult(
          run_id, extraction, transformation, profile_results
        )
        self.run_tracker.complete_run(run_id, _metadata(result, skip_embeddings))
        return result
      except Exception as error:
        if run_id is not None:
          self.run_tracker.fail_run(run_id, error)
        raise


def _metadata(result: OrchestrationResult, skip_embeddings: bool) -> dict:
  return {
    "pipeline": "who_don_rag",
    "skip_embeddings": skip_embeddings,
    "extraction": _jsonable(asdict(result.extraction)),
    "transformation": _jsonable(asdict(result.transformation)),
    "profiles": [
      {
        "chunking": _jsonable(asdict(profile.chunking)),
        "embedding": (
          _jsonable(asdict(profile.embedding)) if profile.embedding else None
        ),
      }
      for profile in result.profiles
    ],
  }


def _jsonable(value):
  if isinstance(value, dict):
    return {key: _jsonable(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_jsonable(item) for item in value]
  if isinstance(value, uuid.UUID):
    return str(value)
  return value
