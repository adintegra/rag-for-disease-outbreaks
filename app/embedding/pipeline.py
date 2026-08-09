import uuid
from dataclasses import dataclass
from time import perf_counter

from app.embedding.model import EmbeddingPreflight, LocalEmbeddingModel
from app.embedding.repository import EmbeddingCandidate, EmbeddingRepository
from app.embedding.tracing import EmbeddingTracer
from app.ingestion.chunk_profiles import ChunkProfile


@dataclass(frozen=True)
class EmbeddingResult:
  run_id: uuid.UUID
  profile_name: str
  selected: int
  embedded: int
  elapsed_seconds: float
  chunks_per_second: float
  estimated_remaining_seconds: float | None


class EmbeddingPipeline:
  def __init__(
    self,
    model: LocalEmbeddingModel,
    repository: EmbeddingRepository,
    tracer: EmbeddingTracer,
  ):
    self.model = model
    self.repository = repository
    self.tracer = tracer

  def preflight(self) -> EmbeddingPreflight:
    return self.model.preflight()

  def run(
    self,
    profile: ChunkProfile,
    *,
    batch_size: int,
    limit: int | None = None,
  ) -> EmbeddingResult:
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
      raise ValueError("limit must be positive")

    started = perf_counter()
    with self.repository.profile_lock(profile, self.model.identity):
      self.model.preflight()
      candidates = self.repository.candidates(profile, self.model.identity, limit)
      run_id = self.repository.create_run(
        profile,
        self.model.identity,
        len(candidates),
        batch_size,
        limit,
      )
      try:
        for batch_number, batch in enumerate(_batches(candidates, batch_size), start=1):
          texts = [candidate.contents for candidate in batch]
          vectors = self.tracer.invoke_batch(
            lambda batch_texts=texts: self.model.embed_documents(batch_texts),
            texts=texts,
            metadata={
              "run_id": str(run_id),
              "profile": profile.name,
              "configuration_hash": profile.configuration_hash,
              "model": self.model.identity.model,
              "model_version": self.model.identity.model_version,
              "batch": batch_number,
              "chunk_ids": [candidate.chunk_id for candidate in batch],
              "chunk_hashes": [candidate.content_hash for candidate in batch],
            },
          )
          self.repository.store_batch(run_id, self.model.identity, batch, vectors)
      except Exception as error:
        self.repository.fail_run(run_id, error, perf_counter() - started)
        raise

      elapsed = perf_counter() - started
      self.repository.finish_run(run_id, elapsed)

    rate = len(candidates) / elapsed if elapsed and candidates else 0.0
    remaining = len(
      self.repository.candidates(profile, self.model.identity, limit=None)
    )
    estimate = remaining / rate if rate else None
    return EmbeddingResult(
      run_id,
      profile.name,
      len(candidates),
      len(candidates),
      elapsed,
      rate,
      estimate,
    )


def _batches(
  candidates: list[EmbeddingCandidate], size: int
) -> list[list[EmbeddingCandidate]]:
  return [candidates[index : index + size] for index in range(0, len(candidates), size)]
