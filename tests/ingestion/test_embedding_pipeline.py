import uuid
from contextlib import contextmanager

import pytest

from app.embedding.model import EmbeddingIdentity, EmbeddingPreflight
from app.embedding.pipeline import EmbeddingPipeline
from app.embedding.repository import EmbeddingCandidate
from app.embedding.tracing import EmbeddingTracer
from app.ingestion.chunk_profiles import ChunkProfile


class FakeModel:
  identity = EmbeddingIdentity("model", "v1", 2)

  def __init__(self):
    self.calls = []

  def preflight(self):
    return EmbeddingPreflight(self.identity, 0.01)

  def embed_documents(self, texts):
    self.calls.append(texts)
    return [[0.1, 0.2] for _ in texts]


class FakeRepository:
  def __init__(self, candidates):
    self.available = list(candidates)
    self.run_id = uuid.uuid4()
    self.stored = []
    self.finished = False
    self.failed = None

  @contextmanager
  def profile_lock(self, profile, identity):
    yield

  def candidates(self, profile, identity, limit):
    remaining = [candidate for candidate in self.available if candidate not in self.stored]
    return remaining[:limit] if limit is not None else remaining

  def create_run(self, profile, identity, selected_count, batch_size, requested_limit):
    self.selected_count = selected_count
    return self.run_id

  def store_batch(self, run_id, identity, candidates, vectors):
    self.stored.extend(candidates)

  def finish_run(self, run_id, elapsed):
    self.finished = True

  def fail_run(self, run_id, error, elapsed):
    self.failed = error


def profile():
  return ChunkProfile("sections", "who_sections", "1", 500, 50)


def candidates(count):
  return [
    EmbeddingCandidate(index, f"hash-{index}", f"text-{index}")
    for index in range(count)
  ]


def test_pipeline_batches_and_persists_embeddings() -> None:
  model = FakeModel()
  repository = FakeRepository(candidates(5))
  pipeline = EmbeddingPipeline(
    model, repository, EmbeddingTracer(enabled=False, project_name="test", trace_content=False)
  )

  result = pipeline.run(profile(), batch_size=2)

  assert [len(call) for call in model.calls] == [2, 2, 1]
  assert result.embedded == 5
  assert repository.finished
  assert repository.failed is None


def test_pipeline_limit_is_maximum_new_chunks() -> None:
  model = FakeModel()
  repository = FakeRepository(candidates(5))
  pipeline = EmbeddingPipeline(
    model, repository, EmbeddingTracer(enabled=False, project_name="test", trace_content=False)
  )

  result = pipeline.run(profile(), batch_size=2, limit=3)

  assert result.selected == 3
  assert len(repository.stored) == 3
  assert len(repository.candidates(profile(), model.identity, None)) == 2


def test_pipeline_rejects_invalid_limits() -> None:
  pipeline = EmbeddingPipeline(
    FakeModel(),
    FakeRepository([]),
    EmbeddingTracer(enabled=False, project_name="test", trace_content=False),
  )

  with pytest.raises(ValueError, match="limit"):
    pipeline.run(profile(), batch_size=2, limit=0)
