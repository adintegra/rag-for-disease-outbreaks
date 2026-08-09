import uuid
from contextlib import contextmanager

import pytest

from app.embedding.pipeline import EmbeddingResult
from app.ingestion.chunk_datasets import ChunkDatasetResult
from app.ingestion.orchestration import OrchestrationPipeline
from app.ingestion.staging import StageResult
from app.ingestion.transformation import TransformResult


class FakeLock:
  def __init__(self):
    self.acquired = False

  @contextmanager
  def acquire(self):
    self.acquired = True
    yield


class FakeExtraction:
  def __init__(self, run_id):
    self.run_id = run_id

  def extract(self):
    return StageResult(self.run_id, 3, 3, 0, 0, 0)


class FakeTransformation:
  def transform(self, run_id):
    return TransformResult(run_id, 3, 0, 0, 3, 0)


class FakeChunking:
  def __init__(self, error=None):
    self.error = error
    self.profiles = []

  def generate(self, profile):
    if self.error:
      raise self.error
    self.profiles.append(profile.name)
    return ChunkDatasetResult(profile.name, 3, 0, 3, 0, 0)


class FakeEmbedding:
  def __init__(self):
    self.profiles = []

  def run(self, profile, batch_size):
    self.profiles.append((profile.name, batch_size))
    return EmbeddingResult(uuid.uuid4(), profile.name, 0, 0, 0.1, 0, None)


class FakeTracker:
  def __init__(self):
    self.completed = None
    self.failed = None

  def complete_run(self, run_id, metadata):
    self.completed = (run_id, metadata)

  def fail_run(self, run_id, error):
    self.failed = (run_id, error)


def test_orchestration_runs_all_stages_and_records_completion() -> None:
  run_id = uuid.uuid4()
  lock = FakeLock()
  chunks = FakeChunking()
  embeddings = FakeEmbedding()
  tracker = FakeTracker()
  pipeline = OrchestrationPipeline(
    lock,
    FakeExtraction(run_id),
    FakeTransformation(),
    chunks,
    embeddings,
    tracker,
  )

  result = pipeline.run(["who-sections-1200"], embedding_batch_size=16)

  assert lock.acquired
  assert result.run_id == run_id
  assert chunks.profiles == ["who-sections-1200"]
  assert embeddings.profiles == [("who-sections-1200", 16)]
  assert tracker.completed[0] == run_id
  assert tracker.completed[1]["pipeline"] == "who_don_rag"
  assert tracker.failed is None


def test_orchestration_can_skip_local_embedding_endpoint() -> None:
  run_id = uuid.uuid4()
  embeddings = FakeEmbedding()
  tracker = FakeTracker()
  pipeline = OrchestrationPipeline(
    FakeLock(),
    FakeExtraction(run_id),
    FakeTransformation(),
    FakeChunking(),
    embeddings,
    tracker,
  )

  result = pipeline.run(
    ["who-sections-1200"], embedding_batch_size=16, skip_embeddings=True
  )

  assert embeddings.profiles == []
  assert result.profiles[0].embedding is None
  assert tracker.completed[1]["skip_embeddings"] is True


def test_orchestration_marks_post_extract_failure() -> None:
  run_id = uuid.uuid4()
  failure = RuntimeError("chunking failed")
  tracker = FakeTracker()
  pipeline = OrchestrationPipeline(
    FakeLock(),
    FakeExtraction(run_id),
    FakeTransformation(),
    FakeChunking(error=failure),
    FakeEmbedding(),
    tracker,
  )

  with pytest.raises(RuntimeError, match="chunking failed"):
    pipeline.run(["who-sections-1200"], embedding_batch_size=16)

  assert tracker.failed == (run_id, failure)
  assert tracker.completed is None


def test_orchestration_requires_a_profile() -> None:
  pipeline = OrchestrationPipeline(
    FakeLock(), FakeExtraction(uuid.uuid4()), FakeTransformation(), FakeChunking(), FakeEmbedding(), FakeTracker()
  )

  with pytest.raises(ValueError, match="profile"):
    pipeline.run([], embedding_batch_size=16)
