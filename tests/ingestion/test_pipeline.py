import uuid

import pytest

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.staging import StageResult


class SourceStub:
  def __init__(self, records=None, error: Exception | None = None):
    self.records = records or []
    self.error = error

  def iter_records(self):
    if self.error:
      raise self.error
    yield from self.records


class StagingSpy:
  def __init__(self):
    self.run_id = uuid.uuid4()
    self.failed: tuple[uuid.UUID, Exception] | None = None

  def create_run(self, run_id=None):
    return run_id or self.run_id

  def stage(self, run_id, records):
    fetched = len(list(records))
    return StageResult(run_id, fetched, fetched, 0, 0, 0)

  def fail_run(self, run_id, error):
    self.failed = (run_id, error)


def test_extract_coordinates_source_and_staging() -> None:
  staging = StagingSpy()
  pipeline = IngestionPipeline(SourceStub([{"UrlName": "don-1"}]), staging)

  result = pipeline.extract()

  assert result.run_id == staging.run_id
  assert result.fetched == 1
  assert staging.failed is None


def test_extract_marks_run_failed_before_reraising() -> None:
  staging = StagingSpy()
  failure = RuntimeError("WHO unavailable")
  pipeline = IngestionPipeline(SourceStub(error=failure), staging)

  with pytest.raises(RuntimeError, match="WHO unavailable"):
    pipeline.extract()

  assert staging.failed == (staging.run_id, failure)
