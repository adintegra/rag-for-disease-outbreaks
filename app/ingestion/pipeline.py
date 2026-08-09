import uuid

from app.ingestion.source.who import WhoDonClient
from app.ingestion.staging import StageResult, StagingRepository


class IngestionPipeline:
  """Coordinate extraction and persistent staging behind one run interface."""

  def __init__(self, source: WhoDonClient, staging: StagingRepository):
    self.source = source
    self.staging = staging

  def extract(self, run_id: uuid.UUID | None = None) -> StageResult:
    identifier = self.staging.create_run(run_id)
    try:
      records = list(self.source.iter_records())
      return self.staging.stage(identifier, records)
    except Exception as error:
      self.staging.fail_run(identifier, error)
      raise
