import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import IngestionRun, RawRecord
from app.ingestion.hashing import canonical_json_hash


@dataclass(frozen=True)
class StageResult:
  run_id: uuid.UUID
  fetched: int
  inserted: int
  updated: int
  unchanged: int
  rejected: int


class ConcurrentIngestionError(RuntimeError):
  pass


class StagingRepository:
  """Persist raw source observations and ingestion run state idempotently."""

  def __init__(self, session_factory: sessionmaker[Session], source: str):
    self.session_factory = session_factory
    self.source = source

  def create_run(self, run_id: uuid.UUID | None = None) -> uuid.UUID:
    identifier = run_id or uuid.uuid4()
    with self.session_factory.begin() as session:
      run = session.get(IngestionRun, identifier)
      if run is None:
        session.add(IngestionRun(id=identifier, source=self.source, status="running"))
      elif run.source != self.source:
        raise ValueError(f"Run {identifier} belongs to a different source")
      elif run.status == "succeeded":
        raise ValueError(f"Run {identifier} has already succeeded")
      else:
        run.status = "running"
        run.error_summary = None
        run.finished_at = None
    return identifier

  def stage(
    self,
    run_id: uuid.UUID,
    records: Iterable[Mapping[str, Any]],
  ) -> StageResult:
    fetched = inserted = updated = unchanged = rejected = 0

    with self.session_factory.begin() as session:
      locked = session.execute(
        text("SELECT pg_try_advisory_xact_lock(hashtext(:source))"),
        {"source": self.source},
      ).scalar_one()
      if not locked:
        raise ConcurrentIngestionError(
          f"Another ingestion run is active for {self.source}"
        )

      run = session.get(IngestionRun, run_id)
      if run is None:
        raise ValueError(f"Unknown ingestion run: {run_id}")

      existing = dict(
        session.execute(
          select(RawRecord.source_id, RawRecord.payload_hash).where(
            RawRecord.run_id == run_id,
            RawRecord.source == self.source,
          )
        ).all()
      )

      for record in records:
        fetched += 1
        source_id = str(record.get("UrlName") or "").strip()
        if not source_id:
          rejected += 1
          continue

        payload = dict(record)
        payload_hash = canonical_json_hash(payload)
        previous_hash = existing.get(source_id)
        if previous_hash == payload_hash:
          unchanged += 1
          continue

        statement = insert(RawRecord).values(
          run_id=run_id,
          source=self.source,
          source_id=source_id,
          source_updated_at=_parse_source_timestamp(record),
          payload=payload,
          payload_hash=payload_hash,
          transform_status="pending",
          transform_error=None,
        )
        statement = statement.on_conflict_do_update(
          constraint="uq_raw_record_run_source",
          set_={
            "source_updated_at": statement.excluded.source_updated_at,
            "payload": statement.excluded.payload,
            "payload_hash": statement.excluded.payload_hash,
            "fetched_at": datetime.now(UTC),
            "transform_status": "pending",
            "transform_error": None,
          },
        )
        session.execute(statement)
        if previous_hash is None:
          inserted += 1
        else:
          updated += 1
        existing[source_id] = payload_hash

      run.status = "extract_succeeded"
      run.fetched_count = fetched
      run.inserted_count = inserted
      run.updated_count = updated
      run.unchanged_count = unchanged
      run.rejected_count = rejected
      run.finished_at = datetime.now(UTC)

    return StageResult(
      run_id=run_id,
      fetched=fetched,
      inserted=inserted,
      updated=updated,
      unchanged=unchanged,
      rejected=rejected,
    )

  def fail_run(self, run_id: uuid.UUID, error: Exception) -> None:
    with self.session_factory.begin() as session:
      run = session.get(IngestionRun, run_id)
      if run is None:
        return
      run.status = "failed"
      run.error_summary = str(error)[:2000]
      run.finished_at = datetime.now(UTC)


def _parse_source_timestamp(record: Mapping[str, Any]) -> datetime | None:
  value = record.get("LastModified") or record.get("PublicationDateAndTime")
  if not value or not isinstance(value, str):
    return None
  try:
    return datetime.fromisoformat(value)
  except ValueError:
    return None
