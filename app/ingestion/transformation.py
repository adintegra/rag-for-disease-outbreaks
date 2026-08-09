import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Document, IngestionRun, RawRecord
from app.ingestion.staging import ConcurrentIngestionError
from app.ingestion.transform import CanonicalDocument, normalize_who_don


@dataclass(frozen=True)
class TransformResult:
  run_id: uuid.UUID
  processed: int
  inserted: int
  updated: int
  unchanged: int
  rejected: int


class TransformationRepository:
  """Transform staged WHO records into canonical documents in bounded batches."""

  def __init__(
    self,
    session_factory: sessionmaker[Session],
    source: str,
    batch_size: int = 100,
  ):
    self.session_factory = session_factory
    self.source = source
    self.batch_size = batch_size

  def transform(self, run_id: uuid.UUID) -> TransformResult:
    processed = inserted = updated = unchanged = rejected = 0
    self._require_run(run_id)

    while raw_ids := self._pending_ids(run_id):
      result = self._transform_batch(run_id, raw_ids)
      processed += result.processed
      inserted += result.inserted
      updated += result.updated
      unchanged += result.unchanged
      rejected += result.rejected

    with self.session_factory.begin() as session:
      run = session.get(IngestionRun, run_id)
      if run is None:
        raise ValueError(f"Unknown ingestion run: {run_id}")
      totals = session.execute(
        select(
          func.count(RawRecord.id).filter(RawRecord.transform_status == "transformed"),
          func.count(RawRecord.id).filter(RawRecord.transform_status == "rejected"),
        ).where(RawRecord.run_id == run_id)
      ).one()
      run.transformed_count = totals[0]
      run.transform_rejected_count = totals[1]
      run.status = "transform_succeeded" if totals[1] == 0 else "partially_failed"
      run.finished_at = datetime.now(UTC)

    return TransformResult(run_id, processed, inserted, updated, unchanged, rejected)

  def _require_run(self, run_id: uuid.UUID) -> None:
    with self.session_factory() as session:
      run = session.get(IngestionRun, run_id)
      if run is None or run.source != self.source:
        raise ValueError(f"Unknown {self.source} ingestion run: {run_id}")
      if run.status == "running":
        raise ValueError(f"Run {run_id} has not completed extraction")

  def _pending_ids(self, run_id: uuid.UUID) -> list[int]:
    with self.session_factory() as session:
      return list(
        session.scalars(
          select(RawRecord.id)
          .where(
            RawRecord.run_id == run_id,
            RawRecord.source == self.source,
            RawRecord.transform_status == "pending",
          )
          .order_by(RawRecord.id)
          .limit(self.batch_size)
        )
      )

  def _transform_batch(
    self, run_id: uuid.UUID, raw_ids: list[int]
  ) -> TransformResult:
    inserted = updated = unchanged = rejected = 0
    with self.session_factory.begin() as session:
      locked = session.execute(
        text("SELECT pg_try_advisory_xact_lock(hashtext(:source))"),
        {"source": f"{self.source}:transform"},
      ).scalar_one()
      if not locked:
        raise ConcurrentIngestionError(
          f"Another transform is active for {self.source}"
        )

      records = list(
        session.scalars(
          select(RawRecord)
          .where(RawRecord.id.in_(raw_ids))
          .order_by(RawRecord.id)
          .with_for_update()
        )
      )
      for raw in records:
        try:
          canonical = normalize_who_don(raw.payload, self.source)
          action = self._upsert_document(session, canonical)
          raw.transform_status = "transformed"
          raw.transform_action = action
          raw.transform_error = None
          if action == "inserted":
            inserted += 1
          elif action == "updated":
            updated += 1
          else:
            unchanged += 1
        except (TypeError, ValueError) as error:
          raw.transform_status = "rejected"
          raw.transform_action = "rejected"
          raw.transform_error = str(error)[:2000]
          rejected += 1

    return TransformResult(
      run_id, len(raw_ids), inserted, updated, unchanged, rejected
    )

  def _upsert_document(
    self, session: Session, canonical: CanonicalDocument
  ) -> str:
    document = session.scalar(
      select(Document).where(
        Document.source == canonical.source,
        Document.source_id == canonical.source_id,
      )
    )
    now = datetime.now(UTC)
    if document is None:
      session.add(Document(**_document_values(canonical), first_seen_at=now, last_seen_at=now))
      return "inserted"

    document.last_seen_at = now
    if (
      document.content_hash == canonical.content_hash
      and document.source_updated_at == canonical.source_updated_at
    ):
      return "unchanged"

    for field, value in _document_values(canonical).items():
      setattr(document, field, value)
    document.transformed_at = now
    return "updated"


def _document_values(canonical: CanonicalDocument) -> dict:
  return {
    "source": canonical.source,
    "source_id": canonical.source_id,
    "source_updated_at": canonical.source_updated_at,
    "title": canonical.title,
    "subtitle": canonical.subtitle,
    "summary": canonical.summary,
    "epidemiology": canonical.epidemiology,
    "assessment": canonical.assessment,
    "overview": canonical.overview,
    "contents": canonical.contents,
    "url": canonical.url,
    "published_at": canonical.published_at,
    "event_date": canonical.event_date,
    "content_hash": canonical.content_hash,
    "document_metadata": canonical.metadata,
  }
