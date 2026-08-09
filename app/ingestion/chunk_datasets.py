from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, ChunkDataset, Document
from app.ingestion.chunk_profiles import ChunkProfile
from app.ingestion.chunking import generate_chunks
from app.ingestion.staging import ConcurrentIngestionError


@dataclass(frozen=True)
class ChunkDatasetResult:
  profile_name: str
  documents: int
  created: int
  skipped: int
  failed: int
  chunks: int


class ChunkDatasetRepository:
  """Create reproducible chunk datasets for one named strategy profile."""

  def __init__(self, session_factory: sessionmaker[Session]):
    self.session_factory = session_factory

  def generate(self, profile: ChunkProfile) -> ChunkDatasetResult:
    documents = created = skipped = failed = chunk_count = 0

    with self.session_factory.begin() as session:
      locked = session.execute(
        text("SELECT pg_try_advisory_xact_lock(hashtext(:profile))"),
        {"profile": f"chunk:{profile.configuration_hash}"},
      ).scalar_one()
      if not locked:
        raise ConcurrentIngestionError(
          f"Another chunking run is active for profile {profile.name}"
        )

      existing_datasets = {
        (dataset.document_id, dataset.document_hash): dataset
        for dataset in session.scalars(
          select(ChunkDataset).where(
            ChunkDataset.configuration_hash == profile.configuration_hash
          )
        )
      }
      document_stream = session.scalars(select(Document).order_by(Document.id))
      for document in document_stream:
        documents += 1
        dataset = existing_datasets.get((document.id, document.content_hash))
        if dataset is not None and dataset.status == "complete":
          skipped += 1
          continue

        if dataset is None:
          dataset = ChunkDataset(
            document_id=document.id,
            profile_name=profile.name,
            strategy=profile.strategy,
            strategy_version=profile.strategy_version,
            parameters=profile.configuration,
            configuration_hash=profile.configuration_hash,
            document_hash=document.content_hash,
            status="pending",
          )
          session.add(dataset)
          session.flush()
        else:
          session.execute(delete(Chunk).where(Chunk.chunk_dataset_id == dataset.id))
          dataset.profile_name = profile.name
          dataset.status = "pending"
          dataset.error_summary = None
          dataset.completed_at = None

        try:
          generated = generate_chunks(document, profile)
          if not generated:
            raise ValueError("Chunk strategy generated no chunks")
          session.add_all(
            [
              Chunk(
                chunk_dataset_id=dataset.id,
                chunk_index=chunk.index,
                section=chunk.section,
                contents=chunk.contents,
                content_hash=chunk.content_hash,
                token_count=len(chunk.contents.split()),
                character_start=chunk.character_start,
                character_end=chunk.character_end,
                chunk_metadata=chunk.metadata,
              )
              for chunk in generated
            ]
          )
          dataset.status = "complete"
          dataset.completed_at = datetime.now(UTC)
          created += 1
          chunk_count += len(generated)
        except ValueError as error:
          dataset.status = "failed"
          dataset.error_summary = str(error)[:2000]
          failed += 1

    return ChunkDatasetResult(
      profile.name, documents, created, skipped, failed, chunk_count
    )
