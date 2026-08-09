import uuid
from datetime import datetime
from typing import ClassVar

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
  BigInteger,
  DateTime,
  ForeignKey,
  Integer,
  String,
  Text,
  UniqueConstraint,
  func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
  pass


class IngestionRun(Base):
  __tablename__ = "run"
  __table_args__: ClassVar[dict[str, str]] = {"schema": "ingest"}

  id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
  source: Mapped[str] = mapped_column(String, nullable=False)
  status: Mapped[str] = mapped_column(String, nullable=False)
  started_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )
  finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
  source_cursor: Mapped[str | None] = mapped_column(String)
  fetched_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  inserted_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  updated_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  unchanged_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  rejected_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  embedded_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  transformed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
  transform_rejected_count: Mapped[int] = mapped_column(
    Integer, default=0, nullable=False
  )
  error_summary: Mapped[str | None] = mapped_column(Text)
  run_metadata: Mapped[dict] = mapped_column(
    "metadata", JSONB, default=dict, nullable=False
  )

  raw_records: Mapped[list["RawRecord"]] = relationship(back_populates="run")


class RawRecord(Base):
  __tablename__ = "raw_record"
  __table_args__ = (
    UniqueConstraint("run_id", "source", "source_id", name="uq_raw_record_run_source"),
    {"schema": "ingest"},
  )

  id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
  run_id: Mapped[uuid.UUID] = mapped_column(
    ForeignKey("ingest.run.id", ondelete="CASCADE"), nullable=False
  )
  source: Mapped[str] = mapped_column(String, nullable=False)
  source_id: Mapped[str] = mapped_column(String, nullable=False)
  source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
  payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
  payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
  fetched_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )
  transform_status: Mapped[str] = mapped_column(
    String, default="pending", nullable=False
  )
  transform_error: Mapped[str | None] = mapped_column(Text)
  transform_action: Mapped[str | None] = mapped_column(String)

  run: Mapped[IngestionRun] = relationship(back_populates="raw_records")


class Document(Base):
  __tablename__ = "document"
  __table_args__ = (
    UniqueConstraint("source", "source_id", name="uq_document_source"),
    {"schema": "rag"},
  )

  id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
  source: Mapped[str] = mapped_column(String, nullable=False)
  source_id: Mapped[str] = mapped_column(String, nullable=False)
  source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
  title: Mapped[str] = mapped_column(Text, nullable=False)
  subtitle: Mapped[str | None] = mapped_column(Text)
  summary: Mapped[str | None] = mapped_column(Text)
  epidemiology: Mapped[str | None] = mapped_column(Text)
  assessment: Mapped[str | None] = mapped_column(Text)
  overview: Mapped[str | None] = mapped_column(Text)
  contents: Mapped[str] = mapped_column(Text, nullable=False)
  url: Mapped[str] = mapped_column(Text, nullable=False)
  published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
  event_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
  content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
  document_metadata: Mapped[dict] = mapped_column(
    "metadata", JSONB, default=dict, nullable=False
  )
  first_seen_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )
  last_seen_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )
  transformed_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )

  chunk_datasets: Mapped[list["ChunkDataset"]] = relationship(
    back_populates="document", cascade="all, delete-orphan"
  )


class ChunkDataset(Base):
  __tablename__ = "chunk_dataset"
  __table_args__ = (
    UniqueConstraint(
      "document_id",
      "document_hash",
      "configuration_hash",
      name="uq_chunk_dataset_configuration",
    ),
    {"schema": "rag"},
  )

  id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
  document_id: Mapped[int] = mapped_column(
    ForeignKey("rag.document.id", ondelete="CASCADE"), nullable=False
  )
  strategy: Mapped[str] = mapped_column(String, nullable=False)
  strategy_version: Mapped[str] = mapped_column(String, nullable=False)
  parameters: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
  configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
  document_hash: Mapped[str] = mapped_column(String(64), nullable=False)
  status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
  error_summary: Mapped[str | None] = mapped_column(Text)
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )
  completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

  document: Mapped[Document] = relationship(back_populates="chunk_datasets")
  chunks: Mapped[list["Chunk"]] = relationship(
    back_populates="dataset", cascade="all, delete-orphan"
  )


class Chunk(Base):
  __tablename__ = "chunk"
  __table_args__ = (
    UniqueConstraint("chunk_dataset_id", "chunk_index", name="uq_chunk_index"),
    {"schema": "rag"},
  )

  id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
  chunk_dataset_id: Mapped[int] = mapped_column(
    ForeignKey("rag.chunk_dataset.id", ondelete="CASCADE"), nullable=False
  )
  chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
  section: Mapped[str | None] = mapped_column(String)
  contents: Mapped[str] = mapped_column(Text, nullable=False)
  content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
  token_count: Mapped[int | None] = mapped_column(Integer)
  character_start: Mapped[int | None] = mapped_column(Integer)
  character_end: Mapped[int | None] = mapped_column(Integer)
  chunk_metadata: Mapped[dict] = mapped_column(
    "metadata", JSONB, default=dict, nullable=False
  )
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )

  dataset: Mapped[ChunkDataset] = relationship(back_populates="chunks")
  embeddings: Mapped[list["Embedding"]] = relationship(
    back_populates="chunk", cascade="all, delete-orphan"
  )


class Embedding(Base):
  __tablename__ = "embedding"
  __table_args__ = (
    UniqueConstraint(
      "chunk_id", "model", "model_version", name="uq_embedding_model"
    ),
    {"schema": "rag"},
  )

  id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
  chunk_id: Mapped[int] = mapped_column(
    ForeignKey("rag.chunk.id", ondelete="CASCADE"), nullable=False
  )
  model: Mapped[str] = mapped_column(String, nullable=False)
  model_version: Mapped[str] = mapped_column(String, nullable=False)
  dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
  chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
  embedding: Mapped[list[float]] = mapped_column(HALFVEC(384), nullable=False)
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), server_default=func.now(), nullable=False
  )

  chunk: Mapped[Chunk] = relationship(back_populates="embeddings")
