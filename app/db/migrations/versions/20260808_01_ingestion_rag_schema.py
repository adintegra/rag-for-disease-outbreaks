"""Create persistent ingestion staging and RAG dataset schemas."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy.dialects import postgresql

revision: str = "20260808_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
  op.execute("CREATE EXTENSION IF NOT EXISTS vector")
  op.execute("CREATE SCHEMA IF NOT EXISTS ingest")
  op.execute("CREATE SCHEMA IF NOT EXISTS rag")

  op.create_table(
    "run",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("source", sa.String(), nullable=False),
    sa.Column("status", sa.String(), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("finished_at", sa.DateTime(timezone=True)),
    sa.Column("source_cursor", sa.String()),
    sa.Column("fetched_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("inserted_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("updated_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("unchanged_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("rejected_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("embedded_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("error_summary", sa.Text()),
    sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    schema="ingest",
  )

  op.create_table(
    "raw_record",
    sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
    sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("ingest.run.id", ondelete="CASCADE"), nullable=False),
    sa.Column("source", sa.String(), nullable=False),
    sa.Column("source_id", sa.String(), nullable=False),
    sa.Column("source_updated_at", sa.DateTime(timezone=True)),
    sa.Column("payload", postgresql.JSONB(), nullable=False),
    sa.Column("payload_hash", sa.String(64), nullable=False),
    sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("transform_status", sa.String(), server_default="pending", nullable=False),
    sa.Column("transform_error", sa.Text()),
    sa.UniqueConstraint("run_id", "source", "source_id", name="uq_raw_record_run_source"),
    schema="ingest",
  )
  op.create_index("ix_raw_record_transform_status", "raw_record", ["transform_status"], schema="ingest")

  op.create_table(
    "document",
    sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
    sa.Column("source", sa.String(), nullable=False),
    sa.Column("source_id", sa.String(), nullable=False),
    sa.Column("source_updated_at", sa.DateTime(timezone=True)),
    sa.Column("title", sa.Text(), nullable=False),
    sa.Column("subtitle", sa.Text()),
    sa.Column("summary", sa.Text()),
    sa.Column("epidemiology", sa.Text()),
    sa.Column("assessment", sa.Text()),
    sa.Column("overview", sa.Text()),
    sa.Column("contents", sa.Text(), nullable=False),
    sa.Column("url", sa.Text(), nullable=False),
    sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("event_date", sa.DateTime(timezone=True)),
    sa.Column("content_hash", sa.String(64), nullable=False),
    sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("transformed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.UniqueConstraint("source", "source_id", name="uq_document_source"),
    schema="rag",
  )

  op.create_table(
    "chunk_dataset",
    sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
    sa.Column("document_id", sa.BigInteger(), sa.ForeignKey("rag.document.id", ondelete="CASCADE"), nullable=False),
    sa.Column("strategy", sa.String(), nullable=False),
    sa.Column("strategy_version", sa.String(), nullable=False),
    sa.Column("parameters", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column("configuration_hash", sa.String(64), nullable=False),
    sa.Column("document_hash", sa.String(64), nullable=False),
    sa.Column("status", sa.String(), server_default="pending", nullable=False),
    sa.Column("error_summary", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("completed_at", sa.DateTime(timezone=True)),
    sa.UniqueConstraint("document_id", "document_hash", "configuration_hash", name="uq_chunk_dataset_configuration"),
    schema="rag",
  )

  op.create_table(
    "chunk",
    sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
    sa.Column("chunk_dataset_id", sa.BigInteger(), sa.ForeignKey("rag.chunk_dataset.id", ondelete="CASCADE"), nullable=False),
    sa.Column("chunk_index", sa.Integer(), nullable=False),
    sa.Column("section", sa.String()),
    sa.Column("contents", sa.Text(), nullable=False),
    sa.Column("content_hash", sa.String(64), nullable=False),
    sa.Column("token_count", sa.Integer()),
    sa.Column("character_start", sa.Integer()),
    sa.Column("character_end", sa.Integer()),
    sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.UniqueConstraint("chunk_dataset_id", "chunk_index", name="uq_chunk_index"),
    schema="rag",
  )

  op.create_table(
    "embedding",
    sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
    sa.Column("chunk_id", sa.BigInteger(), sa.ForeignKey("rag.chunk.id", ondelete="CASCADE"), nullable=False),
    sa.Column("model", sa.String(), nullable=False),
    sa.Column("model_version", sa.String(), nullable=False),
    sa.Column("dimensions", sa.Integer(), nullable=False),
    sa.Column("chunk_hash", sa.String(64), nullable=False),
    sa.Column("embedding", HALFVEC(384), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.UniqueConstraint("chunk_id", "model", "model_version", name="uq_embedding_model"),
    schema="rag",
  )


def downgrade() -> None:
  op.drop_table("embedding", schema="rag")
  op.drop_table("chunk", schema="rag")
  op.drop_table("chunk_dataset", schema="rag")
  op.drop_table("document", schema="rag")
  op.drop_index("ix_raw_record_transform_status", table_name="raw_record", schema="ingest")
  op.drop_table("raw_record", schema="ingest")
  op.drop_table("run", schema="ingest")
  op.execute("DROP SCHEMA IF EXISTS rag")
  op.execute("DROP SCHEMA IF EXISTS ingest")
