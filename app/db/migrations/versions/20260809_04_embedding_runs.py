"""Add embedding run tracking.

Revision ID: 20260809_04
Revises: 20260809_03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260809_04"
down_revision: str | None = "20260809_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
  op.create_table(
    "embedding_run",
    sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
    sa.Column("profile_name", sa.String(), nullable=False),
    sa.Column("configuration_hash", sa.String(64), nullable=False),
    sa.Column("model", sa.String(), nullable=False),
    sa.Column("model_version", sa.String(), nullable=False),
    sa.Column("dimensions", sa.Integer(), nullable=False),
    sa.Column("status", sa.String(), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.Column("finished_at", sa.DateTime(timezone=True)),
    sa.Column("selected_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("embedded_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("failed_count", sa.Integer(), server_default="0", nullable=False),
    sa.Column("batch_size", sa.Integer(), nullable=False),
    sa.Column("requested_limit", sa.Integer()),
    sa.Column("elapsed_seconds", sa.Float()),
    sa.Column("error_summary", sa.Text()),
    sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False),
    schema="rag",
  )
  op.create_index(
    "ix_embedding_run_identity",
    "embedding_run",
    ["configuration_hash", "model", "model_version", "started_at"],
    schema="rag",
  )


def downgrade() -> None:
  op.drop_index(
    "ix_embedding_run_identity", table_name="embedding_run", schema="rag"
  )
  op.drop_table("embedding_run", schema="rag")
