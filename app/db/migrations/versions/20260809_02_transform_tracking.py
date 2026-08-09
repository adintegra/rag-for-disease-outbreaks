"""Add resumable transform accounting.

Revision ID: 20260809_02
Revises: 20260808_01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260809_02"
down_revision: str | None = "20260808_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
  op.add_column(
    "run",
    sa.Column("transformed_count", sa.Integer(), server_default="0", nullable=False),
    schema="ingest",
  )
  op.add_column(
    "run",
    sa.Column(
      "transform_rejected_count", sa.Integer(), server_default="0", nullable=False
    ),
    schema="ingest",
  )
  op.add_column(
    "raw_record", sa.Column("transform_action", sa.String()), schema="ingest"
  )


def downgrade() -> None:
  op.drop_column("raw_record", "transform_action", schema="ingest")
  op.drop_column("run", "transform_rejected_count", schema="ingest")
  op.drop_column("run", "transformed_count", schema="ingest")
