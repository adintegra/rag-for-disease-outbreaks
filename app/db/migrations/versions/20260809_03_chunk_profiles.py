"""Add named chunk profiles and lookup indexes.

Revision ID: 20260809_03
Revises: 20260809_02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260809_03"
down_revision: str | None = "20260809_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
  op.add_column(
    "chunk_dataset",
    sa.Column("profile_name", sa.String(), nullable=True),
    schema="rag",
  )
  op.create_index(
    "ix_chunk_dataset_profile",
    "chunk_dataset",
    ["profile_name", "document_hash", "status"],
    schema="rag",
  )


def downgrade() -> None:
  op.drop_index(
    "ix_chunk_dataset_profile", table_name="chunk_dataset", schema="rag"
  )
  op.drop_column("chunk_dataset", "profile_name", schema="rag")
