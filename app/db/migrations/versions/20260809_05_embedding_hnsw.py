"""Add cosine HNSW index for retrieval.

Revision ID: 20260809_05
Revises: 20260809_04
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260809_05"
down_revision: str | None = "20260809_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
  with op.get_context().autocommit_block():
    op.execute(
      "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_embedding_cosine_hnsw "
      "ON rag.embedding USING hnsw (embedding halfvec_cosine_ops) "
      "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
  with op.get_context().autocommit_block():
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS rag.ix_embedding_cosine_hnsw")
