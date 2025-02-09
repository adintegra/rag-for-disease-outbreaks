from dotenv import load_dotenv
import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, relationship
from vector_store import Base, Document, Embedding


def create_tables():
  engine = create_engine(os.getenv("CONNECTION_STRING"))

  Session = sessionmaker(bind=engine)
  session = Session()
  session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
  session.commit()

  # Base.metadata.create_all(engine)

  # Create only Document and Embedding tables
  Base.metadata.tables = {
    "document": Base.metadata.tables["document"],
    "embedding": Base.metadata.tables["embedding"],
  }

  Document.embedding = relationship(
    "Embedding", order_by=Embedding.id, back_populates="document"
  )

  # Create indexes
  with engine.begin() as connection:
    connection.execute(
      text(
        # f"CREATE INDEX IF NOT EXISTS idx_{COLLECTION_NAME}_embedding ON {COLLECTION_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        "CREATE INDEX IF NOT EXISTS idx_embedding_384 ON embedding USING hnsw (embedding_384 halfvec_l2_ops) WITH (m = 16, ef_construction = 128)"
      )
    )

  with engine.begin() as connection:
    connection.execute(
      text(
        "CREATE INDEX IF NOT EXISTS idx_embedding_768 ON embedding USING hnsw (embedding_768 halfvec_l2_ops) WITH (m = 16, ef_construction = 128)"
      )
    )

  with engine.begin() as connection:
    connection.execute(
      text(
        "CREATE INDEX IF NOT EXISTS idx_embedding_1024 ON embedding USING hnsw (embedding_1024 halfvec_l2_ops) WITH (m = 16, ef_construction = 128)"
      )
    )


def create_view():
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  # begin() commits automatically; this isn't Oracle, commits are needed after DDL
  with engine.begin() as connection:
    connection.execute(
      text(
        """
        CREATE OR REPLACE VIEW v_doc_embedding AS
        SELECT
          d.id AS document_id,
          e.id AS embedding_id,
          e.chunk_id,
          e.model,
          d.meta,
          d.contents,
          d.url,
          d.published_at,
          d.summary,
          d.event_date,
          d.batch,
          CASE
          WHEN model = 'nomic-embed-text-v1.5' THEN e.embedding_768
          WHEN model = 'all-minilm' THEN e.embedding_384
          WHEN model = 'mxbai-embed-large' THEN e.embedding_1024
          ELSE e.embedding_768
          END AS embedding
        FROM
          document d
        JOIN
          embedding e
        ON
          d.id = e.document_id;
        """
      )
    )


def clean_db():
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  # Base.metadata.drop_all(engine)
  with engine.begin() as connection:
    connection.execute(text("DROP VIEW IF EXISTS v_doc_embedding;"))
    connection.execute(text("DROP TABLE IF EXISTS embedding CASCADE;"))
    connection.execute(text("DROP TABLE IF EXISTS document CASCADE;"))


def main():
  load_dotenv("../../.env")

  # Logging
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  clean_db()
  create_tables()
  create_view()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
