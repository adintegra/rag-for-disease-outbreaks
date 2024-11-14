from dotenv import load_dotenv
import os
from sqlalchemy import (
  DateTime,
  create_engine,
  Column,
  Integer,
  Text,
  JSON,
  text,
  ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from pgvector.sqlalchemy import HALFVEC


class Base(DeclarativeBase):
  pass


class Document(Base):
  __tablename__ = "document"
  id = Column(Integer, primary_key=True, autoincrement=True)
  meta = Column(JSON)
  contents = Column(Text)
  url = Column(Text)
  published_at = Column(DateTime)


class Embedding(Base):
  __tablename__ = "embedding"
  id = Column(Integer, primary_key=True, autoincrement=True)
  document_id = Column(Integer, ForeignKey("document.id"), nullable=False)
  document = relationship("Document", back_populates="embedding")
  chunk_id = Column(Integer, nullable=False)
  model = Column(Text)
  embedding_3072 = Column(HALFVEC(3072))


def create_tables():
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Base.metadata.create_all(engine)

  Document.embedding = relationship(
    "Embedding", order_by=Embedding.id, back_populates="document"
  )

  Session = sessionmaker(bind=engine)
  session = Session()
  session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
  session.commit()

  with engine.begin() as connection:
    connection.execute(
      text(
        # f"CREATE INDEX IF NOT EXISTS idx_{COLLECTION_NAME}_embedding ON {COLLECTION_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        "CREATE INDEX IF NOT EXISTS idx_embedding ON embedding USING hnsw (embedding_3072 halfvec_l2_ops) WITH (m = 16, ef_construction = 128)"
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
          d.meta,
          d.contents,
          d.url,
          d.published_at,
          e.id AS embedding_id,
          e.chunk_id,
          e.model,
          e.embedding_3072
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
    connection.execute(text("DROP TABLE IF EXISTS embedding CASCADE;"))
    connection.execute(text("DROP TABLE IF EXISTS document CASCADE;"))


def main():
  load_dotenv()
  clean_db()
  create_tables()
  create_view()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
