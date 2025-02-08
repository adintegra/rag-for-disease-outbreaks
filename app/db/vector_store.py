from sqlalchemy import (
  DateTime,
  Column,
  Integer,
  Text,
  JSON,
  ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import HALFVEC


class VectorStore:
  """A class for managing vector operations and database interactions."""

  def __init__(self):
    self.settings = ""


class Base(DeclarativeBase):
  pass


class Document(Base):
  __tablename__ = "document"

  id = Column(Integer, primary_key=True, autoincrement=True)
  meta = Column(JSON)
  contents = Column(Text)
  url = Column(Text)
  published_at = Column(DateTime)
  summary = Column(Text, nullable=True)
  batch = Column(Integer, nullable=True)

  def __init__(self, meta, contents, url, published_at, summary, batch):
    self.meta = meta
    self.contents = contents
    self.url = url
    self.published_at = published_at
    self.summary = summary
    self.batch = batch

  def __repr__(self):
    return "<id {}>".format(self.id)


class Embedding(Base):
  __tablename__ = "embedding"

  id = Column(Integer, primary_key=True, autoincrement=True)
  document_id = Column(Integer, ForeignKey("document.id"), nullable=False)
  document = relationship("Document", backref="embedding")
  chunk_id = Column(Integer, nullable=False)
  model = Column(Text)
  embedding_256 = Column(HALFVEC(256))
  embedding_384 = Column(HALFVEC(384))
  embedding_512 = Column(HALFVEC(512))
  embedding_768 = Column(HALFVEC(768))
  embedding_1024 = Column(HALFVEC(1024))
  embedding_1536 = Column(HALFVEC(1536))
  embedding_3072 = Column(HALFVEC(3072))
  embedding_4096 = Column(HALFVEC(4096))
  embedding_8192 = Column(HALFVEC(8192))

  def __repr__(self):
    return "<id {}>".format(self.id)


class DocEmbeddingView(Base):
  __tablename__ = "v_doc_embedding"

  document_id = Column(Integer, primary_key=True)
  embedding_id = Column(Integer, primary_key=True)
  chunk_id = Column(Integer)
  model = Column(Text)
  meta = Column(JSON)
  contents = Column(Text)
  url = Column(Text)
  published_at = Column(DateTime)
  summary = Column(Text)
  batch = Column(Integer)
  embedding = Column(HALFVEC)

  __table_args__ = {"extend_existing": True}

  def __repr__(self):
    return "<id {}>".format(self.id)
