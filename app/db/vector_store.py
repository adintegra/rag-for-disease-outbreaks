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

  def __init__(self, url, result_all, result_no_stop_words):
    self.url = url
    self.result_all = result_all
    self.result_no_stop_words = result_no_stop_words

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

  def __init__(self, url, result_all, result_no_stop_words):
    self.url = url
    self.result_all = result_all
    self.result_no_stop_words = result_no_stop_words

  def __repr__(self):
    return "<id {}>".format(self.id)
