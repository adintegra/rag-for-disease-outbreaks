from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import DateTime, create_engine, Column, Integer, Text, JSON
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import HALFVEC


def setup_logging():
  """Configure basic logging for the application."""

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )


# Prepare data for insertion
def prepare_record(row):
  """Prepare a record for insertion into the vector store."""

  # content = f"Title: {row['Title']}\nSummary: {row['Summary']}"
  # TitleSuffix;Epidemiology
  # content = f"Title: {row['Title']}\nSummary: {row['Summary']}\nOverview: {row['Overview']}\nAssessment: {row['Assessment']}"
  content = f"Title: {row['Title']}\nSummary: {row['Summary'] if pd.notna(row['Summary']) else row['Overview']}"

  published = datetime.strptime(
    row["PublicationDate"], "%Y-%m-%dT%H:%M:%SZ"
  ).isoformat()

  return pd.Series(
    {
      "meta": {
        "published_at": published,
        "url": row["Url"],
      },
      "contents": content,
      "url": row["Url"],
      "published_at": published,
    }
  )


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
  document_id = Column(Integer, nullable=False)
  chunk_id = Column(Integer, nullable=False)
  model = Column(Text)
  embedding_3072 = Column(HALFVEC(3072))


def upsert(df: pd.DataFrame):
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  for i, row in df.iterrows():
    record = Document(**row)
    session.add(record)

  session.commit()
  session.close()


def main():
  setup_logging()
  load_dotenv()

  # Read the CSV file and prepare records
  df = pd.read_csv("../data/dons.csv", sep=";")
  records_df = df.apply(prepare_record, axis=1)

  # create_index()  # DiskAnnIndex
  upsert(records_df)


## ------------------------------------------------------
if __name__ == "__main__":
  main()
