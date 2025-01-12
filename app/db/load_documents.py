from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from vector_store import Document


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
  load_dotenv("../../.env")

  # Logging
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  # Read the CSV file and prepare records
  df = pd.read_csv("../data/dons.csv", sep=";")
  records_df = df.apply(prepare_record, axis=1)

  upsert(records_df)


## ------------------------------------------------------
if __name__ == "__main__":
  main()
