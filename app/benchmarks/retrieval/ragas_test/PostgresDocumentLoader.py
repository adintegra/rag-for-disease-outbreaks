from dotenv import load_dotenv
import os
import sys

# Required to import from parent directory
sys.path.append("../../../")

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import List
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from db.vector_store import Document as DBDocument


class PostgresDocumentLoader(BaseLoader):
  def __init__(self):
    """Custom LangChain PostgresDocumentLoader."""
    self.engine = create_engine(os.getenv("CONNECTION_STRING"))
    self.Session = sessionmaker(bind=self.engine)
    self.batch = int(os.getenv("CURRENT_BATCH"))

  def load(self) -> List[Document]:
    """Load documents from Postgres database.

    Returns:
      List of Document objects with content and metadata.
    """
    result = None
    documents = []

    try:
      session = self.Session()
      stmt = select(
        DBDocument.contents, DBDocument.event_date, DBDocument.id, DBDocument.url
      ).where(DBDocument.batch == self.batch)

      result = session.execute(stmt).all()

      # Process results
      for row in result:
        # Create Document object with content and metadata
        doc = Document(
          page_content=row.contents,
          metadata={
            "url": row.url,
            "event_date": row.event_date.isoformat(),
            "doc_id": row.id,
          },
        )
        documents.append(doc)

    except Exception as e:
      print(e)
    finally:
      session.close()

    return documents


def main():
  # Load environment variables
  load_dotenv()

  loader = PostgresDocumentLoader()
  documents = loader.load()
  print(len(documents))


if __name__ == "__main__":
  main()
