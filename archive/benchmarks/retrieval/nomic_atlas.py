import os
import sys

# Required to import from parent directory
sys.path.append("..")

from dotenv import load_dotenv
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from db.vector_store import DocEmbeddingView
from nomic import atlas


def get_embeddings():
  """Retrieve all documents from the database and return as an array."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  result = None

  stmt = select(DocEmbeddingView.embedding).where(
    DocEmbeddingView.batch == 1, DocEmbeddingView.model == "nomic-embed-text"
  )

  try:
    result = session.execute(stmt).all()
  except Exception as e:
    print(e)

  # print(result[0])
  # print(type(result[0][0].to_numpy()))
  # print(np.shape(result[0][0].to_numpy()))

  res = np.empty((len(result), 768))

  for row in result:
    arr = row[0].to_numpy()
    np.append(res, arr[np.newaxis, :], axis=0)

  return res


def main():
  # Load environment variables
  load_dotenv()

  embeddings = get_embeddings()

  # print(embeddings[:10])
  # print(np.shape(embeddings))
  # print(np.size(embeddings))

  # Create Atlas project
  dataset = atlas.map_data(embeddings=embeddings)
  print(dataset)


if __name__ == "__main__":
  main()
