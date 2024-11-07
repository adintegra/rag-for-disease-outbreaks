import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd

from sqlalchemy import DateTime, create_engine, Column, Integer, Text, JSON, text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import HALFVEC

from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaEmbeddings

CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5432"
COLLECTION_NAME = "dons_llama"
VECTOR_DIMENSIONS = 3072
MODEL = "llama"


def setup_logging():
  """Configure basic logging for the application."""

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
  )


def init_llm():
  """Initialize the LLM model."""

  if MODEL == "llama":
    # global VECTOR_DIMENSIONS
    # VECTOR_DIMENSIONS = 3072
    return OllamaEmbeddings(model="llama3.2")
  elif MODEL == "nomic":
    # global VECTOR_DIMENSIONS
    # VECTOR_DIMENSIONS = 768
    return NomicEmbeddings(
      model="nomic-embed-text-v1.5", inference_mode="local", device="gpu"
    )


## Embeddings/LLM
## ------------------------------------------------
def get_embedding(content):
  return embeddings.embed_documents([content])


## Database
## ------------------------------------------------
class Base(DeclarativeBase):
  pass


class Record(Base):
  __tablename__ = COLLECTION_NAME
  id = Column(Integer, primary_key=True, autoincrement=True)
  meta = Column(JSON)
  contents = Column(Text)
  embedding = Column(HALFVEC(VECTOR_DIMENSIONS))
  url = Column(Text)
  published_at = Column(DateTime)


def search(query_text: str, limit: int = 5):
  """
  Query the vector database for similar embeddings based on input text.

  More info:
      https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

  Args:
      query_text: The input text to search for.
      limit: The maximum number of results to return.
      metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
      predicates: A Predicates object for complex metadata filtering.
          - Predicates objects are defined by the name of the metadata key, an operator, and a value.
          - Operators: ==, !=, >, >=, <, <=
          - & is used to combine multiple predicates with AND operator.
          - | is used to combine multiple predicates with OR operator.
      time_range: A tuple of (start_date, end_date) to filter results by time.
      return_dataframe: Whether to return results as a DataFrame (default: True).

  Returns:
      Either a list of tuples or a pandas DataFrame containing the search results.

  Basic Examples:
      Basic search:
          vector_store.search("What are your shipping options?")
      Search with metadata filter:
          vector_store.search("Shipping options", metadata_filter={"category": "Shipping"})

  Predicates Examples:
      Search with predicates:
          vector_store.search("Pricing", predicates=client.Predicates("price", ">", 100))
      Search with complex combined predicates:
          complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                          (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
          vector_store.search("High-quality products", predicates=complex_pred)

  Time-based filtering:
      Search with time range:
          vector_store.search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
  """
  engine = create_engine(CONNECTION_STRING)
  Session = sessionmaker(bind=engine)
  session = Session()

  query_embedding = get_embedding(query_text)

  # Normalize before querying the DB
  # query_embedding = np.linalg.norm(query_embedding[0])

  start_time = time.time()

  search_args = {
    "limit": limit,
  }

  # if metadata_filter:
  #   search_args["filter"] = metadata_filter

  # if predicates:
  #   search_args["predicates"] = predicates

  # if time_range:
  #   start_date, end_date = time_range
  #   search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

  # results = self.vec_client.search(query_embedding, **search_args)

  similarity_threshold = 0.7
  query = (
    session.query(
      Record, Record.embedding.cosine_distance(query_embedding[0]).label("distance")
    )
    .filter(Record.embedding.cosine_distance(query_embedding[0]) < similarity_threshold)
    .order_by("distance")
    .limit(limit)
    .all()
  )

  elapsed_time = time.time() - start_time

  logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

  for record, distance in query:
    logging.info(
      f"\nID: {record.id}, Distance: {distance:.4f}, Meta: {record.meta}, \nContents: {record.contents}, URL: {record.url}, Published At: {record.published_at}\n"
    )

  return


# if return_dataframe:
#   return self._create_dataframe_from_results(results)
# else:
#   return results


# def get_similar_texts(self, text):
#         """
#         1. Generate text embedding.
#         2. Compare similarity against vectorDB and get texts similar to the input text.
#         """
#         embeddings = self.emb_mdl.encode(text)
#         # Normalize before querying the DB
#         fnorm = np.linalg.norm(embeddings)
#         lst = list(embeddings/fnorm)
#         # json supports only np.float64. Convert np.float32
#         embed_str = json.dumps(lst, default=np.float64)
#         sim_txts = self.dbexec(self.dbo_stmts['sim_txts'], (embed_str,), "Get similar texts")
#         #print(f"Similar text ids: {[itm[0] for itm in sim_txts]}")
#         all_txts = []
#         contxt = ''
#         # Avoid duplicate sentences, less noise in context is better for LLM response
#         for itm in sim_txts:
#             for txt in itm[1]:
#                 if txt not in all_txts:
#                     all_txts.append(txt)
#                     contxt = f"{contxt} {txt}"
#                     # Do not exceed the tokens limit
#                     if len(contxt.split()) >= _MAX_TKNLEN*_MAX_SIM_TXTS:
#                         break
#         return contxt


## ------------------------------------------------
setup_logging()

# Initialize the model
embeddings = init_llm()

search("In which countries are Malaria outbreaks most prevalent?")
# search("What is the highest number of cases in a single report of Malaria?")
# search("What time of year are Malaria outbreaks most prevalent?")
# search("Are there more outbreaks of Malaria or Dengue?")
