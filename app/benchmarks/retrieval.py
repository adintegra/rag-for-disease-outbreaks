import os
import sys

# Required to import from parent directory
sys.path.append("..")

from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from db.vector_store import DocEmbeddingView
from sklearn.metrics import precision_score, recall_score


def pgvector_retrieve(query):
  """Retrieve documents using PGVector."""
  model = OllamaEmbeddings(model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL"))
  query_vector = model.embed_query(query)

  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  # See https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy
  # l2_distance also works
  stmt = (
    select(
      DocEmbeddingView.document_id,
      DocEmbeddingView.embedding_id,
      DocEmbeddingView.chunk_id,
      DocEmbeddingView.contents,
      DocEmbeddingView.embedding.cosine_distance(query_vector).label("l2_distance"),
    )
    .filter(DocEmbeddingView.model == model.model)
    .order_by(DocEmbeddingView.embedding.cosine_distance(query_vector))
    .limit(5)
  )

  retrieved_docs = session.execute(stmt).all()

  return [doc.contents.lower() for doc in retrieved_docs]


def get_corpus():
  """Retrieve all documents from the database and return as an array."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  metadata = sa.MetaData()
  documents_table = sa.Table("document", metadata, autoload_with=engine)

  with engine.connect() as connection:
    query = sa.select(documents_table.c.contents)
    result = connection.execute(query).fetchall()

  return [row[0].lower() for row in result]


# Perform BM25 Retrieval
def bm25_retrieve(query, corpus):
  # https://www.youtube.com/watch?v=ruBm9WywevM&t=498s
  tokenized_corpus = [doc.split() for doc in corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  tokenized_query = query.split(" ")
  # print(bm25.get_scores(tokenized_query))
  return bm25.get_top_n(tokenized_query, corpus, n=5)


# Evaluate Retrieval Performance
def evaluate_retrieval(true_relevant_docs, retrieved_docs):
  y_true = [1 if doc in true_relevant_docs else 0 for doc in retrieved_docs]
  y_pred = [1] * len(retrieved_docs)
  precision = precision_score(y_true, y_pred, zero_division=1)
  recall = recall_score(y_true, y_pred, zero_division=1)
  return {"precision": precision, "recall": recall}


def print_results(results):
  for result in results:
    print(result + "\n")
  print("--------------------")


def main():
  load_dotenv("../../.env")

  corpus = get_corpus()

  # query = "malaria kenya"
  query = "recent ebola outbreak in africa"

  retrieved_docs_pgvector = pgvector_retrieve(query)
  retrieved_docs_bm25 = bm25_retrieve(query, corpus)

  # Define true relevant documents (for evaluation purposes, should be manually labeled)
  true_relevant_docs = []
  with open("relevant_texts.txt", "r") as file:
    true_relevant_docs = [line.strip() for line in file.readlines()]

  # Evaluate retrieval performance
  evaluation_pgvector = evaluate_retrieval(true_relevant_docs, retrieved_docs_pgvector)
  evaluation_bm25 = evaluate_retrieval(true_relevant_docs, retrieved_docs_bm25)

  print("\nPGVector Results:\n")
  print_results(retrieved_docs_pgvector)

  print("\nBM25 Results:\n")
  print_results(retrieved_docs_bm25)

  print("\nPGVector Evaluation:\n", evaluation_pgvector)
  print("\nBM25 Evaluation:\n", evaluation_bm25)


## ------------------------------------------------------
if __name__ == "__main__":
  main()
