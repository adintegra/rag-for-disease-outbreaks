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


# Set vars before testing
BATCH = 1
RELEVANT_DOCS = []
PG_RETRIEVED = []
CORPUS = "Full"
# CORPUS = "Summarized"

MODEL = "all-minilm"  # dim 384 / context window 512
# MODEL = "nomic-embed-text"  # dim 768 / context window 2048
# MODEL = "mxbai-embed-large"  # dim 1024 / context window 512


def get_relevant_docs():
  """Retrieve the true relevant documents from the DB."""
  col = ""
  if CORPUS == "Full":
    col = "contents"
  elif CORPUS == "Summarized":
    col = "summary"

  sql = f"""
    select id,
    {col}
    from document
    where contents like 'Title: Ebola%'
    and published_at >= '2021-01-01'
    and right(meta->>'don_id', 6) in ('DON433','DON428','DON425','DON423','DON421','DON411','DON410','DON404','DON398','DON377','DON351','DON328','DON325','DON312','DON310','ongo_1')
    and batch = 1;
  """

  engine = create_engine(os.getenv("CONNECTION_STRING"))
  with engine.connect() as connection:
    result = connection.execute(sa.text(sql))
    docs = []
    # Can only access the cursor once
    for row in result:
      RELEVANT_DOCS.append(row[0])
      docs.append(row[1].lower())
    return docs


def pgvector_retrieve(query):
  """Retrieve documents using PGVector."""
  emb_model = OllamaEmbeddings(model=MODEL, base_url=os.getenv("OLLAMA_BASE_URL"))
  query_vector = emb_model.embed_query(query)

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
    .filter(DocEmbeddingView.model == emb_model.model)
    .filter(DocEmbeddingView.batch == BATCH)
    .order_by(DocEmbeddingView.embedding.cosine_distance(query_vector))
    .limit(8)
  )

  retrieved_docs = session.execute(stmt).all()

  for doc in retrieved_docs:
    PG_RETRIEVED.append(doc.document_id)
    #   print("PGVector: ", doc.document_id)
    #   print(doc.l2_distance)

  return [doc.contents.lower() for doc in retrieved_docs]


def get_corpus():
  """Retrieve all documents from the database and return as an array."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  metadata = sa.MetaData()
  documents_table = sa.Table("document", metadata, autoload_with=engine)

  with engine.connect() as connection:
    if CORPUS == "Full":
      query = sa.select(documents_table.c.contents).filter(
        documents_table.c.batch == BATCH
      )
    elif CORPUS == "Summarized":
      query = sa.select(documents_table.c.summary).filter(
        documents_table.c.batch == BATCH
      )
    result = connection.execute(query).fetchall()

  return [row[0].lower() for row in result]


# Perform BM25 Retrieval
def bm25_retrieve(query, corpus):
  # https://www.youtube.com/watch?v=ruBm9WywevM&t=498s
  tokenized_corpus = [doc.split() for doc in corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  tokenized_query = query.split()
  # print(bm25.get_scores(tokenized_query))
  return bm25.get_top_n(tokenized_query, corpus, n=8)


def get_bm25_docids(docs):
  """Retrieve the document IDs for the BM25 retrieved documents."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  metadata = sa.MetaData()
  documents_table = sa.Table("document", metadata, autoload_with=engine)

  result = []
  with engine.connect() as connection:
    for doc in docs:
      query = sa.select(documents_table.c.id).where(
        sa.func.lower(documents_table.c.contents) == doc
      )
      doc_result = connection.execute(query).fetchone()
      if doc_result:
        result.append(doc_result[0])

  return result


# Evaluate Retrieval Performance
def evaluate_retrieval(true_relevant_docs, retrieved_docs):
  y_true = [1 if doc in true_relevant_docs else 0 for doc in retrieved_docs]
  y_pred = [1] * len(retrieved_docs)
  precision = precision_score(y_true, y_pred, zero_division=1)
  recall = recall_score(y_true, y_pred, zero_division=1)
  return {"precision": precision, "recall": recall}


def print_results(results):
  for result in results:
    print("-- DOC ------------------")
    print(result[:275].replace("\n", " "), "...")
    print("-- END ------------------")


def bm25_v_embeddings(q):
  corpus = get_corpus()

  # Search for query in corpus
  retrieved_docs_pgvector = pgvector_retrieve(q)
  retrieved_docs_bm25 = bm25_retrieve(q, corpus)

  bm25_docids = get_bm25_docids(retrieved_docs_bm25)

  # Define true relevant documents/ground truth (for evaluation purposes, should be manually labeled)
  true_relevant_docs = get_relevant_docs()

  # Evaluate retrieval performance
  evaluation_pgvector = evaluate_retrieval(true_relevant_docs, retrieved_docs_pgvector)
  evaluation_bm25 = evaluate_retrieval(true_relevant_docs, retrieved_docs_bm25)

  print(
    "-- Evaluation Results ---------------------------------------------------------"
  )
  print("Model:", MODEL)
  print("Query:", q)
  print(CORPUS, "corpus")

  print("\nEmbedding Search Results:\n")
  print_results(retrieved_docs_pgvector)

  print("\nEmbedding Retrieved Docs:", len(retrieved_docs_pgvector))
  print(PG_RETRIEVED)
  print("True Positives:", list(set(PG_RETRIEVED) & set(RELEVANT_DOCS)))

  print("\nBM25 Search Results:\n")
  print_results(retrieved_docs_bm25)

  print("\nBM25 Retrieved Docs:", len(retrieved_docs_bm25))
  print(bm25_docids)
  print("True Positives:", list(set(bm25_docids) & set(RELEVANT_DOCS)))

  print("\nGround Truth:", len(true_relevant_docs))
  print(RELEVANT_DOCS)

  print("\nEmbedding Evaluation:\n", evaluation_pgvector)
  print("\nBM25 Evaluation:\n", evaluation_bm25)

  print("\n\n")


def main():
  load_dotenv("../../.env")

  # Define test query
  # query = "malaria kenya"
  # query = "recent ebola outbreak in africa"
  query = "ebola outbreaks in africa after 2021"

  bm25_v_embeddings(query)


## ------------------------------------------------------
if __name__ == "__main__":
  main()
