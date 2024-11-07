import logging
import time
from datetime import datetime
import pandas as pd

from sqlalchemy import DateTime, create_engine, Column, Integer, Text, JSON, text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import HALFVEC

from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaEmbeddings


# from langchain.document_loaders import TextLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.pgvector import PGVector

# import os
# ### LLM
# from langchain_ollama import ChatOllama

# local_llm = "llama3.2:3b-instruct-fp16"
# llm = ChatOllama(model=local_llm, temperature=0)
# llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")


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


# Prepare data for insertion
def prepare_record(row):
  """Prepare a record for insertion into the vector store."""

  content = f"Title: {row['Title']}\nSummary: {row['Summary']}"
  published = datetime.strptime(
    row["PublicationDate"], "%Y-%m-%dT%H:%M:%SZ"
  ).isoformat()

  start_time = time.time()
  embedding = get_embedding(content)
  # print(str(embedding))
  elapsed_time = time.time() - start_time
  logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")

  return pd.Series(
    {
      "meta": {
        "published_at": published,
        "url": row["Url"],
      },
      "contents": content,
      "embedding": embedding[0],
      "url": row["Url"],
      "published_at": published,
    }
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


def create_tables():
  engine = create_engine(CONNECTION_STRING)
  Base.metadata.create_all(engine)


def upsert(df: pd.DataFrame):
  engine = create_engine(CONNECTION_STRING)
  Session = sessionmaker(bind=engine)
  session = Session()

  session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

  for i, row in df.iterrows():
    record = Record(**row)
    session.add(record)

  session.commit()
  session.close()


def create_index():
  engine = create_engine(CONNECTION_STRING)
  with engine.connect() as connection:
    connection.execute(
      text(
        # f"CREATE INDEX IF NOT EXISTS idx_{COLLECTION_NAME}_embedding ON {COLLECTION_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        f"CREATE INDEX IF NOT EXISTS idx_{COLLECTION_NAME}_embedding ON {COLLECTION_NAME} USING hnsw (embedding halfvec_l2_ops) WITH (m = 16, ef_construction = 128)"
      )
    )


## ------------------------------------------------
setup_logging()

# Initialize the model
embeddings = init_llm()

# Set up the DB
create_tables()
create_index()

# Read the CSV file and prepare records
df = pd.read_csv("../data/dons.csv", sep=";")
records_df = df.apply(prepare_record, axis=1)

# create_index()  # DiskAnnIndex
upsert(records_df)


# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20)

# files = os.listdir('./corpus')

# for file in files:
#     file_path = f"./corpus/{file}"
#     print(f"Loading: {file_path}")
#     loader = TextLoader(file_path)
#     document = loader.load()
#     texts = text_splitter.split_documents(document)
#     sentence_embeddings = embeddings.embed_documents([t.page_content for t in texts[:5]])

#     db = PGVector.from_documents(
#             embedding=embeddings,
#             documents=texts,
#             collection_name=COLLECTION_NAME,
#             connection_string=CONNECTION_STRING)


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import SKLearnVectorStore
# from langchain_nomic.embeddings import NomicEmbeddings


# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
# ]

# # Load documents
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=1000, chunk_overlap=200
# )
# doc_splits = text_splitter.split_documents(docs_list)

# # Add to vectorDB
# vectorstore = SKLearnVectorStore.from_documents(
#     documents=doc_splits,
#     embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
# )

# # Create retriever
# retriever = vectorstore.as_retriever(k=3)


# text = "..." # your text
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 256,
#     chunk_overlap  = 20
# )

# docs = text_splitter.create_documents([text])