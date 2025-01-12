from dotenv import load_dotenv
import os
import logging
import time
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from langchain_nomic import NomicEmbeddings
from langchain_ollama import OllamaEmbeddings

# from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from vector_store import Document, Embedding


MODEL = "all-minilm"


def clean_db():
  """Truncate the embedding table to remove all records."""
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  try:
    session.execute("TRUNCATE TABLE embedding")
    session.commit()
  except Exception as e:
    session.rollback()
  finally:
    session.close()


def init_llm():
  """Initialize the LLM model."""

  # embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

  if MODEL == "llama":
    # global VECTOR_DIMENSIONS
    # VECTOR_DIMENSIONS = 3072
    return OllamaEmbeddings(model="llama3.2")

  elif MODEL == "nomic":
    # global VECTOR_DIMENSIONS
    # VECTOR_DIMENSIONS = 768
    # return NomicEmbeddings(
    #   model="nomic-embed-text-v1.5", inference_mode="local", device="gpu"
    return OllamaEmbeddings(
      model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
    )

  elif MODEL == "all-minilm":
    return OllamaEmbeddings(model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL"))


def get_embedding(content):
  # start_time = time.time()
  # embedding = embeddings.embed_documents(content)
  embedding = embeddings.embed_query(content)
  # elapsed_time = time.time() - start_time
  # logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
  return embedding


def process_and_store_embeddings():
  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  # Retrieve all documents
  documents = session.query(Document).all()

  # Initialize text splitter
  # fixed chunk size, not really useful
  # text_splitter = RecursiveCharacterTextSplitter(
  #   chunk_size=1000, chunk_overlap=200, add_start_index=True
  # )

  text_splitter = SemanticChunker(
    NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local", device="gpu")
  )

  with tqdm(total=len(documents), desc="Embedding documents") as pbar:
    for document in documents:
      # Split the text into chunks
      chunks = text_splitter.create_documents([document.contents])

      if MODEL == "llama":
        for chunk_id, chunk in enumerate(chunks):
          # Get the embedding for each chunk
          embedding = get_embedding(chunk.page_content)

          # Create an Embedding record
          embedding_record = Embedding(
            document_id=document.id,
            chunk_id=chunk_id,
            model=MODEL,
            embedding_256=None,
            embedding_512=None,
            embedding_768=embedding[0],
            embedding_1536=None,
            embedding_3072=None,
            embedding_4096=None,
            embedding_8192=None,
          )

          # Add the embedding record to the session
          session.add(embedding_record)

        # Commit for every document
        session.commit()

      elif MODEL == "nomic":
        # Nomic has a context window of 8192 tokens, our max doc size is around 5000 (no chunking)
        # Time: about 15mins
        embedding = get_embedding(document.contents)

        # print(document.contents)
        # print(embedding.shape)

        # Create an Embedding record
        embedding_record = Embedding(
          document_id=document.id,
          chunk_id=0,
          model=MODEL,
          embedding_256=None,
          embedding_512=None,
          embedding_768=embedding,
          embedding_1536=None,
          embedding_3072=None,
          embedding_4096=None,
          embedding_8192=None,
        )

        # Add the embedding record to the session
        session.add(embedding_record)

        # Commit for every document
        session.commit()

      elif MODEL == "all-minilm":
        # Context window of 256 tokens, our max doc size is around 5000 (no chunking)
        embedding = get_embedding(document.contents)

        # print(document.contents)
        # print(embedding.shape)

        # Create an Embedding record
        embedding_record = Embedding(
          document_id=document.id,
          chunk_id=0,
          model=MODEL,
          embedding_256=None,
          embedding_384=embedding,
          embedding_512=None,
          embedding_768=None,
          embedding_1536=None,
          embedding_3072=None,
          embedding_4096=None,
          embedding_8192=None,
        )

        # Add the embedding record to the session
        session.add(embedding_record)

        # Commit for every document
        session.commit()

      pbar.update(1)

  # Close the session cleanly
  session.close()


def main():
  load_dotenv("../../.env")

  # Logging – conflicts with tqdm
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  # clean_db()
  process_and_store_embeddings()


## ------------------------------------------------------
if __name__ == "__main__":
  # Initialize the model
  embeddings = init_llm()
  main()
