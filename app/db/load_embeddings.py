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
from vector_store import Document, Embedding


# Run this script to generate embeddings for all documents in the database
# MODEL = "all-minilm"  # dim 384 / context window 512
MODEL = "nomic-embed-text"  # dim 768 / context window 2048
# MODEL = "mxbai-embed-large"  # dim 1024 / context window 512

embeddings = ""


class EmbeddingModel:
  def __init__(self, name=MODEL):
    self.name = MODEL
    self.dims = (
      384
      if name == "all-minilm"
      else 768
      if name == "nomic-embed-text"
      else 1024
      if name == "mxbai-embed-large"
      else 0
    )
    self.ctx = (
      512
      if name == "all-minilm"
      else 2048
      if name == "nomic-embed-text"
      else 512
      if name == "mxbai-embed-large"
      else 0
    )


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


def init_llm(model):
  """Initialize the LLM model."""
  return OllamaEmbeddings(
    model=model.name, base_url=os.getenv("OLLAMA_BASE_URL"), num_ctx=model.ctx
  )


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

  llm = EmbeddingModel()

  # Initialize text splitter
  # fixed chunk size, not really useful
  # text_splitter = RecursiveCharacterTextSplitter(
  #   chunk_size=1000, chunk_overlap=200, add_start_index=True
  # )
  # text_splitter = SemanticChunker(
  #   NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local", device="gpu")
  # )

  with tqdm(total=len(documents), desc="Embedding documents") as pbar:
    for document in documents:
      # Split the text into chunks
      # chunks = text_splitter.create_documents([document.contents])

      # if MODEL == "llama":
      #   for chunk_id, chunk in enumerate(chunks):
      #     # Get the embedding for each chunk
      #     embedding = get_embedding(chunk.page_content)

      #     # Create an Embedding record
      #     embedding_record = Embedding(
      #       document_id=document.id,
      #       chunk_id=chunk_id,
      #       model=MODEL,
      #       embedding_256=None,
      #       embedding_512=None,
      #       embedding_768=embedding[0],
      #       embedding_1536=None,
      #       embedding_3072=None,
      #       embedding_4096=None,
      #       embedding_8192=None,
      #     )

      #     # Add the embedding record to the session
      #     session.add(embedding_record)

      #   # Commit for every document
      #   session.commit()

      if llm.dims == 768:
        # Time: about 15mins
        embedding = get_embedding(document.contents)

        # Create an Embedding record
        embedding_record = Embedding(
          document_id=document.id,
          chunk_id=0,
          model=llm.name,
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

      elif llm.dims == 384:
        embedding = get_embedding(document.contents)

        # print(document.contents)
        # print(embedding.shape)

        # Create an Embedding record
        embedding_record = Embedding(
          document_id=document.id,
          chunk_id=0,
          model=llm.name,
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

  # Initialize the model
  em = EmbeddingModel(MODEL)

  global embeddings
  embeddings = init_llm(em)

  # Logging – conflicts with tqdm
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  # clean_db()
  process_and_store_embeddings()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
