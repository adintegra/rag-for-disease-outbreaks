from dotenv import load_dotenv
import os
import logging

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from vector_store import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def init_vector_store():
  embeddings = OllamaEmbeddings(
    model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL")
  )

  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  docs = [doc.contents for doc in session.query(Document).all()]

  session.close()

  # text_splitter = RecursiveCharacterTextSplitter(
  #   chunk_size=500,
  #   chunk_overlap=50,
  #   length_function=len,
  # )
  # split_documents = []
  # for doc in documents:
  #   chunks = text_splitter.split_text(doc)
  #   split_documents.extend(chunks)

  vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    connection=os.getenv("CONNECTION_STRING"),
    use_jsonb=True,
  )

  return vectorstore


def main():
  load_dotenv("../../.env")

  # Logging
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  init_vector_store()


## ------------------------------------------------------
if __name__ == "__main__":
  main()
