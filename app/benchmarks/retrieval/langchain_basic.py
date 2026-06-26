from dotenv import load_dotenv
import os
import sys

# Required to import from parent directory
sys.path.append("../../../")

import ragas.PostgresDocumentLoader as PostgresDocumentLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def load_documents():
  loader = PostgresDocumentLoader.PostgresDocumentLoader()
  documents = loader.load()
  return documents


def main():
  # Load environment variables
  load_dotenv()

  embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
  )

  vector_store = Chroma(
    embedding_function=embeddings, persist_directory="./vector_store"
  )

  # Load documents from the URLs
  docs = load_documents()
  docs_list = [item for sublist in docs for item in sublist]

  # # Initialize a text splitter with specified chunk size and overlap
  # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
  #   chunk_size=250, chunk_overlap=0
  # )

  # # Split the documents into chunks
  # doc_splits = text_splitter.split_documents(docs_list)

  # # Add the document chunks to the "vector store" using OpenAIEmbeddings
  # vectorstore = InMemoryVectorStore.from_documents(
  #   documents=doc_splits,
  #   embedding=OpenAIEmbeddings(),
  # )

  # With langchain we can easily turn any vector store into a retrieval component:
  retriever = vector_store.as_retriever(k=6)


if __name__ == "__main__":
  main()
