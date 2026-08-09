from dotenv import load_dotenv
import os
import sys

# Required to import from parent directory
sys.path.append("../../../")

import numpy as np
import matplotlib.pyplot as plt
import PostgresDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector, EmbeddingIndexType
from langchain_chroma import Chroma


CHUNK_SIZE = 2048


# Function to inspect chunk lengths in matplotlit.
def plot_chunk_lengths(chunked_docs, title_keyword):
  # Get chunk lengths.
  lengths = [len(doc.page_content) for doc in chunked_docs]

  # Mean, median lengths.
  mean_length = np.mean(lengths)
  median_length = np.median(lengths)

  # Assemble the title.
  title = f"Chunk Lengths from {title_keyword} Chunking"

  # Plot the lengths.
  plt.figure(figsize=(10, 6))  # Adjust figure size
  plt.plot(lengths, marker="o")  # Plot lengths with circle markers
  plt.title(title, fontsize=20, fontweight="bold")
  plt.xlabel("Document Index")  # X-axis label
  plt.ylabel("Length")  # Y-axis label
  plt.grid(True)  # Show grid

  # Add a horizontal line at mean and median length
  plt.axhline(y=mean_length, color="g", linestyle="-")
  plt.axhline(y=median_length, color="r", linestyle="-")
  plt.text(
    len(lengths) - 1,
    mean_length,
    f"mean = {mean_length:.0f}",
    va="center",
    ha="left",
    backgroundcolor="w",
    fontsize=12,
  )
  plt.text(
    0,
    median_length,
    f"median = {median_length:.0f}",
    va="center",
    ha="right",
    backgroundcolor="w",
    fontsize=12,
  )

  plt.show()  # Display the plot


def split_docs(docs):
  """Split documents into chunks."""
  chunk_overlap = int(round(CHUNK_SIZE * 0.10, 0))

  print(f"Loaded {len(docs)} documents")
  print(f"Chunk size: {CHUNK_SIZE}, Chunk Overlap: {chunk_overlap}")

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=chunk_overlap,
    add_start_index=True,
  )
  all_splits = text_splitter.split_documents(docs)

  print(f"{len(docs)} docs split into {len(all_splits)} sub-documents.")
  plot_chunk_lengths(all_splits, "Recursive Character")

  return all_splits


def index(add_embeddings=True):
  # Init the embeddings model: context for nomic 2048
  embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
  )

  # Init the vector store
  # vector_store = PGVector(
  #   embeddings=embeddings,
  #   collection_name="dons",
  #   connection=os.getenv("CONNECTION_STRING") + "/postgres",
  #   embedding_length=768,
  #   pre_delete_collection=True,
  # )

  vector_store = Chroma(
    embedding_function=embeddings, persist_directory="./vector_store"
  )

  # embedding_index=EmbeddingIndexType.hnsw,
  # embedding_index_ops={"m": 16, "efConstruction": 128},
  # embedding_index_ops={"m": 16, "efConstruction": 128, "efSearch": 128},
  # Load documents from Postgres
  loader = PostgresDocumentLoader.PostgresDocumentLoader()
  documents = loader.load()

  # Split documents into chunks
  all_splits = split_docs(documents)

  # Add documents to the vector store
  if add_embeddings:
    document_ids = vector_store.add_documents(documents=all_splits)

    print(document_ids[:3])
    print("Created vector store with chunks:", len(document_ids))


def main():
  # Load environment variables
  load_dotenv()

  index(add_embeddings=False)


if __name__ == "__main__":
  main()
