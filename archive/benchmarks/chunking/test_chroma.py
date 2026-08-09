from dotenv import load_dotenv
import os
import sys

# Required to import from parent directory
sys.path.append("../../")

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def test_chroma_connectivity():
  try:
    embeddings = OllamaEmbeddings(
      model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL")
    )

    vector_store = Chroma(
      embedding_function=embeddings, persist_directory="./vector_store"
    )

    # Get default collection
    results = vector_store.similarity_search_with_score("Where is Ebola prevalent?")

    print("Successfully connected to Chroma")

    for res, score in results:
      print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}\n")

    print(f"Number of documents found: {len(results)}")

    return True
  except Exception as e:
    print(f"Failed to connect to Chroma: {str(e)}")
    return False


if __name__ == "__main__":
  load_dotenv()
  test_chroma_connectivity()
