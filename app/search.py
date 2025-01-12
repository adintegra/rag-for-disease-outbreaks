from dotenv import load_dotenv
import os
import sys
import logging

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_postgres import PGVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.vector_store import Document, Embedding
from pgvector.sqlalchemy import Vector
from sqlalchemy import select


# TODO: Finish
def init_vector_store():
  """Use langChain to search for documents based on a query (unfinished)."""
  embeddings = OllamaEmbeddings(
    model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL")
  )

  # Ollama - should not use LLM model for embeddings
  # embeddings = OllamaEmbeddings(model="llama3.2:1b")

  vectorstore = PGVector.from_existing_index(
    embedding=embeddings,
    connection=os.getenv("CONNECTION_STRING"),
    use_jsonb=True,
  )

  return vectorstore


def search(query):
  """Search for documents directly using pgvector."""
  # langchain_search(query)

  embeddings = OllamaEmbeddings(
    model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL")
  )

  query_vector = embeddings.embed_query(query)

  engine = create_engine(os.getenv("CONNECTION_STRING"))
  Session = sessionmaker(bind=engine)
  session = Session()

  # See https://github.com/pgvector/pgvector-python?tab=readme-ov-file#sqlalchemy
  # l2_distance also works
  stmt = (
    select(
      Document.id,
      Document.contents,
      Document.url,
      Embedding.embedding_384.cosine_distance(query_vector).label("l2_distance"),
    )
    .join(Embedding)
    .filter(Embedding.model == "all-minilm")
    .order_by(Embedding.embedding_384.cosine_distance(query_vector))
    .limit(5)
  )

  # print(stmt)
  # retrieved_docs = session.execute(stmt).scalars().all()
  retrieved_docs = session.execute(stmt).all()

  # if len(retrieved_docs) == 0:
  #   print("No documents found.")
  #   return
  # else:
  #   print(f"Number of documents retrieved: {len(retrieved_docs)}")
  #   # for doc in retrieved_docs:
  #   #   print(f"* {doc.id}")
  #   #   print(f"URL: {doc.url}")
  #   #   print(f"Similarity: {doc.l2_distance}")

  session.close()

  # Create a prompt that combines the query and retrieved documents
  prompt_text = f"""Answer the following question based on the provided context:

Question: {query}

Context:
"""
  for doc in retrieved_docs:
    prompt_text += f"\n{doc.contents}\n"

  prompt_text += "\nAnswer:"

  # for doc in retrieved_docs:
  #   print(f"*** {doc.page_content}")
  #   print(f"** [{doc.metadata}]")
  return prompt_text


# TODO: Similarity search using langchain
def langchain_search(query):
  """Use langChain to search for documents based on a query (unfinished)."""
  vectorstore = init_vector_store()

  retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
  retrieved_docs = retriever.invoke(query)

  print(f"Number of documents retrieved: {len(retrieved_docs)}")
  # print(f"* {retrieved_docs[0].page_content}")
  # print(f"* [{retrieved_docs[0].metadata}]")

  # Create a prompt that combines the query and retrieved documents
  prompt_text = f"""Answer the following question based on the provided context:

Question: {query}

Context:
"""
  for doc in retrieved_docs:
    prompt_text += f"\n{doc.page_content}\n"

  prompt_text += "\nAnswer:"

  # for doc in retrieved_docs:
  #   print(f"*** {doc.page_content}")
  #   print(f"** [{doc.metadata}]")
  return prompt_text


def prompt(p):
  """Send the prompt to the LLM."""
  llm = OllamaLLM(
    model="llama3.2", base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0.5
  )
  response = llm.invoke(p)
  print(f"\nLLM Response:\n{response}\n- - - - - - -\n")
  return response


def main():
  # Check for command line arguments
  if len(sys.argv) < 2:
    logging.warning("Please provide a search query as a command line argument")
    sys.exit(1)

  load_dotenv("../.env")

  # Logging
  if os.getenv("LOGS"):
    logging.basicConfig(
      level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

  # q = "In which countries is Malaria most prevalent?"
  # q = "Where were the largest outbreaks of Malaria in 2020?"

  query = " ".join(sys.argv[1:])

  print(f"\n- - - - - - -\nSearch query (with context): {query}")

  p = search(query)
  prompt(p)

  print("\n- - - - - - -\nHere is the same query without providing additional context:")
  prompt(query)

  # # embeddings = OllamaEmbeddings(model="llama3.2")
  # embeddings = NomicEmbeddings(
  #   model="nomic-embed-text-v1.5", inference_mode="local", device="gpu"
  # )

  # vector_store = PGVector(
  #   embeddings=embeddings,
  #   collection_name="langchain",
  #   connection=os.getenv("CONNECTION_STRING"),
  #   use_jsonb=True,
  # )

  # docs = [
  #   Document(
  #     page_content="there are cats in the pond",
  #     metadata={"id": 1, "location": "pond", "topic": "animals"},
  #   ),
  #   Document(
  #     page_content="fresh apples are available at the market",
  #     metadata={"id": 3, "location": "market", "topic": "food"},
  #   ),
  #   Document(
  #     page_content="the market also sells fresh oranges",
  #     metadata={"id": 4, "location": "market", "topic": "food"},
  #   ),
  #   Document(
  #     page_content="the new art exhibit is fascinating",
  #     metadata={"id": 5, "location": "museum", "topic": "art"},
  #   ),
  #   Document(
  #     page_content="a sculpture exhibit is also at the museum",
  #     metadata={"id": 6, "location": "museum", "topic": "art"},
  #   ),
  #   Document(
  #     page_content="a new coffee shop opened on Main Street",
  #     metadata={"id": 7, "location": "Main Street", "topic": "food"},
  #   ),
  #   Document(
  #     page_content="the book club meets at the library",
  #     metadata={"id": 8, "location": "library", "topic": "reading"},
  #   ),
  # ]
  # vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

  # retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
  # retriever.invoke("malaria")


## ------------------------------------------------------
if __name__ == "__main__":
  main()
