from flask import Flask, render_template, request, jsonify
from langchain.vectorstores import PGVector
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv

# from langchain_postgres import PGVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.vector_store import Document, Embedding

# from pgvector.sqlalchemy import Vector
from sqlalchemy import select


app = Flask(__name__)

# Initialize the embedding model and vector store
load_dotenv("../.env")
embeddings = OllamaEmbeddings(model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL"))
# llm = OllamaLLM(
#   model="llama3.2", base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0.5
# )
llm = OllamaLLM(model="phi4", base_url=os.getenv("OLLAMA_BASE_URL"))

vector_store = None

# PostgreSQL connection settings
CONNECTION_STRING = "postgresql+psycopg://postgres:password@localhost:5432"
COLLECTION_NAME = "embedding"  # Replace with your collection name


def init_vector_store():
  global vector_store
  if vector_store is None:
    # Initialize PGVector
    vector_store = PGVector(
      connection_string=CONNECTION_STRING,
      embedding_function=embeddings,
      collection_name=COLLECTION_NAME,
    )


def similarity_search(query):
  """Search for documents directly using pgvector."""
  # langchain_search(query)
  query_vector = embeddings.embed_query(query)

  engine = create_engine(CONNECTION_STRING)
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

  response = llm.invoke(prompt_text)

  formatted_response = response
  formatted_response += "<br /><br />- - - - - - - - - - - -<br />Here are the retrieved documents most relevant to the query:<br /><br />"

  for doc in retrieved_docs:
    formatted_response += f"<a href='{doc.url}' target='_blank'>Document</a>"
    formatted_response += f"<p>Cosine Similarity: {doc.l2_distance}</p>"
    formatted_response += f"<p>Content: {doc.contents[:350]}...</p>"

  return formatted_response


@app.route("/")
def home():
  return render_template("./index.html")


@app.route("/search", methods=["POST"])
def search():
  try:
    data = request.json
    query = data.get("query", "")

    if not query:
      return jsonify({"error": "No query provided"}), 400

    # Initialize vector store if not already done
    # init_vector_store()

    # Perform similarity search
    docs = similarity_search(query)

    # Format results
    # for doc in docs:
    #   results.append({"content": doc.page_content, "metadata": doc.metadata})

    return jsonify({"response": docs})

  except Exception as e:
    return jsonify({"error": str(e)}), 500
