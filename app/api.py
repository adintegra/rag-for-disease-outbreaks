from flask import Flask, render_template, request, jsonify
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
import os

app = Flask(__name__)

# Initialize the embedding model and vector store
embeddings = OpenAIEmbeddings()
vector_store = None

# PostgreSQL connection settings
CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/vectordb"
COLLECTION_NAME = "your_collection_name"  # Replace with your collection name


def init_vector_store():
  global vector_store
  if vector_store is None:
    # Initialize PGVector
    vector_store = PGVector(
      connection_string=CONNECTION_STRING,
      embedding_function=embeddings,
      collection_name=COLLECTION_NAME,
    )


@app.route("/")
def home():
  return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
  try:
    data = request.json
    query = data.get("query", "")

    if not query:
      return jsonify({"error": "No query provided"}), 400

    # Initialize vector store if not already done
    init_vector_store()

    # Perform similarity search
    docs = vector_store.similarity_search(query, k=3)

    # Format results
    results = []
    for doc in docs:
      results.append({"content": doc.page_content, "metadata": doc.metadata})

    return jsonify({"results": results})

  except Exception as e:
    return jsonify({"error": str(e)}), 500


# ... rest of the code remains the same ...
