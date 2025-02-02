from flask import Flask, render_template, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import time
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sqlalchemy import select
from db.vector_store import Document, Embedding


load_dotenv("../.env")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("CONNECTION_STRING") + "/postgres"
app.config["SQLALCHEMY_POOL_SIZE"] = 10  # Set the pool size
app.config["SQLALCHEMY_MAX_OVERFLOW"] = 5  # Allow additional connections

db = SQLAlchemy(app)

# Initialize the embedding models and vector store
bert_embeddings = OllamaEmbeddings(
  model="all-minilm", base_url=os.getenv("OLLAMA_BASE_URL")
)
nomic_embeddings = OllamaEmbeddings(
  model="nomic-text-embed-v1.5", base_url=os.getenv("OLLAMA_BASE_URL")
)
embeddings = bert_embeddings

llm = OllamaLLM(model=os.getenv("LLM"), base_url=os.getenv("OLLAMA_BASE_URL"))
# llm = OllamaLLM(
#   model="llama3.2", base_url=os.getenv("OLLAMA_BASE_URL"), temperature=0.5
# )


def similarity_search(query):
  """Search for documents directly using pgvector."""
  g.search_starttime = time.time()
  app.logger.info(f"{time.time() - g.search_starttime}s: Embedding query vector...")
  app.logger.info(f"Using model: {embeddings.model}")

  query_vector = embeddings.embed_query(query)

  app.logger.info(f"{time.time() - g.search_starttime}s: Querying vector store...")

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
    .filter(Embedding.model == embeddings.model)
    .order_by(Embedding.embedding_384.cosine_distance(query_vector))
    .limit(5)
  )

  retrieved_docs = db.session.execute(stmt).all()

  if len(retrieved_docs) == 0:
    raise ValueError("No documents found.")

  # else:
  #   print(f"Number of documents retrieved: {len(retrieved_docs)}")
  #   # for doc in retrieved_docs:
  #   #   print(f"* {doc.id}")
  #   #   print(f"URL: {doc.url}")
  #   #   print(f"Similarity: {doc.l2_distance}")

  app.logger.info(f"{time.time() - g.search_starttime}s: Querying LLM...")

  #  Query the LLM model to generate a response
  formatted_response = query_llm(retrieved_docs, query)

  app.logger.info(
    f"{time.time() - g.pop('search_starttime', None)}s: Sending response to browser..."
  )

  return formatted_response


def query_llm(docs, query):
  """Formulate an LLM prompt providing context from our knowledgebase."""

  # Create a prompt that combines the query and retrieved documents
  prompt_text = f"""Answer the following question based on the provided context:

Question: {query}

Context:
"""
  for doc in docs:
    prompt_text += f"\n{doc.contents}\n"

  prompt_text += "\nAnswer:"

  response = llm.invoke(prompt_text)

  formatted_response = response
  formatted_response += "<br /><br />- - - - - - - - - - - -<br />Here are the retrieved documents most relevant to the query:<br /><br />"

  for doc in docs:
    formatted_response += f"<a href='{doc.url}' target='_blank'>Document</a>"
    formatted_response += f"<p>Cosine Similarity: {doc.l2_distance}</p>"
    formatted_response += f"<p>Content: {doc.contents[:350]}...</p>"

  return formatted_response


@app.before_request
def log_route_start():
  g.start_time = time.time()


@app.after_request
def log_route_end(response):
  route = request.endpoint
  print(f"{route} processed in {time.time() - g.pop('start_time', None)}s")
  return response


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

    # Perform similarity search
    docs = similarity_search(query)

    # Format results
    # for doc in docs:
    #   results.append({"content": doc.page_content, "metadata": doc.metadata})

    return jsonify({"response": docs})

  except Exception as e:
    return jsonify({"error": str(e)}), 500
