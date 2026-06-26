import logging

from flask import Flask, jsonify, render_template, request

from core import config
from core.rag import generate_answer, similarity_search

app = Flask(__name__)


def _check_api_key():
  if not config.api_key or config.api_key == "CHANGEME":
    return None
  provided = request.headers.get("X-API-Key") or (request.json or {}).get("api_key")
  if provided != config.api_key:
    return jsonify({"error": "Unauthorized"}), 401
  return None


@app.route("/")
def home():
  return render_template("./index.html", api_key=config.api_key)


@app.route("/search", methods=["POST"])
def search():
  blocked = _check_api_key()
  if blocked:
    return blocked

  try:
    query = (request.json or {}).get("query", "").strip()
    if not query:
      return jsonify({"error": "No query provided"}), 400

    docs = similarity_search(query)
    answer = generate_answer(query, docs)
    sources = [
      {
        "url": doc.url,
        "similarity": round(doc.similarity, 4),
        "excerpt": doc.contents[:350] + ("..." if len(doc.contents) > 350 else ""),
      }
      for doc in docs
    ]
    return jsonify({"answer": answer, "sources": sources})

  except ValueError as exc:
    return jsonify({"error": str(exc)}), 404
  except Exception:
    app.logger.exception("Search request failed")
    return jsonify({"error": "An internal error occurred"}), 500


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  app.run()
