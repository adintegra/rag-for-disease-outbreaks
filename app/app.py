import hmac
import logging
from functools import lru_cache
from threading import BoundedSemaphore

from flask import Flask, jsonify, render_template, request

from app.core.config import Settings, get_settings
from app.core.database import get_session_factory
from app.embedding.model import LocalEmbeddingModel
from app.generation.model import LocalChatModel
from app.generation.service import AnswerService
from app.retrieval.profiles import get_retrieval_profile
from app.retrieval.repository import RetrievalRepository
from app.retrieval.search import RetrievalSearch
from app.retrieval.tracing import RetrievalTracer


@lru_cache
def build_answer_service() -> AnswerService:
  settings = get_settings()
  search = RetrievalSearch(
    LocalEmbeddingModel.from_settings(settings),
    RetrievalRepository(get_session_factory()),
    RetrievalTracer(
      enabled=settings.langsmith_tracing,
      project_name=settings.langsmith_project,
      trace_query=settings.langsmith_trace_query,
    ),
  )
  return AnswerService(
    search,
    LocalChatModel.from_settings(settings),
    max_context_characters=settings.rag_context_max_characters,
  )


def create_app(
  *,
  settings: Settings | None = None,
  answer_service: AnswerService | None = None,
) -> Flask:
  settings = settings or get_settings()
  if settings.require_api_key and not settings.api_key:
    raise ValueError("API_KEY is required when REQUIRE_API_KEY is enabled")

  flask_app = Flask(__name__)
  flask_app.config["MAX_CONTENT_LENGTH"] = settings.http_max_body_bytes
  generation_slots = BoundedSemaphore(settings.max_concurrent_generations)

  @flask_app.get("/")
  def home():
    return render_template("index.html")

  @flask_app.post("/search")
  def search():
    blocked = _check_api_key(settings)
    if blocked is not None:
      return blocked
    if not request.is_json:
      return jsonify({"error": "Content-Type must be application/json"}), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
      return jsonify({"error": "Request body must be a JSON object"}), 400
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
      return jsonify({"error": "A non-empty query is required"}), 400
    query = query.strip()
    if len(query) > settings.rag_query_max_characters:
      return jsonify({"error": "Query is too long"}), 400

    profile_name = payload.get("profile", settings.rag_retrieval_profile)
    if not isinstance(profile_name, str):
      return jsonify({"error": "Profile must be a string"}), 400
    limit = payload.get("limit")
    if limit is not None and (
      not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20
    ):
      return jsonify({"error": "Limit must be an integer between 1 and 20"}), 400

    if not generation_slots.acquire(blocking=False):
      return jsonify({"error": "Generation capacity is busy; retry later"}), 503
    try:
      profile = get_retrieval_profile(profile_name)
      service = answer_service or build_answer_service()
      result = service.answer(query, profile, limit=limit)
      sources = [
        {
          "citation": index,
          "chunk_id": chunk.chunk_id,
          "document_id": chunk.document_id,
          "title": chunk.title,
          "section": chunk.section,
          "published_at": chunk.published_at.isoformat(),
          "url": chunk.url,
          "similarity": round(chunk.similarity, 4),
          "excerpt": chunk.contents[:350]
          + ("..." if len(chunk.contents) > 350 else ""),
        }
        for index, chunk in enumerate(result.search.chunks, start=1)
      ]
      return jsonify(
        {
          "answer": result.answer,
          "profile": result.search.profile_name,
          "sources": sources,
          "timings": {
            "query_embedding_seconds": result.search.query_embedding_seconds,
            "database_seconds": result.search.database_seconds,
            "retrieval_seconds": result.search.total_seconds,
            "generation_seconds": result.generation_seconds,
          },
        }
      )
    except ValueError as error:
      return jsonify({"error": str(error)}), 400
    except Exception:
      flask_app.logger.exception("Search request failed")
      return jsonify({"error": "An internal error occurred"}), 500
    finally:
      generation_slots.release()

  return flask_app


def _check_api_key(settings: Settings):
  if not settings.require_api_key:
    return None
  provided = request.headers.get("X-API-Key", "")
  if not hmac.compare_digest(provided, settings.api_key or ""):
    return jsonify({"error": "Unauthorized"}), 401
  return None


app = create_app()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  app.run()
