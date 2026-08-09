from datetime import UTC, datetime

from app.app import create_app
from app.core.config import Settings
from app.generation.service import GeneratedAnswer
from app.retrieval.models import RetrievedChunk, SearchResponse


class FakeAnswerService:
  def answer(self, query, profile, limit=None):
    chunk = RetrievedChunk(
      1,
      42,
      7,
      "Ebola in Uganda",
      "summary",
      "Relevant evidence",
      "https://example.test/don",
      datetime(2026, 1, 1, tzinfo=UTC),
      0.2,
    )
    search = SearchResponse("sections", [chunk], 0.01, 0.02, 0.03)
    return GeneratedAnswer("Answer with citation [1].", search, 0.04, 100)


def settings(**overrides):
  values = {
    "DATABASE_URL": "postgresql://user:secret@example.test/database",
    "REQUIRE_API_KEY": False,
    "RAG_QUERY_MAX_CHARACTERS": 100,
  }
  values.update(overrides)
  return Settings(**values)


def test_home_does_not_render_api_key() -> None:
  app = create_app(
    settings=settings(API_KEY="server-secret"), answer_service=FakeAnswerService()
  )

  response = app.test_client().get("/")

  assert response.status_code == 200
  assert b"server-secret" not in response.data


def test_search_returns_cited_chunk_sources() -> None:
  app = create_app(settings=settings(), answer_service=FakeAnswerService())

  response = app.test_client().post("/search", json={"query": "Ebola?"})

  assert response.status_code == 200
  body = response.get_json()
  assert body["answer"] == "Answer with citation [1]."
  assert body["sources"][0]["citation"] == 1
  assert body["sources"][0]["chunk_id"] == 42
  assert body["sources"][0]["title"] == "Ebola in Uganda"


def test_search_rejects_non_json_and_long_queries() -> None:
  app = create_app(settings=settings(), answer_service=FakeAnswerService())
  client = app.test_client()

  assert client.post("/search", data="query").status_code == 415
  assert client.post("/search", json={"query": "x" * 101}).status_code == 400


def test_api_key_is_header_only_and_fails_closed_when_required() -> None:
  app = create_app(
    settings=settings(REQUIRE_API_KEY=True, API_KEY="server-secret"),
    answer_service=FakeAnswerService(),
  )
  client = app.test_client()

  assert client.post("/search", json={"query": "Ebola?"}).status_code == 401
  assert (
    client.post(
      "/search", json={"query": "Ebola?", "api_key": "server-secret"}
    ).status_code
    == 401
  )
  response = client.post(
    "/search",
    json={"query": "Ebola?", "api_key": "server-secret"},
    headers={"X-API-Key": "server-secret"},
  )
  assert response.status_code == 200


def test_required_authentication_without_key_fails_at_startup() -> None:
  try:
    create_app(
      settings=settings(REQUIRE_API_KEY=True, API_KEY=None),
      answer_service=FakeAnswerService(),
    )
  except ValueError as error:
    assert "API_KEY" in str(error)
  else:
    raise AssertionError("Expected startup configuration error")
