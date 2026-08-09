from datetime import UTC, datetime

from app.generation.service import AnswerService
from app.retrieval.models import RetrievedChunk, SearchResponse


class FakeSearch:
  def search(self, query, profile, limit=None):
    chunks = [
      RetrievedChunk(
        1,
        1,
        1,
        "Ebola in Uganda",
        "summary",
        "Evidence text",
        "https://example.test",
        datetime(2026, 1, 1, tzinfo=UTC),
        0.1,
      )
    ]
    return SearchResponse("sections", chunks, 0.01, 0.02, 0.03)


class FakeChat:
  def __init__(self):
    self.messages = None

  def invoke(self, messages):
    self.messages = messages
    return "Uganda reported Ebola [1]."


def test_answer_service_returns_only_prompted_sources() -> None:
  chat = FakeChat()
  service = AnswerService(FakeSearch(), chat, max_context_characters=2000)

  result = service.answer("What happened?", object())

  assert result.answer == "Uganda reported Ebola [1]."
  assert [chunk.chunk_id for chunk in result.search.chunks] == [1]
  assert result.context_characters > 0
  assert chat.messages is not None
