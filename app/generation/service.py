from dataclasses import dataclass
from time import perf_counter

from app.generation.model import LocalChatModel
from app.generation.prompt import build_messages
from app.retrieval.models import SearchResponse


@dataclass(frozen=True)
class GeneratedAnswer:
  answer: str
  search: SearchResponse
  generation_seconds: float
  context_characters: int


class AnswerService:
  def __init__(
    self,
    search,
    chat: LocalChatModel,
    *,
    max_context_characters: int,
  ):
    self.search = search
    self.chat = chat
    self.max_context_characters = max_context_characters

  def answer(self, query: str, profile, *, limit: int | None = None) -> GeneratedAnswer:
    search_response = self.search.search(query, profile, limit=limit)
    prompt = build_messages(
      query,
      search_response.chunks,
      max_context_characters=self.max_context_characters,
    )
    started = perf_counter()
    answer = self.chat.invoke(prompt.messages)
    return GeneratedAnswer(
      answer,
      SearchResponse(
        search_response.profile_name,
        prompt.chunks,
        search_response.query_embedding_seconds,
        search_response.database_seconds,
        search_response.total_seconds,
      ),
      perf_counter() - started,
      prompt.characters,
    )
