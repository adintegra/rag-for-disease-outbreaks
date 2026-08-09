import argparse
import logging

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.embedding.model import LocalEmbeddingModel
from app.generation.model import LocalChatModel
from app.generation.service import AnswerService
from app.retrieval.profiles import get_retrieval_profile
from app.retrieval.repository import RetrievalRepository
from app.retrieval.search import RetrievalSearch
from app.retrieval.tracing import RetrievalTracer


def main() -> None:
  parser = argparse.ArgumentParser(description="Answer a question using WHO DON evidence")
  parser.add_argument("query")
  parser.add_argument("--profile")
  parser.add_argument("--limit", type=int)
  args = parser.parse_args()
  settings = get_settings()
  logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  profile = get_retrieval_profile(args.profile or settings.rag_retrieval_profile)
  search = RetrievalSearch(
    LocalEmbeddingModel.from_settings(settings),
    RetrievalRepository(get_session_factory()),
    RetrievalTracer(
      enabled=settings.langsmith_tracing,
      project_name=settings.langsmith_project,
      trace_query=settings.langsmith_trace_query,
    ),
  )
  service = AnswerService(
    search,
    LocalChatModel.from_settings(settings),
    max_context_characters=settings.rag_context_max_characters,
  )
  result = service.answer(args.query, profile, limit=args.limit)
  print(result.answer)
  print("\nSources:")
  for index, chunk in enumerate(result.search.chunks, start=1):
    print(f"[{index}] {chunk.title} — {chunk.url}")
