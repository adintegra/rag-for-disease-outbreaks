import argparse
import logging
from datetime import UTC, datetime

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.embedding.model import LocalEmbeddingModel
from app.retrieval.models import SearchFilters
from app.retrieval.profiles import get_retrieval_profile, load_retrieval_profiles
from app.retrieval.repository import RetrievalRepository
from app.retrieval.search import RetrievalSearch
from app.retrieval.tracing import RetrievalTracer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Search indexed WHO DON chunks")
  subparsers = parser.add_subparsers(dest="command", required=True)
  subparsers.add_parser("profiles", help="List configured retrieval profiles")
  search = subparsers.add_parser("search", help="Run profile-aware cosine retrieval")
  search.add_argument("query")
  search.add_argument("--profile", default="sections")
  search.add_argument("--limit", type=int)
  search.add_argument("--published-after", type=_datetime)
  search.add_argument("--published-before", type=_datetime)
  search.add_argument("--section", action="append", dest="sections")
  search.add_argument("--source")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  settings = get_settings()
  logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

  if args.command == "profiles":
    for profile in load_retrieval_profiles().values():
      print(
        f"{profile.name}: chunks={profile.chunk_profile_name} "
        f"model={profile.embedding_model}@{profile.embedding_model_version} "
        f"distance={profile.distance} limit={profile.default_limit}"
      )
    return

  profile = get_retrieval_profile(args.profile)
  model = LocalEmbeddingModel.from_settings(settings)
  if (
    model.identity.model != profile.embedding_model
    or model.identity.model_version != profile.embedding_model_version
  ):
    raise ValueError(
      "Configured query embedding model does not match the retrieval profile"
    )
  search = RetrievalSearch(
    model,
    RetrievalRepository(get_session_factory()),
    RetrievalTracer(
      enabled=settings.langsmith_tracing,
      project_name=settings.langsmith_project,
      trace_query=settings.langsmith_trace_query,
    ),
  )
  response = search.search(
    args.query,
    profile,
    limit=args.limit,
    filters=SearchFilters(
      published_after=args.published_after,
      published_before=args.published_before,
      sections=frozenset(args.sections) if args.sections else None,
      source=args.source,
    ),
  )
  print(
    f"profile={response.profile_name} results={len(response.chunks)} "
    f"embedding_ms={response.query_embedding_seconds * 1000:.1f} "
    f"database_ms={response.database_seconds * 1000:.1f} "
    f"total_ms={response.total_seconds * 1000:.1f}"
  )
  for chunk in response.chunks:
    excerpt = " ".join(chunk.contents.split())[:240]
    print(
      f"\n{chunk.rank}. similarity={chunk.similarity:.4f} "
      f"document={chunk.document_id} chunk={chunk.chunk_id} "
      f"section={chunk.section or 'unknown'} published={chunk.published_at.date()}"
    )
    print(chunk.title)
    print(chunk.url)
    print(excerpt)


def _datetime(value: str) -> datetime:
  parsed = datetime.fromisoformat(value)
  return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
