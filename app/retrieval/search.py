from time import perf_counter

from app.embedding.model import LocalEmbeddingModel
from app.retrieval.models import SearchFilters, SearchResponse
from app.retrieval.profiles import RetrievalProfile
from app.retrieval.repository import RetrievalRepository, diversify
from app.retrieval.tracing import RetrievalTracer


class RetrievalSearch:
  def __init__(
    self,
    model: LocalEmbeddingModel,
    repository: RetrievalRepository,
    tracer: RetrievalTracer,
  ):
    self.model = model
    self.repository = repository
    self.tracer = tracer

  def search(
    self,
    query: str,
    profile: RetrievalProfile,
    *,
    limit: int | None = None,
    filters: SearchFilters | None = None,
  ) -> SearchResponse:
    query = query.strip()
    if not query:
      raise ValueError("Search query cannot be empty")
    final_limit = limit or profile.default_limit
    if final_limit <= 0:
      raise ValueError("Search limit must be positive")
    filters = filters or SearchFilters()

    total_started = perf_counter()
    embedding_started = perf_counter()
    query_vector = self.model.embed_query(query)
    embedding_seconds = perf_counter() - embedding_started

    database_started = perf_counter()
    candidates = self.repository.search(
      query_vector,
      profile,
      filters,
      candidate_limit=max(final_limit, final_limit * profile.candidate_multiplier),
    )
    chunks = diversify(
      candidates,
      limit=final_limit,
      max_chunks_per_document=profile.max_chunks_per_document,
    )
    database_seconds = perf_counter() - database_started
    response = SearchResponse(
      profile.name,
      chunks,
      embedding_seconds,
      database_seconds,
      perf_counter() - total_started,
    )
    self.tracer.record(query, profile, filters, response)
    return response
