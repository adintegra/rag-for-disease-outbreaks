import langsmith as ls

from app.retrieval.models import SearchFilters, SearchResponse
from app.retrieval.profiles import RetrievalProfile


class RetrievalTracer:
  def __init__(
    self,
    *,
    enabled: bool,
    project_name: str,
    trace_query: bool,
  ):
    self.enabled = enabled
    self.project_name = project_name
    self.trace_query = trace_query

  def record(
    self,
    query: str,
    profile: RetrievalProfile,
    filters: SearchFilters,
    response: SearchResponse,
  ) -> None:
    if not self.enabled:
      return

    inputs = {
      "query_length": len(query),
      "profile": profile.name,
      "configuration_hash": profile.chunk_profile.configuration_hash,
      "published_after": str(filters.published_after or ""),
      "published_before": str(filters.published_before or ""),
      "sections": sorted(filters.sections or []),
    }
    if self.trace_query:
      inputs["query"] = query

    try:
      with (
        ls.tracing_context(enabled=True, project_name=self.project_name),
        ls.trace(
          "retrieve-don-chunks",
          run_type="retriever",
          inputs=inputs,
          tags=["retrieval", "don", f"profile:{profile.name}"],
          metadata={
            "embedding_model": profile.embedding_model,
            "embedding_model_version": profile.embedding_model_version,
            "distance": profile.distance,
            "query_embedding_seconds": response.query_embedding_seconds,
            "database_seconds": response.database_seconds,
          },
        ) as run,
      ):
        run.end(
          outputs={
            "chunk_ids": [chunk.chunk_id for chunk in response.chunks],
            "document_ids": [chunk.document_id for chunk in response.chunks],
            "distances": [chunk.distance for chunk in response.chunks],
            "urls": [chunk.url for chunk in response.chunks],
          }
        )
    except Exception:  # noqa: BLE001
      # Retrieval results remain authoritative when observability is unavailable.
      return
