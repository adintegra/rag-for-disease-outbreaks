from datetime import UTC, datetime

import pytest

from app.embedding.model import EmbeddingIdentity
from app.ingestion.chunk_profiles import ChunkProfile
from app.retrieval.models import RetrievedChunk, SearchFilters
from app.retrieval.profiles import RetrievalProfile
from app.retrieval.repository import diversify
from app.retrieval.search import RetrievalSearch


class FakeModel:
  identity = EmbeddingIdentity("model", "v1", 2)

  def __init__(self):
    self.queries = []

  def embed_query(self, query):
    self.queries.append(query)
    return [0.1, 0.2]


class FakeRepository:
  def __init__(self, results):
    self.results = results
    self.calls = []

  def search(self, vector, profile, filters, candidate_limit):
    self.calls.append((vector, profile, filters, candidate_limit))
    return self.results


class FakeTracer:
  def __init__(self):
    self.calls = []

  def record(self, query, profile, filters, response):
    self.calls.append((query, profile, filters, response))


def profile():
  return RetrievalProfile(
    "test",
    "chunks",
    ChunkProfile("chunks", "who_sections", "1", 500, 50),
    "model",
    "v1",
    "cosine",
    2,
    3,
    1,
  )


def result(rank, chunk_id, document_id, distance):
  return RetrievedChunk(
    rank,
    chunk_id,
    document_id,
    f"Document {document_id}",
    "summary",
    "Evidence",
    "https://example.test",
    datetime(2026, 1, 1, tzinfo=UTC),
    distance,
  )


def test_diversify_limits_chunks_per_document_and_reassigns_rank() -> None:
  candidates = [
    result(1, 1, 10, 0.1),
    result(2, 2, 10, 0.2),
    result(3, 3, 20, 0.3),
  ]

  selected = diversify(candidates, limit=2, max_chunks_per_document=1)

  assert [chunk.chunk_id for chunk in selected] == [1, 3]
  assert [chunk.rank for chunk in selected] == [1, 2]


def test_search_embeds_query_and_requests_expanded_candidates() -> None:
  model = FakeModel()
  repository = FakeRepository([result(1, 1, 10, 0.1), result(2, 2, 20, 0.2)])
  tracer = FakeTracer()
  search = RetrievalSearch(model, repository, tracer)
  filters = SearchFilters(sections=frozenset({"summary"}))

  response = search.search("  Ebola in Uganda  ", profile(), filters=filters)

  assert model.queries == ["Ebola in Uganda"]
  assert repository.calls[0][3] == 6
  assert repository.calls[0][2] == filters
  assert len(response.chunks) == 2
  assert tracer.calls[0][0] == "Ebola in Uganda"


def test_search_rejects_empty_query() -> None:
  search = RetrievalSearch(FakeModel(), FakeRepository([]), FakeTracer())

  with pytest.raises(ValueError, match="empty"):
    search.search(" ", profile())
