from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RetrievedChunk:
  rank: int
  chunk_id: int
  document_id: int
  title: str
  section: str | None
  contents: str
  url: str
  published_at: datetime
  distance: float

  @property
  def similarity(self) -> float:
    return 1.0 - self.distance


@dataclass(frozen=True)
class SearchFilters:
  published_after: datetime | None = None
  published_before: datetime | None = None
  sections: frozenset[str] | None = None
  source: str | None = None


@dataclass(frozen=True)
class SearchResponse:
  profile_name: str
  chunks: list[RetrievedChunk]
  query_embedding_seconds: float
  database_seconds: float
  total_seconds: float
