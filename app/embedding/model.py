import math
from dataclasses import dataclass
from time import perf_counter

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings


@dataclass(frozen=True)
class EmbeddingIdentity:
  model: str
  model_version: str
  dimensions: int


@dataclass(frozen=True)
class EmbeddingPreflight:
  identity: EmbeddingIdentity
  latency_seconds: float


class LocalEmbeddingModel:
  """Validated LangChain adapter for a local OpenAI-compatible endpoint."""

  def __init__(self, embeddings: Embeddings, identity: EmbeddingIdentity):
    self.embeddings = embeddings
    self.identity = identity

  @classmethod
  def from_settings(cls, settings: Settings) -> "LocalEmbeddingModel":
    embeddings = OpenAIEmbeddings(
      model=settings.embedding_model,
      base_url=settings.embedding_base_url,
      api_key=settings.embedding_api_key,
      check_embedding_ctx_length=False,
      model_kwargs={"encoding_format": "float"},
      max_retries=settings.embedding_max_retries,
      request_timeout=settings.embedding_request_timeout_seconds,
    )
    return cls(
      embeddings,
      EmbeddingIdentity(
        model=settings.embedding_model,
        model_version=settings.effective_embedding_model_version,
        dimensions=settings.embedding_dimensions,
      ),
    )

  def preflight(self) -> EmbeddingPreflight:
    started = perf_counter()
    self.embed_documents(["Disease outbreak embedding preflight."])
    return EmbeddingPreflight(self.identity, perf_counter() - started)

  def embed_query(self, query: str) -> list[float]:
    if not query.strip():
      raise ValueError("Embedding query must contain non-empty text")
    vector = self.embeddings.embed_query(query)
    self._validate([vector], 1)
    return vector

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    if not texts or any(not text.strip() for text in texts):
      raise ValueError("Embedding inputs must contain non-empty text")
    vectors = self.embeddings.embed_documents(texts)
    self._validate(vectors, len(texts))
    return vectors

  def _validate(self, vectors: list[list[float]], expected_count: int) -> None:
    if len(vectors) != expected_count:
      raise ValueError(
        f"Embedding endpoint returned {len(vectors)} vectors for {expected_count} inputs"
      )
    for index, vector in enumerate(vectors):
      if len(vector) != self.identity.dimensions:
        raise ValueError(
          f"Embedding {index} has {len(vector)} dimensions; "
          f"expected {self.identity.dimensions}"
        )
      if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"Embedding {index} contains non-finite values")
