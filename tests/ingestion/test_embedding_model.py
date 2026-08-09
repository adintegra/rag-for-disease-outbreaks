import math

import pytest

from app.embedding.model import EmbeddingIdentity, LocalEmbeddingModel


class FakeEmbeddings:
  def __init__(self, vectors):
    self.vectors = vectors
    self.calls = []

  def embed_documents(self, texts):
    self.calls.append(texts)
    return self.vectors

  def embed_query(self, text):
    return self.vectors[0]


def test_embedding_model_validates_a_complete_batch() -> None:
  adapter = FakeEmbeddings([[0.1, 0.2], [0.3, 0.4]])
  model = LocalEmbeddingModel(adapter, EmbeddingIdentity("model", "v1", 2))

  assert model.embed_documents(["one", "two"]) == adapter.vectors
  assert adapter.calls == [["one", "two"]]


def test_embedding_model_embeds_and_validates_query() -> None:
  adapter = FakeEmbeddings([[0.1, 0.2]])
  model = LocalEmbeddingModel(adapter, EmbeddingIdentity("model", "v1", 2))

  assert model.embed_query("outbreak") == [0.1, 0.2]


def test_embedding_model_rejects_empty_query() -> None:
  model = LocalEmbeddingModel(
    FakeEmbeddings([[0.1, 0.2]]), EmbeddingIdentity("model", "v1", 2)
  )

  with pytest.raises(ValueError, match="non-empty"):
    model.embed_query("  ")


def test_embedding_model_rejects_dimension_mismatch() -> None:
  model = LocalEmbeddingModel(
    FakeEmbeddings([[0.1]]), EmbeddingIdentity("model", "v1", 2)
  )

  with pytest.raises(ValueError, match="dimensions"):
    model.embed_documents(["one"])


def test_embedding_model_rejects_count_mismatch() -> None:
  model = LocalEmbeddingModel(
    FakeEmbeddings([[0.1, 0.2]]), EmbeddingIdentity("model", "v1", 2)
  )

  with pytest.raises(ValueError, match="1 vectors for 2 inputs"):
    model.embed_documents(["one", "two"])


def test_embedding_model_rejects_non_finite_values() -> None:
  model = LocalEmbeddingModel(
    FakeEmbeddings([[math.nan, 0.2]]), EmbeddingIdentity("model", "v1", 2)
  )

  with pytest.raises(ValueError, match="non-finite"):
    model.embed_documents(["one"])


def test_embedding_model_rejects_empty_input() -> None:
  model = LocalEmbeddingModel(
    FakeEmbeddings([]), EmbeddingIdentity("model", "v1", 2)
  )

  with pytest.raises(ValueError, match="non-empty"):
    model.embed_documents([])
