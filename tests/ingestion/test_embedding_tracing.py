from app.embedding.tracing import EmbeddingTracer


def test_disabled_tracer_invokes_operation_once() -> None:
  calls = []
  tracer = EmbeddingTracer(enabled=False, project_name="test", trace_content=False)

  result = tracer.invoke_batch(
    lambda: calls.append("called") or [[0.1]],
    texts=["source text"],
    metadata={"chunk_ids": [1]},
  )

  assert result == [[0.1]]
  assert calls == ["called"]
