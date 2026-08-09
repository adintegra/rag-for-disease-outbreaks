from app.retrieval.tracing import RetrievalTracer


def test_disabled_retrieval_tracer_is_a_noop() -> None:
  tracer = RetrievalTracer(enabled=False, project_name="test", trace_query=False)

  assert tracer.record("query", None, None, None) is None
