from collections.abc import Callable
from typing import TypeVar

import langsmith as ls

T = TypeVar("T")


class EmbeddingTracer:
  """Emit sanitized LangSmith batch traces without making tracing load-bearing."""

  def __init__(
    self,
    *,
    enabled: bool,
    project_name: str,
    trace_content: bool,
  ):
    self.enabled = enabled
    self.project_name = project_name
    self.trace_content = trace_content

  def invoke_batch(
    self,
    operation: Callable[[], T],
    *,
    texts: list[str],
    metadata: dict,
  ) -> T:
    if not self.enabled:
      return operation()

    inputs = {
      "count": len(texts),
      "text_lengths": [len(text) for text in texts],
      **metadata,
    }
    if self.trace_content:
      inputs["texts"] = texts

    result: T | None = None
    completed = False
    try:
      with (
        ls.tracing_context(enabled=True, project_name=self.project_name),
        ls.trace(
          "local-embedding-batch",
          run_type="embedding",
          inputs=inputs,
          tags=["embedding", "local", "openai-compatible"],
          metadata=metadata,
        ) as run,
      ):
        with ls.tracing_context(enabled=False):
          result = operation()
          completed = True
        run.end(outputs={"vectors": len(result) if isinstance(result, list) else 1})
      return result
    except Exception:
      if completed:
        # A LangSmith transport failure after successful embedding is non-fatal.
        return result
      raise
