import langsmith as ls
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.core.config import Settings


class LocalChatModel:
  """LangChain chat adapter for a local OpenAI-compatible endpoint."""

  def __init__(self, model: BaseChatModel, *, suppress_content_tracing: bool = False):
    self.model = model
    self.suppress_content_tracing = suppress_content_tracing

  @classmethod
  def from_settings(cls, settings: Settings) -> "LocalChatModel":
    if settings.llm == settings.embedding_model:
      raise ValueError(
        "LLM_MODEL must identify a chat/instruct model, not the embedding model"
      )
    return cls(
      ChatOpenAI(
        model=settings.llm,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
      ),
      suppress_content_tracing=(
        settings.langsmith_tracing and not settings.langsmith_trace_content
      ),
    )

  def invoke(self, messages: list) -> str:
    if self.suppress_content_tracing:
      with ls.tracing_context(enabled=False):
        response = self.model.invoke(messages)
    else:
      response = self.model.invoke(messages)
    if not isinstance(response.content, str) or not response.content.strip():
      raise ValueError("Generation endpoint returned an empty answer")
    return response.content.strip()
