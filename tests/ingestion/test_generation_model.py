import pytest
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.generation.model import LocalChatModel


class FakeChatModel:
  def __init__(self, content):
    self.content = content

  def invoke(self, messages):
    return AIMessage(content=self.content)


def test_chat_model_returns_trimmed_answer() -> None:
  model = LocalChatModel(FakeChatModel("  Answer [1].  "))

  assert model.invoke([]) == "Answer [1]."


def test_chat_model_rejects_embedding_model_as_generation_model() -> None:
  settings = Settings(
    DATABASE_URL="postgresql://user:secret@example.test/database",
    LLM_MODEL="same-model",
    EMBEDDING_MODEL="same-model",
  )

  with pytest.raises(ValueError, match="chat/instruct"):
    LocalChatModel.from_settings(settings)


def test_chat_model_rejects_empty_answer() -> None:
  model = LocalChatModel(FakeChatModel(" "))

  with pytest.raises(ValueError, match="empty"):
    model.invoke([])
