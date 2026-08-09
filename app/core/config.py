from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
  """Validated runtime configuration loaded from the process environment."""

  model_config = SettingsConfigDict(
    env_file=ROOT_DIR / ".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
  )

  database_url: str = Field(
    validation_alias=AliasChoices("DATABASE_URL", "CONNECTION_STRING")
  )
  database_url_direct: str | None = Field(default=None, alias="DATABASE_URL_DIRECT")
  ollama_base_url: str = "http://localhost:11434"
  llm: str = "llama3.2"
  embedding_model: str = "text-embedding-all-minilm-l6-v2-embedding"
  embedding_model_version: str | None = None
  embedding_dimensions: PositiveInt = 384
  embedding_provider: Literal["openai_compatible"] = "openai_compatible"
  embedding_base_url: str = Field(
    default="http://localhost:1234/v1",
    validation_alias=AliasChoices(
      "EMBEDDING_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"
    ),
  )
  embedding_api_key: str = Field(
    default="lm-studio",
    validation_alias=AliasChoices("EMBEDDING_API_KEY", "OPENAI_API_KEY"),
  )
  embedding_batch_size: PositiveInt = 16
  embedding_max_retries: int = Field(default=3, ge=0)
  embedding_request_timeout_seconds: PositiveInt = 120
  langsmith_tracing: bool = False
  langsmith_project: str = "don-rag-ingestion"
  langsmith_trace_content: bool = False
  api_key: str | None = None
  current_batch: int = 1

  who_don_api_url: str = (
    "https://www.who.int/api/emergencies/diseaseoutbreaknews"
  )
  who_request_timeout_seconds: PositiveInt = 30
  who_page_size: PositiveInt = 20
  ingestion_source: str = "who_don"
  ingestion_batch_size: PositiveInt = 100
  log_level: str = "INFO"

  @field_validator("database_url", "database_url_direct", mode="before")
  @classmethod
  def use_psycopg3(cls, value: str | None) -> str | None:
    if value is None:
      return None
    if value.startswith("postgresql://"):
      return value.replace("postgresql://", "postgresql+psycopg://", 1)
    if value.startswith("postgres://"):
      return value.replace("postgres://", "postgresql+psycopg://", 1)
    return value

  @property
  def migration_database_url(self) -> str:
    return self.database_url_direct or self.database_url

  @property
  def effective_embedding_model_version(self) -> str:
    return self.embedding_model_version or self.embedding_model


@lru_cache
def get_settings() -> Settings:
  return Settings()


_LEGACY_ATTRIBUTES = {
  "database_url": "database_url",
  "ollama_base_url": "ollama_base_url",
  "llm": "llm",
  "current_batch": "current_batch",
  "embedding_model": "embedding_model",
  "api_key": "api_key",
}


def __getattr__(name: str):
  """Keep existing callers working while they migrate to ``get_settings``."""
  setting_name = _LEGACY_ATTRIBUTES.get(name)
  if setting_name is None:
    raise AttributeError(name)
  return getattr(get_settings(), setting_name)
