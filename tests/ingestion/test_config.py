from app.core.config import Settings


def test_neon_postgresql_url_uses_psycopg3_driver() -> None:
  settings = Settings(
    DATABASE_URL="postgresql://user:secret@example.neon.tech/database?sslmode=require"
  )

  assert settings.database_url.startswith("postgresql+psycopg://")
  assert settings.database_url.endswith("?sslmode=require")


def test_scheduled_chunk_profiles_parse_comma_separated_names() -> None:
  settings = Settings(
    DATABASE_URL="postgresql://user:secret@example.test/database",
    INGESTION_CHUNK_PROFILES="who-sections-1200, recursive-1000-150",
  )

  assert settings.scheduled_chunk_profiles == [
    "who-sections-1200",
    "recursive-1000-150",
  ]


def test_explicit_sqlalchemy_driver_is_preserved() -> None:
  settings = Settings(
    DATABASE_URL="postgresql+psycopg://user:secret@example.test/database"
  )

  assert settings.database_url.startswith("postgresql+psycopg://")
