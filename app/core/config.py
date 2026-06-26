import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_connection = os.getenv(
  "CONNECTION_STRING", "postgresql+psycopg://postgres:password@localhost:5432"
).rstrip("/")
database_url = (
  _connection if _connection.endswith("/postgres") else f"{_connection}/postgres"
)

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = os.getenv("LLM", "llama3.2")
current_batch = int(os.getenv("CURRENT_BATCH", "1"))
embedding_model = os.getenv("EMBEDDING_MODEL", "all-minilm")
api_key = os.getenv("API_KEY")
