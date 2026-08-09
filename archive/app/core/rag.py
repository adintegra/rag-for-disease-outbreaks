from dataclasses import dataclass

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core import config
from db.vector_store import DocEmbeddingView


@dataclass
class RetrievedDocument:
  document_id: int
  contents: str
  url: str | None
  similarity: float


def similarity_search(
  query: str,
  *,
  model: str | None = None,
  batch: int | None = None,
  limit: int = 5,
) -> list[RetrievedDocument]:
  embedding_model = model or config.embedding_model
  query_vector = OllamaEmbeddings(
    model=embedding_model, base_url=config.ollama_base_url
  ).embed_query(query)

  stmt = (
    select(
      DocEmbeddingView.document_id,
      DocEmbeddingView.contents,
      DocEmbeddingView.url,
      DocEmbeddingView.embedding.cosine_distance(query_vector).label("similarity"),
    )
    .filter(DocEmbeddingView.model == embedding_model)
    .filter(DocEmbeddingView.batch == (batch if batch is not None else config.current_batch))
    .order_by(DocEmbeddingView.embedding.cosine_distance(query_vector))
    .limit(limit)
  )

  engine = create_engine(config.database_url)
  Session = sessionmaker(bind=engine)
  with Session() as session:
    rows = session.execute(stmt).all()

  if not rows:
    raise ValueError("No documents found.")

  return [
    RetrievedDocument(
      document_id=row.document_id,
      contents=row.contents or "",
      url=row.url,
      similarity=float(row.similarity),
    )
    for row in rows
  ]


def build_prompt(query: str, docs: list[RetrievedDocument]) -> str:
  context = "\n\n---\n\n".join(
    f"URL: {doc.url or 'unknown'}\n{doc.contents}" for doc in docs
  )
  return f"Answer using only this context.\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"


def generate_answer(query: str, docs: list[RetrievedDocument], *, llm_model: str | None = None) -> str:
  prompt = build_prompt(query, docs)
  llm = OllamaLLM(model=llm_model or config.llm, base_url=config.ollama_base_url)
  return llm.invoke(prompt)
