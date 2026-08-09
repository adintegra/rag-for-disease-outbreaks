from app.db.models import Base


def test_new_schema_tables_are_registered() -> None:
  assert {
    "ingest.run",
    "ingest.raw_record",
    "rag.document",
    "rag.chunk_dataset",
    "rag.chunk",
    "rag.embedding",
  }.issubset(Base.metadata.tables)


def test_chunk_datasets_are_unique_by_document_content_and_configuration() -> None:
  table = Base.metadata.tables["rag.chunk_dataset"]
  unique_columns = {
    tuple(constraint.columns.keys())
    for constraint in table.constraints
    if constraint.__class__.__name__ == "UniqueConstraint"
  }

  assert ("document_id", "document_hash", "configuration_hash") in unique_columns
