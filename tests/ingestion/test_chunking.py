from types import SimpleNamespace

from app.ingestion.chunk_profiles import ChunkProfile
from app.ingestion.chunking import generate_chunks, split_recursive_with_offsets


def document(**overrides):
  values = {
    "title": "Ebola disease",
    "summary": "Summary text.",
    "epidemiology": "Epidemiology text.",
    "assessment": "Assessment text.",
    "overview": "Overview text.",
    "contents": "## Title\n\nEbola disease\n\n## Summary\n\nSummary text.",
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def test_who_sections_creates_one_chunk_per_short_section() -> None:
  profile = ChunkProfile("sections", "who_sections", "1", 500, 50)

  chunks = generate_chunks(document(), profile)

  assert [chunk.section for chunk in chunks] == [
    "summary",
    "epidemiology",
    "assessment",
    "overview",
  ]
  assert chunks[0].contents.startswith("# Ebola disease\n\n## Summary")


def test_recursive_strategy_records_source_offsets() -> None:
  profile = ChunkProfile("recursive", "recursive_characters", "1", 30, 5)
  source = "First paragraph has words.\n\nSecond paragraph has more words."

  chunks = generate_chunks(document(contents=source), profile)

  assert len(chunks) > 1
  for chunk in chunks:
    assert source[chunk.character_start : chunk.character_end] in chunk.contents


def test_recursive_split_respects_size_and_overlap() -> None:
  source = "one two three four five six seven eight nine ten"

  chunks = split_recursive_with_offsets(source, maximum=20, overlap=5)

  assert all(len(contents) <= 20 for contents, _, _ in chunks)
  assert chunks[0][1] == 0
  assert chunks[-1][2] == len(source)


def test_chunk_generation_is_deterministic() -> None:
  profile = ChunkProfile("recursive", "recursive_characters", "1", 30, 5)

  first = generate_chunks(document(), profile)
  second = generate_chunks(document(), profile)

  assert first == second
