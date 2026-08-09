from dataclasses import dataclass

from app.db.models import Document
from app.ingestion.chunk_profiles import ChunkProfile
from app.ingestion.hashing import text_hash


@dataclass(frozen=True)
class GeneratedChunk:
  index: int
  section: str | None
  contents: str
  content_hash: str
  character_start: int | None
  character_end: int | None
  metadata: dict


def generate_chunks(document: Document, profile: ChunkProfile) -> list[GeneratedChunk]:
  if profile.strategy == "who_sections":
    candidates = _section_chunks(document, profile)
  elif profile.strategy == "recursive_characters":
    candidates = _recursive_document_chunks(document, profile)
  else:
    raise ValueError(f"Unsupported chunk strategy: {profile.strategy}")

  return [
    GeneratedChunk(
      index=index,
      section=section,
      contents=contents,
      content_hash=text_hash(contents),
      character_start=start,
      character_end=end,
      metadata={"profile": profile.name},
    )
    for index, (section, contents, start, end) in enumerate(candidates)
    if contents.strip()
  ]


def _section_chunks(
  document: Document, profile: ChunkProfile
) -> list[tuple[str, str, None, None]]:
  chunks: list[tuple[str, str, None, None]] = []
  for section, value in (
    ("summary", document.summary),
    ("epidemiology", document.epidemiology),
    ("assessment", document.assessment),
    ("overview", document.overview),
  ):
    if not value:
      continue
    pieces = split_recursive(
      value, profile.max_characters, profile.overlap_characters
    )
    for piece in pieces:
      chunks.append(
        (
          section,
          _with_context(document.title, section, piece, profile.include_title),
          None,
          None,
        )
      )
  if not chunks:
    chunks.append(
      (
        "document",
        _with_context(
          document.title, "document", document.contents, profile.include_title
        ),
        None,
        None,
      )
    )
  return chunks


def _recursive_document_chunks(
  document: Document, profile: ChunkProfile
) -> list[tuple[None, str, int, int]]:
  pieces = split_recursive_with_offsets(
    document.contents, profile.max_characters, profile.overlap_characters
  )
  return [
    (
      None,
      _with_context(document.title, None, piece, profile.include_title),
      start,
      end,
    )
    for piece, start, end in pieces
  ]


def _with_context(
  title: str, section: str | None, contents: str, include_title: bool
) -> str:
  prefix: list[str] = []
  if include_title:
    prefix.append(f"# {title}")
  if section:
    prefix.append(f"## {section.replace('_', ' ').title()}")
  prefix.append(contents.strip())
  return "\n\n".join(prefix)


def split_recursive(text: str, maximum: int, overlap: int) -> list[str]:
  return [piece for piece, _, _ in split_recursive_with_offsets(text, maximum, overlap)]


def split_recursive_with_offsets(
  text: str, maximum: int, overlap: int
) -> list[tuple[str, int, int]]:
  if not text.strip():
    return []

  chunks: list[tuple[str, int, int]] = []
  start = 0
  length = len(text)
  while start < length:
    target = min(start + maximum, length)
    end = target
    if target < length:
      minimum_break = start + maximum // 2
      for separator in ("\n\n", "\n", ". ", " "):
        position = text.rfind(separator, minimum_break, target)
        if position >= minimum_break:
          end = position + len(separator)
          break

    raw_piece = text[start:end]
    leading = len(raw_piece) - len(raw_piece.lstrip())
    trailing = len(raw_piece) - len(raw_piece.rstrip())
    piece_start = start + leading
    piece_end = end - trailing
    if piece_start < piece_end:
      chunks.append((text[piece_start:piece_end], piece_start, piece_end))

    if end >= length:
      break
    next_start = max(end - overlap, start + 1)
    while next_start < length and text[next_start].isspace():
      next_start += 1
    start = next_start

  return chunks
