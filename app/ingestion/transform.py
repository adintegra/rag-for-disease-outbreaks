import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from markdownify import MarkdownConverter

from app.ingestion.hashing import text_hash

WHO_ITEM_URL = "https://www.who.int/emergencies/disease-outbreak-news/item"


class InvalidSourceRecord(ValueError):
  pass


@dataclass(frozen=True)
class CanonicalDocument:
  source: str
  source_id: str
  source_updated_at: datetime | None
  title: str
  subtitle: str | None
  summary: str | None
  epidemiology: str | None
  assessment: str | None
  overview: str | None
  contents: str
  url: str
  published_at: datetime
  event_date: datetime | None
  content_hash: str
  metadata: dict[str, Any]


def normalize_who_don(payload: dict[str, Any], source: str) -> CanonicalDocument:
  source_id = _required_text(payload, "UrlName")
  title = _clean_markdown(_required_text(payload, "Title"))
  if not title:
    raise InvalidSourceRecord("Title is empty after normalization")

  published_at = _required_datetime(payload, "PublicationDate")
  source_updated_at = _optional_datetime(payload.get("LastModified"))
  emergency = payload.get("EmergencyEvent")
  event_date = (
    _optional_datetime(emergency.get("EmergencyEventStartDate"))
    if isinstance(emergency, dict)
    else None
  )

  subtitle = _optional_markdown(payload.get("TitleSuffix"))
  summary = _optional_markdown(payload.get("Summary"))
  epidemiology = _optional_markdown(payload.get("Epidemiology"))
  assessment = _optional_markdown(payload.get("Assessment"))
  overview = _optional_markdown(payload.get("Overview"))

  sections = [("Title", title)]
  for label, value in (
    ("Subtitle", subtitle),
    ("Summary", summary),
    ("Epidemiology", epidemiology),
    ("Assessment", assessment),
    ("Overview", overview),
  ):
    if value:
      sections.append((label, value))
  contents = "\n\n".join(f"## {label}\n\n{value}" for label, value in sections)

  return CanonicalDocument(
    source=source,
    source_id=source_id,
    source_updated_at=source_updated_at,
    title=title,
    subtitle=subtitle,
    summary=summary,
    epidemiology=epidemiology,
    assessment=assessment,
    overview=overview,
    contents=contents,
    url=f"{WHO_ITEM_URL}/{source_id}",
    published_at=published_at,
    event_date=event_date,
    content_hash=text_hash(contents),
    metadata={
      "don_id": payload.get("DonId"),
      "item_default_url": payload.get("ItemDefaultUrl"),
      "formatted_date": payload.get("FormattedDate"),
    },
  )


def _required_text(payload: dict[str, Any], field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not value.strip():
    raise InvalidSourceRecord(f"Missing required field: {field}")
  return value.strip()


def _required_datetime(payload: dict[str, Any], field: str) -> datetime:
  value = _optional_datetime(payload.get(field))
  if value is None:
    raise InvalidSourceRecord(f"Missing or invalid datetime: {field}")
  return value


def _optional_datetime(value: Any) -> datetime | None:
  if not isinstance(value, str) or not value.strip():
    return None
  try:
    return datetime.fromisoformat(value)
  except ValueError:
    return None


def _optional_markdown(value: Any) -> str | None:
  if not isinstance(value, str) or not value.strip():
    return None
  cleaned = _clean_markdown(value)
  return cleaned or None


def _clean_markdown(value: str) -> str:
  warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
  soup = BeautifulSoup(value, "html.parser")
  markdown = MarkdownConverter(
    newline_style="BACKSLASH", table_infer_header=True
  ).convert_soup(soup)
  lines = [line.rstrip() for line in markdown.splitlines()]
  return "\n".join(lines).strip()
