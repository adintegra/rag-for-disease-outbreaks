from datetime import UTC, datetime

import pytest

from app.ingestion.transform import InvalidSourceRecord, normalize_who_don


def source_payload(**overrides):
  payload = {
    "UrlName": "ebola-disease-caused-by-sudan-virus-uganda",
    "Title": "<p>Ebola disease</p>",
    "TitleSuffix": "Uganda",
    "Summary": "<p>Two cases were reported.</p>",
    "Epidemiology": "<p>Transmission occurred.</p>",
    "Assessment": "<p>Risk is moderate.</p>",
    "Overview": "<p>An overview.</p>",
    "PublicationDate": "2026-01-02T10:00:00Z",
    "LastModified": "2026-01-03T10:00:00Z",
    "DonId": "DON123",
    "EmergencyEvent": {"EmergencyEventStartDate": "2025-12-20T00:00:00Z"},
  }
  payload.update(overrides)
  return payload


def test_normalize_who_don_builds_stable_canonical_document() -> None:
  document = normalize_who_don(source_payload(), "who_don")

  assert document.source_id == "ebola-disease-caused-by-sudan-virus-uganda"
  assert document.title == "Ebola disease"
  assert document.published_at == datetime(2026, 1, 2, 10, tzinfo=UTC)
  assert document.event_date == datetime(2025, 12, 20, tzinfo=UTC)
  assert document.url.endswith(document.source_id)
  assert "## Summary\n\nTwo cases were reported." in document.contents
  assert document.metadata["don_id"] == "DON123"
  assert len(document.content_hash) == 64


def test_normalize_who_don_preserves_markdown_tables() -> None:
  payload = source_payload(
    Summary="<table><tr><th>Country</th><th>Cases</th></tr><tr><td>Uganda</td><td>2</td></tr></table>"
  )

  document = normalize_who_don(payload, "who_don")

  assert "Country" in document.summary
  assert "Uganda" in document.summary
  assert "|" in document.summary


def test_normalization_hash_is_deterministic() -> None:
  first = normalize_who_don(source_payload(), "who_don")
  second = normalize_who_don(source_payload(), "who_don")

  assert first.content_hash == second.content_hash
  assert first.contents == second.contents


@pytest.mark.parametrize("field", ["UrlName", "Title", "PublicationDate"])
def test_normalize_who_don_rejects_missing_required_fields(field: str) -> None:
  payload = source_payload()
  payload[field] = None

  with pytest.raises(InvalidSourceRecord):
    normalize_who_don(payload, "who_don")
