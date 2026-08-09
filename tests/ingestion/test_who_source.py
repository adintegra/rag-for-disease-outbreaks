from typing import Any

import pytest

from app.ingestion.source.who import WhoDonClient


class FakeResponse:
  def __init__(self, body: dict[str, Any], status_code: int = 200):
    self.body = body
    self.status_code = status_code

  def raise_for_status(self) -> None:
    if self.status_code >= 400:
      raise RuntimeError(f"HTTP {self.status_code}")

  def json(self) -> dict[str, Any]:
    return self.body


class FakeSession:
  def __init__(self, responses: list[FakeResponse]):
    self.responses = iter(responses)
    self.calls: list[dict[str, Any]] = []

  def get(self, url: str, **kwargs: Any) -> FakeResponse:
    self.calls.append({"url": url, **kwargs})
    return next(self.responses)


def test_iter_records_paginates_using_returned_count() -> None:
  session = FakeSession(
    [
      FakeResponse({"@odata.count": 3, "value": [{"UrlName": "one"}, {"UrlName": "two"}]}),
      FakeResponse({"@odata.count": 3, "value": [{"UrlName": "three"}]}),
    ]
  )
  client = WhoDonClient("https://example.test", page_size=2, session=session)

  assert [record["UrlName"] for record in client.iter_records()] == [
    "one",
    "two",
    "three",
  ]
  assert [call["params"]["$skip"] for call in session.calls] == [0, 2]


def test_iter_records_stops_on_an_empty_page_without_count() -> None:
  session = FakeSession(
    [FakeResponse({"value": [{"UrlName": "one"}]}), FakeResponse({"value": []})]
  )
  client = WhoDonClient("https://example.test", page_size=1, session=session)

  assert list(client.iter_records()) == [{"UrlName": "one"}]
  assert len(session.calls) == 2


def test_iter_records_rejects_an_invalid_value_shape() -> None:
  client = WhoDonClient(
    "https://example.test",
    session=FakeSession([FakeResponse({"value": {"UrlName": "one"}})]),
  )

  with pytest.raises(TypeError, match="must be a list"):
    list(client.iter_records())
