from collections.abc import Iterator
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WhoDonClient:
  """Retrieve complete WHO Disease Outbreak News pages with bounded retries."""

  def __init__(
    self,
    base_url: str,
    *,
    timeout_seconds: int = 30,
    page_size: int = 20,
    session: requests.Session | None = None,
  ):
    self.base_url = base_url
    self.timeout_seconds = timeout_seconds
    self.page_size = page_size
    self.session = session or self._retrying_session()

  @staticmethod
  def _retrying_session() -> requests.Session:
    retry = Retry(
      total=4,
      connect=4,
      read=4,
      status=4,
      backoff_factor=1,
      status_forcelist=(429, 500, 502, 503, 504),
      allowed_methods=frozenset({"GET"}),
      respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
      {"User-Agent": "mas-master-thesis-who-don-ingestion/0.1"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

  def iter_records(self) -> Iterator[dict[str, Any]]:
    skip = 0
    expected_total: int | None = None

    while expected_total is None or skip < expected_total:
      response = self.session.get(
        self.base_url,
        params={
          "sf_provider": "dynamicProvider372",
          "sf_culture": "en",
          "$orderby": "PublicationDateAndTime desc",
          "$expand": "EmergencyEvent",
          "$format": "json",
          "$count": "true",
          "$top": self.page_size,
          "$skip": skip,
        },
        timeout=self.timeout_seconds,
      )
      response.raise_for_status()
      body = response.json()
      records = body.get("value", [])
      if not isinstance(records, list):
        raise TypeError("WHO response field 'value' must be a list")

      count = body.get("@odata.count") or body.get("count")
      if count is not None:
        expected_total = int(count)

      if not records:
        break

      for record in records:
        if not isinstance(record, dict):
          raise TypeError("WHO response records must be objects")
        yield record

      skip += len(records)
      if len(records) < self.page_size:
        break
