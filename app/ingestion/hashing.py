import hashlib
import json
from collections.abc import Mapping
from typing import Any


def text_hash(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Mapping[str, Any]) -> str:
  serialized = json.dumps(
    value,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
  )
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
