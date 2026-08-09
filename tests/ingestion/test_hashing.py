from app.ingestion.hashing import canonical_json_hash


def test_canonical_json_hash_ignores_mapping_key_order() -> None:
  assert canonical_json_hash({"title": "Ebola", "count": 2}) == canonical_json_hash(
    {"count": 2, "title": "Ebola"}
  )


def test_canonical_json_hash_changes_with_payload() -> None:
  assert canonical_json_hash({"count": 2}) != canonical_json_hash({"count": 3})
