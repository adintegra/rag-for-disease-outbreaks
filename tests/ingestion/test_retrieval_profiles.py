from pathlib import Path

import pytest

from app.retrieval.profiles import get_retrieval_profile, load_retrieval_profiles


def profile_file(path: Path) -> Path:
  path.write_text(
    """[profiles.test]\nchunk_profile = \"who-sections-1200\"\nembedding_model = \"model\"\nembedding_model_version = \"v1\"\ndistance = \"cosine\"\ndefault_limit = 5\ncandidate_multiplier = 3\nmax_chunks_per_document = 2\n"""
  )
  return path


def test_load_retrieval_profile_resolves_chunk_configuration(tmp_path: Path) -> None:
  profiles = load_retrieval_profiles(profile_file(tmp_path / "retrieval.toml"))

  assert profiles["test"].chunk_profile.name == "who-sections-1200"
  assert profiles["test"].embedding_model == "model"
  assert profiles["test"].default_limit == 5


def test_unknown_retrieval_profile_lists_available_names(tmp_path: Path) -> None:
  path = profile_file(tmp_path / "retrieval.toml")

  with pytest.raises(ValueError, match="test"):
    get_retrieval_profile("missing", path)
