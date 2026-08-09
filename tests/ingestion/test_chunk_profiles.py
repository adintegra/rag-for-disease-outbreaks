from pathlib import Path

import pytest

from app.ingestion.chunk_profiles import (
  ChunkProfile,
  get_chunk_profile,
  load_chunk_profiles,
)


def test_profiles_load_from_toml(tmp_path: Path) -> None:
  path = tmp_path / "profiles.toml"
  path.write_text(
    """[profiles.test]\nstrategy = \"who_sections\"\nstrategy_version = \"1\"\nmax_characters = 500\noverlap_characters = 50\ninclude_title = true\n"""
  )

  profiles = load_chunk_profiles(path)

  assert profiles["test"].max_characters == 500
  assert len(profiles["test"].configuration_hash) == 64


def test_profile_name_does_not_change_configuration_identity() -> None:
  first = ChunkProfile("first", "who_sections", "1", 500, 50)
  second = ChunkProfile("second", "who_sections", "1", 500, 50)

  assert first.configuration_hash == second.configuration_hash


def test_profile_rejects_overlap_equal_to_chunk_size() -> None:
  with pytest.raises(ValueError, match="smaller"):
    ChunkProfile("invalid", "who_sections", "1", 500, 500)


def test_unknown_profile_lists_available_profiles(tmp_path: Path) -> None:
  path = tmp_path / "profiles.toml"
  path.write_text(
    """[profiles.available]\nstrategy = \"recursive_characters\"\nstrategy_version = \"1\"\nmax_characters = 500\noverlap_characters = 50\n"""
  )

  with pytest.raises(ValueError, match="available"):
    get_chunk_profile("missing", path)
