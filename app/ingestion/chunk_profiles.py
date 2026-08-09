import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from app.ingestion.hashing import canonical_json_hash

DEFAULT_PROFILES_PATH = (
  Path(__file__).resolve().parents[1] / "config" / "chunk_profiles.toml"
)


@dataclass(frozen=True)
class ChunkProfile:
  name: str
  strategy: Literal["who_sections", "recursive_characters"]
  strategy_version: str
  max_characters: int
  overlap_characters: int
  include_title: bool = True

  def __post_init__(self) -> None:
    if self.max_characters <= 0:
      raise ValueError("max_characters must be positive")
    if self.overlap_characters < 0:
      raise ValueError("overlap_characters cannot be negative")
    if self.overlap_characters >= self.max_characters:
      raise ValueError("overlap_characters must be smaller than max_characters")

  @property
  def configuration(self) -> dict:
    values = asdict(self)
    values.pop("name")
    return values

  @property
  def configuration_hash(self) -> str:
    return canonical_json_hash(self.configuration)


def load_chunk_profiles(path: Path = DEFAULT_PROFILES_PATH) -> dict[str, ChunkProfile]:
  with path.open("rb") as file:
    raw_profiles = tomllib.load(file).get("profiles", {})
  return {
    name: ChunkProfile(name=name, **configuration)
    for name, configuration in raw_profiles.items()
  }


def get_chunk_profile(name: str, path: Path = DEFAULT_PROFILES_PATH) -> ChunkProfile:
  profiles = load_chunk_profiles(path)
  try:
    return profiles[name]
  except KeyError as error:
    available = ", ".join(sorted(profiles)) or "none"
    raise ValueError(
      f"Unknown chunk profile '{name}'. Available profiles: {available}"
    ) from error
