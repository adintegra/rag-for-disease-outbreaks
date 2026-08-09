import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.ingestion.chunk_profiles import ChunkProfile, get_chunk_profile

DEFAULT_RETRIEVAL_PROFILES_PATH = (
  Path(__file__).resolve().parents[1] / "config" / "retrieval_profiles.toml"
)


@dataclass(frozen=True)
class RetrievalProfile:
  name: str
  chunk_profile_name: str
  chunk_profile: ChunkProfile
  embedding_model: str
  embedding_model_version: str
  distance: Literal["cosine"]
  default_limit: int
  candidate_multiplier: int
  max_chunks_per_document: int

  def __post_init__(self) -> None:
    if self.default_limit <= 0:
      raise ValueError("default_limit must be positive")
    if self.candidate_multiplier <= 0:
      raise ValueError("candidate_multiplier must be positive")
    if self.max_chunks_per_document <= 0:
      raise ValueError("max_chunks_per_document must be positive")


def load_retrieval_profiles(
  path: Path = DEFAULT_RETRIEVAL_PROFILES_PATH,
) -> dict[str, RetrievalProfile]:
  with path.open("rb") as file:
    raw_profiles = tomllib.load(file).get("profiles", {})

  profiles = {}
  for name, configuration in raw_profiles.items():
    chunk_profile_name = configuration.pop("chunk_profile")
    profiles[name] = RetrievalProfile(
      name=name,
      chunk_profile_name=chunk_profile_name,
      chunk_profile=get_chunk_profile(chunk_profile_name),
      **configuration,
    )
  return profiles


def get_retrieval_profile(
  name: str, path: Path = DEFAULT_RETRIEVAL_PROFILES_PATH
) -> RetrievalProfile:
  profiles = load_retrieval_profiles(path)
  try:
    return profiles[name]
  except KeyError as error:
    available = ", ".join(sorted(profiles)) or "none"
    raise ValueError(
      f"Unknown retrieval profile '{name}'. Available profiles: {available}"
    ) from error
