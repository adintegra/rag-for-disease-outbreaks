import argparse
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.ingestion.chunk_datasets import ChunkDatasetRepository
from app.ingestion.chunk_profiles import get_chunk_profile, load_chunk_profiles
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.source.who import WhoDonClient
from app.ingestion.staging import StagingRepository
from app.ingestion.transformation import TransformationRepository

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Ingest WHO Disease Outbreak News")
  subparsers = parser.add_subparsers(dest="command", required=True)
  extract = subparsers.add_parser("extract", help="Fetch and persist raw WHO records")
  extract.add_argument("--run-id", type=uuid.UUID, help="Resume a failed run")
  transform = subparsers.add_parser(
    "transform", help="Normalize staged records into canonical documents"
  )
  transform.add_argument("--run-id", type=uuid.UUID, required=True)
  chunk = subparsers.add_parser(
    "chunk", help="Generate a reproducible chunk dataset"
  )
  chunk.add_argument("--profile", required=True)
  subparsers.add_parser("profiles", help="List configured chunk profiles")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  settings = get_settings()
  logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

  if args.command == "extract":
    source = WhoDonClient(
      settings.who_don_api_url,
      timeout_seconds=settings.who_request_timeout_seconds,
      page_size=settings.who_page_size,
    )
    staging = StagingRepository(get_session_factory(), settings.ingestion_source)
    result = IngestionPipeline(source, staging).extract(args.run_id)
    logger.info(
      "Ingestion extract complete run_id=%s fetched=%d inserted=%d updated=%d unchanged=%d rejected=%d",
      result.run_id,
      result.fetched,
      result.inserted,
      result.updated,
      result.unchanged,
      result.rejected,
    )
  elif args.command == "transform":
    repository = TransformationRepository(
      get_session_factory(),
      settings.ingestion_source,
      settings.ingestion_batch_size,
    )
    result = repository.transform(args.run_id)
    logger.info(
      "Ingestion transform complete run_id=%s processed=%d inserted=%d updated=%d unchanged=%d rejected=%d",
      result.run_id,
      result.processed,
      result.inserted,
      result.updated,
      result.unchanged,
      result.rejected,
    )
  elif args.command == "chunk":
    profile = get_chunk_profile(args.profile)
    result = ChunkDatasetRepository(get_session_factory()).generate(profile)
    logger.info(
      "Chunk dataset complete profile=%s documents=%d created=%d skipped=%d failed=%d chunks=%d",
      result.profile_name,
      result.documents,
      result.created,
      result.skipped,
      result.failed,
      result.chunks,
    )
  elif args.command == "profiles":
    for profile in load_chunk_profiles().values():
      print(
        f"{profile.name}: {profile.strategy}@{profile.strategy_version} "
        f"max={profile.max_characters} overlap={profile.overlap_characters} "
        f"hash={profile.configuration_hash[:12]}"
      )
