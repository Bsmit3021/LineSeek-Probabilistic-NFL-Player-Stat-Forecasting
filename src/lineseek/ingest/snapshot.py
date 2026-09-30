"""Append-only Parquet snapshots: one file + one manifest per pull, never overwritten."""

import hashlib
import io
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import polars as pl

from lineseek.contracts import CONTRACT_VERSION

DATA_ROOT_ENV = "LINESEEK_DATA_ROOT"
DEFAULT_DATA_ROOT = "data/raw"
REPO_DIR = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT))


def git_revision() -> str | None:
    """HEAD sha, suffixed '-dirty' if tracked files are modified; None outside a git checkout."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
        return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class Manifest:
    source: str
    schema_variant: str
    seasons: list[int]
    weeks: list[int] | None
    pulled_at: str
    row_count: int
    schema: dict[str, str]
    sha256: str
    bytes: int
    path: str
    nflreadpy_version: str
    git_revision: str | None
    contract_version: str
    temporal_quality: dict[str, str] = field(default_factory=dict)
    # Upstream snapshots a derived artifact was built from: {source, path, sha256}.
    inputs: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)


def write_snapshot(
    df: pl.DataFrame,
    source: str,
    schema_variant: str,
    seasons: list[int],
    weeks: list[int] | None,
    pulled_at: datetime,
    temporal_quality: dict[str, str] | None = None,
    root: Path | None = None,
    inputs: list[dict[str, str]] | None = None,
) -> Manifest:
    if pulled_at.utcoffset() != timedelta(0):
        raise ValueError("pulled_at must be tz-aware UTC")
    root = root or data_root()
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd", statistics=True)
    blob = buf.getvalue()
    sha = hashlib.sha256(blob).hexdigest()
    part = f"season={seasons[0]}" if len(seasons) == 1 else "season=multi"
    stem = f"{pulled_at.astimezone(UTC):%Y%m%dT%H%M%S%fZ}_{sha[:12]}"
    rel = Path(source) / part / f"{stem}.parquet"
    manifest = Manifest(
        source=source,
        schema_variant=schema_variant,
        seasons=seasons,
        weeks=weeks,
        pulled_at=pulled_at.astimezone(UTC).isoformat(),
        row_count=df.height,
        schema={c: str(t) for c, t in df.schema.items()},
        sha256=sha,
        bytes=len(blob),
        path=rel.as_posix(),
        nflreadpy_version=version("nflreadpy"),
        git_revision=git_revision(),
        contract_version=CONTRACT_VERSION,
        temporal_quality=temporal_quality or {},
        inputs=inputs or [],
    )
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "xb") as f:  # "x" = fail if exists; never overwrite
        f.write(blob)
    with open(target.with_suffix(".manifest.json"), "x") as f:
        f.write(manifest.to_json() + "\n")
    return manifest


def list_manifests(source: str, season: int | None = None, root: Path | None = None) -> list[dict]:
    base = (root or data_root()) / source
    pattern = f"season={season}/*.manifest.json" if season is not None else "*/*.manifest.json"
    return [json.loads(p.read_text()) for p in sorted(base.glob(pattern))]


def latest_manifest(source: str, season: int, root: Path | None = None) -> dict:
    manifests = list_manifests(source, season, root)
    if not manifests:
        raise FileNotFoundError(f"no snapshot for {source} season={season}")
    return manifests[-1]


def read_latest(source: str, season: int, root: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet((root or data_root()) / latest_manifest(source, season, root)["path"])
