"""`lineseek-features` CLI: build decision-time features from snapshots + labels, or validate."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import polars as pl

from lineseek.features.player_game import (
    FEATURE_SET_VERSION,
    build_season,
    features_root,
    summarize,
    validate_features,
    write_features,
)
from lineseek.grade.labels import labels_root
from lineseek.ingest.snapshot import data_root

log = logging.getLogger("lineseek.features")


def _seasons(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def validate_file(path: Path) -> pl.DataFrame:
    """Validate a feature Parquet file, and its checksum if a manifest sidecar exists."""
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.exists():
        expected = json.loads(sidecar.read_text())["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{path}: checksum does not match {sidecar.name}")
    return validate_features(pl.read_parquet(path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-features")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build features for labelled player-games of a season")
    b.add_argument("--season", required=True, help="season or range, e.g. 2025 or 2018-2025")
    v = sub.add_parser("validate", help="validate a feature Parquet file")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "validate":
        df = validate_file(args.path)
        print(f"OK {args.path}: {df.height:,} rows, feature set v{FEATURE_SET_VERSION}")
        return 0
    log.info("raw root: %s, labels root: %s, features root: %s", data_root().resolve(),
             labels_root().resolve(), features_root().resolve())
    for season in _seasons(args.season):
        build, inputs = build_season(season)
        m = write_features(build, season, inputs)
        print(f"{season}: {json.dumps(summarize(build.features))}; skipped history games="
              f"{list(build.skipped_history_games)}; history games without snaps="
              f"{list(build.history_games_without_snaps)} -> {features_root() / m.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
