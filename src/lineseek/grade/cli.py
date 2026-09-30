"""`lineseek-labels` CLI: build canonical prop labels from raw snapshots, or validate a file."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import polars as pl

from lineseek.contracts import validate_frame
from lineseek.grade.labels import build_season, labels_root, write_labels
from lineseek.ingest.snapshot import data_root

log = logging.getLogger("lineseek.labels")


def _seasons(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def _summary(df: pl.DataFrame) -> str:
    status = dict(df.group_by("label_status").len().sort("label_status").iter_rows())
    return f"{df.height:,} labels, {df.select('game_id').n_unique()} games, status={status}"


def validate_file(path: Path) -> pl.DataFrame:
    """Validate a prop_labels Parquet file, and its checksum if a manifest sidecar exists."""
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.exists():
        expected = json.loads(sidecar.read_text())["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{path}: checksum does not match {sidecar.name}")
    return validate_frame(pl.read_parquet(path), "prop_labels")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-labels")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build labels from the latest raw snapshots per season")
    b.add_argument("--season", required=True, help="season or range, e.g. 2025 or 2018-2025")
    b.add_argument("--skip-incomplete", action="store_true",
                   help="exclude completed games missing stats/snaps instead of failing")
    v = sub.add_parser("validate", help="validate a prop_labels Parquet file")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "validate":
        print(f"OK {args.path}: {_summary(validate_file(args.path))}")
        return 0
    log.info("raw root: %s, labels root: %s", data_root().resolve(), labels_root().resolve())
    for season in _seasons(args.season):
        build, inputs = build_season(season, skip_incomplete=args.skip_incomplete)
        m = write_labels(build, season, inputs)
        print(f"{season}: {_summary(build.labels)}; quarantined={build.quarantined.height} "
              f"null-id stat rows; snap-team mismatches={build.snap_team_mismatches}; "
              f"skipped incomplete games={build.incomplete_games} -> "
              f"{labels_root() / m.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
