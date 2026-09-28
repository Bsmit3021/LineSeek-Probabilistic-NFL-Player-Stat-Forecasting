"""`lineseek-ingest` CLI: backfill history once, snapshot the current week, check injury rule."""

import argparse
import logging
import sys

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

from lineseek.ingest.pipeline import HISTORY_SEASONS, backfill, weekly
from lineseek.ingest.snapshot import data_root
from lineseek.ingest.sources import SOURCES

log = logging.getLogger("lineseek.ingest")


def _seasons(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-ingest")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="snapshot history once (skips existing source/season)")
    b.add_argument("--seasons", default=f"{HISTORY_SEASONS[0]}-{HISTORY_SEASONS[-1]}")
    b.add_argument("--sources", default=",".join(SOURCES))
    b.add_argument("--force", action="store_true", help="snapshot again even if present")
    w = sub.add_parser("weekly", help="snapshot the current season/week")
    w.add_argument("--season", type=int)
    w.add_argument("--week", type=int)
    sub.add_parser("injury-rule-report", help="validate injury rule on 2018-2024 snapshots")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    update_config(cache_mode="off")  # snapshots must reflect the source at pulled_at
    log.info("data root: %s", data_root().resolve())

    if args.cmd == "backfill":
        names = args.sources.split(",")
        unknown = set(names) - set(SOURCES)
        if unknown:
            p.error(f"unknown sources: {sorted(unknown)}")
        manifests = backfill(_seasons(args.seasons), names, args.force)
    elif args.cmd == "weekly":
        season = args.season or nfl.get_current_season()
        week = args.week or nfl.get_current_week()
        manifests = weekly(season, week)
    else:
        from lineseek.ingest.injury_rule import injury_rule_report

        with pl.Config(tbl_rows=-1, float_precision=4):
            print(injury_rule_report())
        return 0
    rows = sum(m.row_count for m in manifests)
    size = sum(m.bytes for m in manifests)
    print(f"{len(manifests)} snapshot(s), {rows:,} rows, {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
