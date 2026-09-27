"""Reproduce the Phase 1 source audit (docs/audit/phase1_sources.md). Requires network."""

import sys
from importlib.metadata import version

import nflreadpy as nfl
import polars as pl

from lineseek.contracts import PROP_MARKETS

SEASONS = [2024, 2025]


def dupes(df: pl.DataFrame, keys: list[str]) -> int:
    return df.height - df.unique(keys).height


def main() -> None:
    failures: list[str] = []
    print("nflreadpy", version("nflreadpy"))

    sched = nfl.load_schedules(SEASONS)
    print(f"schedules: rows={sched.height} dup game_id={dupes(sched, ['game_id'])} "
          f"null gametime={sched['gametime'].null_count()}")

    ps = nfl.load_player_stats(SEASONS)
    needed = {m.stat_column for m in PROP_MARKETS.values()}
    missing = needed - set(ps.columns)
    if missing:
        failures.append(f"player_stats missing target columns {missing}")
    print(f"player_stats: rows={ps.height} dup (player_id,game_id)={dupes(ps, ['player_id', 'game_id'])} "
          f"null player_id={ps['player_id'].null_count()}")

    rw = nfl.load_rosters_weekly(SEASONS)
    print(f"rosters_weekly: rows={rw.height} dup (gsis_id,season,week)="
          f"{dupes(rw, ['gsis_id', 'season', 'week'])} null gsis_id={rw['gsis_id'].null_count()}")

    for season in SEASONS:
        inj = nfl.load_injuries(season)
        dm = inj["date_modified"].null_count() if "date_modified" in inj.columns else "absent"
        print(f"injuries {season}: rows={inj.height} date_modified nulls={dm}")
        dc = nfl.load_depth_charts(season)
        ts = "dt" in dc.columns
        print(f"depth_charts {season}: rows={dc.height} timestamped={ts} "
              f"snapshots={dc['dt'].n_unique() if ts else 'n/a'}")

    pbp = nfl.load_pbp(SEASONS[-1]).filter(pl.col("season_type") == "REG")
    reg = ps.filter(pl.col("season") == SEASONS[-1], pl.col("season_type") == "REG")
    cmp = (
        reg.group_by("player_id").agg(pl.col("receiving_yards").sum().alias("ps"))
        .join(pbp.group_by(pl.col("receiver_player_id").alias("player_id"))
              .agg(pl.col("receiving_yards").sum().alias("pbp")), on="player_id")
    )
    print(f"pbp vs player_stats receiving yds mismatches: "
          f"{cmp.filter(pl.col('ps') != pl.col('pbp')).height}/{cmp.height}")

    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
