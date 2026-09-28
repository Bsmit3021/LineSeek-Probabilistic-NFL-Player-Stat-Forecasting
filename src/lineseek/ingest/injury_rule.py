"""Check the 4pm-ET-day-before injury rule against real 2018-2024 `date_modified` timestamps."""

from pathlib import Path

import polars as pl

from lineseek.contracts import kickoff_utc
from lineseek.contracts.artifacts import UTC_TS
from lineseek.ingest.snapshot import read_latest
from lineseek.ingest.sources import CANCELLED_GAMES

TIMESTAMPED_SEASONS = range(2018, 2025)


def _team_kickoffs(sched: pl.DataFrame) -> pl.DataFrame:
    cancelled = CANCELLED_GAMES.select(
        "season", "week", "team",
        pl.Series([kickoff_utc(d, t) for d, t in CANCELLED_GAMES.select("gameday", "gametime")
                   .iter_rows()], dtype=UTC_TS).alias("kickoff_utc"),
    )
    return pl.concat([
        *[sched.select("season", "week", pl.col(side).alias("team"), "kickoff_utc")
          for side in ("home_team", "away_team")],
        cancelled,
    ]).unique(["season", "week", "team"])


def injury_rule_report(root: Path | None = None) -> pl.DataFrame:
    """Per-season share of true timestamps later than the rule (and later than kickoff)."""
    df = pl.concat([
        read_latest("injuries", s, root).join(
            _team_kickoffs(read_latest("schedules", s, root)), on=["season", "week", "team"],
            how="left",
        )
        for s in TIMESTAMPED_SEASONS
    ], how="diagonal_relaxed")
    if df["kickoff_utc"].null_count():
        raise ValueError("injury rows without kickoff")
    late = pl.col("date_modified") > pl.col("rule_available_at")
    aggs = [
        pl.len().alias("rows"),
        late.sum().alias("after_rule"),
        late.mean().alias("frac_after_rule"),
        (pl.col("date_modified") > pl.col("kickoff_utc")).mean().alias("frac_after_kickoff"),
    ]
    by_season = df.group_by(pl.col("season").cast(pl.String)).agg(aggs).sort("season")
    return pl.concat([by_season, df.select(pl.lit("all").alias("season"), *aggs)])
