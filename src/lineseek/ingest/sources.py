"""nflverse source adapters: load, conform to frozen schema, attach temporal provenance."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version

import nflreadpy as nfl
import polars as pl

from lineseek.contracts import (
    SOURCE_TEMPORAL_QUALITY,
    TemporalQuality,
    injury_report_available_at,
    kickoff_utc,
    validate_frame,
)
from lineseek.contracts.artifacts import UTC_TS
from lineseek.contracts.raw import conform, schema_variant

Prepare = Callable[[pl.DataFrame, int, datetime, pl.DataFrame | None], pl.DataFrame]


class IngestError(ValueError):
    pass


def _provenance(df: pl.DataFrame, source: str, observed_at: datetime,
                quality: TemporalQuality | pl.Expr, available_at: pl.Expr) -> pl.DataFrame:
    q = pl.lit(quality.value) if isinstance(quality, TemporalQuality) else quality
    return df.with_columns(
        pl.lit(source).alias("source"),
        pl.lit(f"nflreadpy=={version('nflreadpy')}").alias("source_version"),
        q.alias("temporal_quality"),
        available_at.cast(UTC_TS).alias("available_at"),
        pl.lit(observed_at, dtype=UTC_TS).alias("observed_at"),
    )


def _null_ts() -> pl.Expr:
    return pl.lit(None, dtype=UTC_TS)


def prepare_schedules(df, season, observed_at, schedules):
    kick = [kickoff_utc(d, t) for d, t in zip(df["gameday"], df["gametime"], strict=True)]
    df = df.with_columns(
        pl.Series("kickoff_utc", kick, dtype=UTC_TS),
        pl.when(pl.col("game_type") == "REG").then(pl.lit("REG")).otherwise(pl.lit("POST"))
        .alias("season_type"),
    )
    validate_frame(df, "games")
    # Row provenance describes kickoff (EXACT); scores/lines qualities live in the manifest.
    return _provenance(df, "schedules", observed_at, TemporalQuality.EXACT, pl.col("kickoff_utc"))


# Reviewed games absent from schedules but present in injury reports: (season, week, team).
CANCELLED_GAMES = pl.DataFrame(
    {"season": [2022, 2022], "week": [17, 17], "team": ["BUF", "CIN"],
     "gameday": ["2023-01-02", "2023-01-02"], "gametime": ["20:30", "20:30"]},  # BUF@CIN
    schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String, "gameday": pl.String,
            "gametime": pl.String},
)


def team_gamedays(schedules: pl.DataFrame) -> pl.DataFrame:
    cols = ["season", "week", "gameday"]
    sides = [schedules.select(*cols, pl.col(s).alias("team")) for s in ("home_team", "away_team")]
    scheduled = pl.concat(sides).select("season", "week", "team", "gameday")
    extra = CANCELLED_GAMES.drop("gametime").join(scheduled, on=["season", "week", "team"],
                                                  how="anti")
    return pl.concat([scheduled, extra])


def prepare_injuries(df, season, observed_at, schedules):
    if schedules is None:
        raise IngestError("injuries need the same season's schedules")
    joined = df.join(team_gamedays(schedules), on=["season", "week", "team"], how="left")
    missing = joined.filter(pl.col("gameday").is_null())
    if missing.height:
        keys = missing.select("season", "week", "team").unique().sort("week").rows()
        raise IngestError(f"injuries rows without a scheduled game: {keys}")
    rule = {d: injury_report_available_at(d) for d in joined["gameday"].unique()}
    joined = joined.with_columns(
        pl.col("gameday").replace_strict(rule, return_dtype=UTC_TS).alias("rule_available_at")
    ).drop("gameday")
    if "date_modified" in joined.columns:
        return _provenance(joined, "injuries", observed_at, TemporalQuality.EXACT,
                           pl.col("date_modified"))
    # Untimestamped: rule, tightened by our own observation time (both are >= truth).
    avail = pl.min_horizontal("rule_available_at", pl.lit(observed_at, dtype=UTC_TS))
    return _provenance(joined, "injuries", observed_at, TemporalQuality.INFERRED, avail)


def prepare_depth_charts(df, season, observed_at, schedules):
    if "dt" not in df.columns:
        return _provenance(df, "depth_charts", observed_at, TemporalQuality.UNKNOWN, _null_ts())
    df = df.with_columns(
        pl.lit(season, dtype=pl.Int32).alias("season"),
        pl.col("dt").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_unit="us", time_zone="UTC",
                                     strict=True).alias("snapshot_at"),
    )
    return _provenance(df, "depth_charts", observed_at, TemporalQuality.EXACT,
                       pl.col("snapshot_at"))


def _postgame(source: str) -> Prepare:
    def prepare(df, season, observed_at, schedules):
        return _provenance(df, source, observed_at, TemporalQuality.POSTGAME,
                           pl.lit(observed_at, dtype=UTC_TS))
    return prepare


def _unknown(source: str) -> Prepare:
    def prepare(df, season, observed_at, schedules):
        return _provenance(df, source, observed_at, TemporalQuality.UNKNOWN, _null_ts())
    return prepare


@dataclass(frozen=True)
class Source:
    name: str
    loader: Callable[[int], pl.DataFrame]
    prepare: Prepare
    # Weekly run keeps only rows from completed games.
    completed_games_only: bool = False

    def temporal_quality(self, season: int) -> dict[str, str]:
        variant = schema_variant(self.name, season)
        suffix = variant.split("/")[1] if "/" in variant else None
        groups = {g: q.value for (s, g), q in SOURCE_TEMPORAL_QUALITY.items() if s == self.name}
        if self.name in ("injuries", "depth_charts"):
            key = "2024_and_earlier" if season <= 2024 else "2025_and_later"
            groups = {suffix: groups[key]}
        return groups

    def build(self, raw: pl.DataFrame, season: int, observed_at: datetime,
              schedules: pl.DataFrame | None = None) -> pl.DataFrame:
        return self.prepare(conform(raw, self.name, season), season, observed_at, schedules)


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("schedules", nfl.load_schedules, prepare_schedules),
        Source("player_stats", nfl.load_player_stats, _postgame("player_stats"), True),
        Source("snap_counts", nfl.load_snap_counts, _postgame("snap_counts"), True),
        Source("pbp", nfl.load_pbp, _postgame("pbp"), True),
        Source("rosters_weekly", nfl.load_rosters_weekly, _unknown("rosters_weekly")),
        Source("injuries", nfl.load_injuries, prepare_injuries),
        Source("depth_charts", nfl.load_depth_charts, prepare_depth_charts),
    )
}
