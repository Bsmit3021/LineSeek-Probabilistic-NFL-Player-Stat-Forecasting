"""Participation (`played`) for market-track grading: (offense_snaps + st_snaps) > 0."""

from datetime import datetime, timedelta
from importlib.resources import files

import polars as pl

from lineseek.contracts import TemporalQuality

# Reviewed PFR->GSIS fixes; a blank gsis_id is an explicit exclusion (not in nflverse).
OVERRIDES_PATH = files("lineseek.grade") / "pfr_gsis_overrides.csv"
OVERRIDE_SCHEMA = {"pfr_player_id": pl.String, "gsis_id": pl.String, "note": pl.String}


class UnmatchedPlayerError(ValueError):
    def __init__(self, pfr_ids: list[str]):
        self.pfr_ids = sorted(pfr_ids)
        super().__init__(f"{len(self.pfr_ids)} unmatched PFR id(s): {', '.join(self.pfr_ids)}")


def load_overrides() -> pl.DataFrame:
    return pl.read_csv(OVERRIDES_PATH.read_bytes(), schema=OVERRIDE_SCHEMA)


def build_pfr_gsis_map(
    snaps: pl.DataFrame,
    players: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    overrides: pl.DataFrame,
) -> pl.DataFrame:
    """Map every snap-count pfr id to gsis: players table, then roster name match, then overrides."""
    ids = snaps.select("pfr_player_id", "player", "team", "season").unique()
    primary = players.select(pl.col("pfr_id").alias("pfr_player_id"), "gsis_id").drop_nulls()
    primary = primary.unique()
    if primary["pfr_player_id"].is_duplicated().any():
        raise ValueError("players table maps a pfr_id to multiple gsis_ids")

    # Roster fallback: unique exact (name, team, season) match whose gsis has no other pfr id.
    roster_names = rosters_weekly.select(
        pl.col("full_name").alias("player"), "team", "season", "gsis_id"
    ).drop_nulls().unique()
    by_name = (
        ids.join(primary, on="pfr_player_id", how="anti")
        .join(roster_names, on=["player", "team", "season"])
        .group_by("pfr_player_id").agg(pl.col("gsis_id").unique())
        .filter(pl.col("gsis_id").list.len() == 1)
        .explode("gsis_id", empty_as_null=True)
        .filter(~pl.col("gsis_id").is_in(primary["gsis_id"].implode()))
    )
    manual = (
        overrides.select("pfr_player_id", "gsis_id")
        .join(primary, on="pfr_player_id", how="anti")
        .join(by_name, on="pfr_player_id", how="anti")
    )
    mapping = pl.concat([primary, by_name, manual])
    unmatched = ids.join(mapping, on="pfr_player_id", how="anti")["pfr_player_id"].unique()
    if len(unmatched):
        raise UnmatchedPlayerError(unmatched.to_list())
    return mapping.join(ids.select("pfr_player_id").unique(), on="pfr_player_id")


def derive_played(
    projected: pl.DataFrame,
    snaps: pl.DataFrame,
    id_map: pl.DataFrame,
    observed_at: datetime,
    source_version: str,
) -> pl.DataFrame:
    """Add `played` and POSTGAME provenance to projected (game_id, player_id) rows."""
    if observed_at.utcoffset() != timedelta(0):
        raise ValueError("observed_at must be tz-aware UTC")
    unmatched = snaps.join(id_map, on="pfr_player_id", how="anti")["pfr_player_id"].unique()
    if len(unmatched):
        raise UnmatchedPlayerError(unmatched.to_list())
    snap_total = pl.col("offense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)
    snap_played = (
        snaps.join(id_map.drop_nulls("gsis_id"), on="pfr_player_id")
        .group_by("game_id", pl.col("gsis_id").alias("player_id"))
        .agg((snap_total.sum() > 0).alias("played"))
    )
    ts = pl.lit(observed_at, dtype=pl.Datetime("us", "UTC"))
    return projected.join(snap_played, on=["game_id", "player_id"], how="left").with_columns(
        pl.col("played").fill_null(False),  # projected but no snap row -> did not play
        pl.lit("snap_counts").alias("source"),
        pl.lit(source_version).alias("source_version"),
        pl.lit(TemporalQuality.POSTGAME.value).alias("temporal_quality"),
        ts.alias("available_at"),
        ts.alias("observed_at"),
    )
