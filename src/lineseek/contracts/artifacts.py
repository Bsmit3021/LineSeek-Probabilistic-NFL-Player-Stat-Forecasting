"""Artifact contracts: required columns and dtypes for every persisted table.

Extra columns are allowed; required columns must exist with the exact dtype. Datetimes are
always Datetime("us", "UTC"). Raw source pulls are stored as Parquet under
data/raw/{source}/season={season}/ alongside a provenance sidecar; derived artifacts live in
the DuckDB warehouse under the table names below.
"""

import polars as pl

UTC_TS = pl.Datetime("us", "UTC")

_PROVENANCE = {
    "source": pl.String,
    "source_version": pl.String,
    "temporal_quality": pl.String,
    "available_at": UTC_TS,
    "observed_at": UTC_TS,
}

ARTIFACT_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    # One row per game. kickoff_utc derived via contracts.temporal.kickoff_utc.
    "games": {
        "game_id": pl.String,
        "season": pl.Int32,
        "week": pl.Int32,
        "season_type": pl.String,
        "home_team": pl.String,
        "away_team": pl.String,
        "kickoff_utc": UTC_TS,
    },
    # Realized labels: one row per (game_id, player_id, market) for every projected player,
    # including those who missed the game. `actual` is null when there is no stats row;
    # `played` is the participation flag used by the market track (null until known).
    "prop_labels": {
        "game_id": pl.String,
        "player_id": pl.String,
        "team": pl.String,
        "market": pl.String,
        "actual": pl.Float64,
        "played": pl.Boolean,
        **_PROVENANCE,
    },
    # Timestamped sportsbook lines: one row per (event, book, market, player, side, snapshot).
    "prop_lines": {
        "game_id": pl.String,
        "player_id": pl.String,
        "player_name_raw": pl.String,
        "book": pl.String,
        "market": pl.String,
        "side": pl.String,
        "line": pl.Float64,
        "price_american": pl.Int32,
        "provider_event_id": pl.String,
        **_PROVENANCE,
    },
    # Model inputs: one row per (game_id, player_id, decision_time). Feature columns are extra.
    "features": {
        "game_id": pl.String,
        "player_id": pl.String,
        "decision_time": UTC_TS,
        "max_available_at": UTC_TS,
        "worst_temporal_quality": pl.String,
        "feature_set_version": pl.String,
    },
    # Model outputs: distribution summary per (game_id, player_id, market, decision_time).
    "predictions": {
        "game_id": pl.String,
        "player_id": pl.String,
        "market": pl.String,
        "decision_time": UTC_TS,
        "model_version": pl.String,
        "mean": pl.Float64,
        "p_over": pl.Float64,
        "line": pl.Float64,
    },
    # Graded predictions: one row per prediction per track ("forecast" | "market").
    # `actual` is the track-resolved value (0 for non-participants on forecast, null on void).
    "grades": {
        "game_id": pl.String,
        "player_id": pl.String,
        "market": pl.String,
        "decision_time": UTC_TS,
        "model_version": pl.String,
        "track": pl.String,
        "line": pl.Float64,
        "actual": pl.Float64,
        "outcome": pl.String,
    },
}


class ContractViolation(ValueError):
    pass


def validate_frame(df: pl.DataFrame, artifact: str) -> pl.DataFrame:
    schema = ARTIFACT_SCHEMAS[artifact]
    missing = [c for c in schema if c not in df.columns]
    wrong = {c: (str(df.schema[c]), str(t)) for c, t in schema.items() if c in df.columns
             and df.schema[c] != t}
    if missing or wrong:
        raise ContractViolation(f"{artifact}: missing={missing} wrong_dtype={wrong}")
    if artifact == "features":
        leaked = df.filter(pl.col("max_available_at") > pl.col("decision_time")).height
        if leaked:
            raise ContractViolation(f"features: {leaked} row(s) use data after decision_time")
    return df
