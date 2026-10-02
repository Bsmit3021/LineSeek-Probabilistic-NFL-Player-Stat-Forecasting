"""Artifact contracts: required columns and dtypes for every persisted table.

Extra columns are allowed; required columns must exist with the exact dtype. Datetimes are
always Datetime("us", "UTC"). Raw source pulls are stored as Parquet under
data/raw/{source}/season={season}/ alongside a provenance sidecar; derived artifacts live in
the DuckDB warehouse under the table names below.
"""

import polars as pl

from lineseek.contracts.identifiers import GAME_ID_RE, PLAYER_ID_RE
from lineseek.contracts.targets import PROP_MARKETS, LabelStatus

UTC_TS = pl.Datetime("us", "UTC")

_PROVENANCE = {
    "source": pl.String,
    "source_version": pl.String,
    "temporal_quality": pl.String,
    "available_at": UTC_TS,
    "observed_at": UTC_TS,
}

ARTIFACT_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    # Every raw nflverse pull: source columns (frozen in raw_schemas.json) plus provenance.
    "raw_pull": dict(_PROVENANCE),
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
    # Postgame ground truth, unique on (game_id, player_id, market); never a model input.
    # `actual` is null only when label_status is did_not_play (see docs/adr/0002).
    "prop_labels": {
        "game_id": pl.String,
        "player_id": pl.String,
        "season": pl.Int32,
        "week": pl.Int32,
        "team": pl.String,
        "opponent": pl.String,
        "market": pl.String,
        "actual": pl.Float64,
        "played": pl.Boolean,
        "label_status": pl.String,
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
    # Model inputs, unique on (game_id, player_id, market, decision_time); feature columns are
    # extra and defined per feature-set version. max_available_at is null if nothing contributed.
    "features": {
        "game_id": pl.String,
        "player_id": pl.String,
        "season": pl.Int32,
        "week": pl.Int32,
        "team": pl.String,
        "opponent": pl.String,
        "market": pl.String,
        "kickoff_utc": UTC_TS,
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
    # Phase 4A deterministic baselines (docs/adr/0004): one row per
    # (game_id, player_id, market, decision_time, baseline); prediction is null without history.
    # Built from feature rows (and, for EWMA, Phase 3 eligible history) only; never from labels.
    "baseline_predictions": {
        "game_id": pl.String,
        "player_id": pl.String,
        "season": pl.Int32,
        "week": pl.Int32,
        "team": pl.String,
        "opponent": pl.String,
        "market": pl.String,
        "kickoff_utc": UTC_TS,
        "decision_time": UTC_TS,
        "baseline": pl.String,
        "prediction": pl.Float64,
        "history_n": pl.Int32,
        "prior_games_n": pl.Int32,
        "feature_set_version": pl.String,
        "baseline_set_version": pl.String,
    },
    # Walk-forward evaluation folds (docs/adr/0004): one fold per test decision_time; training
    # rows are those whose outcome was available (kickoff + POSTGAME_STATS_LAG) by then.
    "walk_forward_folds": {
        "fold_id": pl.Int32,
        "decision_time": UTC_TS,
        "season": pl.Int32,
        "first_week": pl.Int32,
        "last_week": pl.Int32,
        "test_games": pl.Int32,
        "test_rows": pl.Int32,
        "test_rows_primary": pl.Int32,
        "test_rows_outcome_admitted": pl.Int32,
        "train_rows": pl.Int32,
        "train_rows_primary": pl.Int32,
        "train_first_kickoff": UTC_TS,
        "train_last_available_at": UTC_TS,
        "started_unavailable_rows": pl.Int32,
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
        _validate_features(df)
    if artifact == "prop_labels":
        _validate_prop_labels(df)
    if artifact == "baseline_predictions":
        _validate_baseline_predictions(df)
    if artifact == "walk_forward_folds":
        _validate_folds(df)
    return df


PROP_LABEL_KEY = ("game_id", "player_id", "market")
BASELINE_PREDICTION_KEY = ("game_id", "player_id", "market", "decision_time", "baseline")
FEATURE_KEY = ("game_id", "player_id", "market", "decision_time")
# Target-game outcomes; their presence in a feature artifact is a leak by construction.
LABEL_ONLY_COLUMNS = ("actual", "played", "label_status")
FEATURE_QUALITIES = ("exact", "observed", "inferred")


def _key_checks(artifact: str, key: tuple[str, ...], nullable: tuple[str, ...]) -> dict:
    """Identifier rules shared by player-game-market artifacts."""
    parts = pl.col("game_id").str.split("_")
    away, home = parts.list.get(2, null_on_oob=True), parts.list.get(3, null_on_oob=True)
    return {
        "null required column": pl.any_horizontal(
            pl.col(c).is_null() for c in ARTIFACT_SCHEMAS[artifact] if c not in nullable
        ),
        f"duplicate ({', '.join(key)})": pl.struct(key).is_duplicated(),
        "unsupported market": ~pl.col("market").is_in(list(PROP_MARKETS)),
        "invalid player_id": ~pl.col("player_id").str.contains(PLAYER_ID_RE.pattern),
        "invalid game_id": ~pl.col("game_id").str.contains(GAME_ID_RE.pattern),
        "season/week disagree with game_id": (
            (pl.col("season") != parts.list.get(0, null_on_oob=True).cast(pl.Int32, strict=False))
            | (pl.col("week") != parts.list.get(1, null_on_oob=True).cast(pl.Int32, strict=False))
        ),
        "season/week out of range": (pl.col("season") < 1999) | ~pl.col("week").is_between(1, 22),
        "team/opponent not the game's two teams": ~(
            ((pl.col("team") == away) & (pl.col("opponent") == home))
            | ((pl.col("team") == home) & (pl.col("opponent") == away))
        ),
    }


def _raise_violations(artifact: str, df: pl.DataFrame, checks: dict) -> None:
    # Nulls are caught by the null check; other rules only fire on definite violations.
    counts = df.select(e.fill_null(False).sum().alias(k) for k, e in checks.items())
    problems = [f"{n} row(s) {k}" for k, n in counts.row(0, named=True).items() if n]
    if problems:
        raise ContractViolation(f"{artifact}: " + "; ".join(problems))


def _validate_features(df: pl.DataFrame) -> None:
    """Grain, identifier and decision-time rules for features; feature columns are extra."""
    leaked = [c for c in LABEL_ONLY_COLUMNS if c in df.columns]
    if leaked:
        raise ContractViolation(f"features: label-only column(s) present: {leaked}")
    _raise_violations("features", df, {
        **_key_checks("features", FEATURE_KEY, ("max_available_at",)),
        "decision_time not before kickoff": pl.col("decision_time") >= pl.col("kickoff_utc"),
        "use data after decision_time": pl.col("max_available_at") > pl.col("decision_time"),
        "worst_temporal_quality not usable as a feature":
            ~pl.col("worst_temporal_quality").is_in(FEATURE_QUALITIES),
    })


def _validate_prop_labels(df: pl.DataFrame) -> None:
    """Domain rules for prop_labels; all violations are reported together."""
    actual, played, status = pl.col("actual"), pl.col("played"), pl.col("label_status")
    nonneg = [k for k, m in PROP_MARKETS.items() if m.nonnegative]
    checks = {
        **_key_checks("prop_labels", PROP_LABEL_KEY, ("actual",)),
        "temporal_quality is not postgame": pl.col("temporal_quality") != "postgame",
        "unknown label_status": ~status.is_in([s.value for s in LabelStatus]),
        "non-finite or non-integer actual": actual.is_not_null()
        & (~actual.is_finite() | (actual != actual.round())),
        "negative count market": pl.col("market").is_in(nonneg) & (actual < 0),
        "recorded row not played or null": (status == LabelStatus.RECORDED)
        & (~played | actual.is_null()),
        "participant_no_stats_row not played or nonzero": (
            status == LabelStatus.PARTICIPANT_NO_STATS_ROW
        ) & (~played | (actual != 0) | actual.is_null()),
        "did_not_play row played or has actual": (status == LabelStatus.DID_NOT_PLAY)
        & (played | actual.is_not_null()),
    }
    _raise_violations("prop_labels", df, checks)


def _validate_baseline_predictions(df: pl.DataFrame) -> None:
    """Grain, identifiers and decision-time rules; prediction is null only without history."""
    pred, n = pl.col("prediction"), pl.col("history_n")
    nonneg = [k for k, m in PROP_MARKETS.items() if m.nonnegative]
    _raise_violations("baseline_predictions", df, {
        **_key_checks("baseline_predictions", BASELINE_PREDICTION_KEY, ("prediction",)),
        "decision_time not before kickoff": pl.col("decision_time") >= pl.col("kickoff_utc"),
        "non-finite prediction": pred.is_not_null() & ~pred.is_finite(),
        "negative prediction for a count market": pl.col("market").is_in(nonneg) & (pred < 0),
        "prediction null iff history_n == 0 violated": pred.is_null() != (n == 0),
        "history_n outside [0, prior_games_n]": (n < 0) | (n > pl.col("prior_games_n")),
    })


def _validate_folds(df: pl.DataFrame) -> None:
    """Folds are chronological, one per decision_time, and train only on available outcomes."""
    dt = pl.col("decision_time")
    _raise_violations("walk_forward_folds", df, {
        "null required column": pl.any_horizontal(
            pl.col(c).is_null() for c in ARTIFACT_SCHEMAS["walk_forward_folds"]
            if c not in ("train_first_kickoff", "train_last_available_at")),
        "duplicate fold_id or decision_time": pl.col("fold_id").is_duplicated()
        | dt.is_duplicated(),
        "fold_id not 0..n-1 in decision_time order": pl.col("fold_id")
        != dt.rank("ordinal").cast(pl.Int32) - 1,
        "training outcome available after decision_time": pl.col("train_last_available_at") > dt,
        "training kickoff not before decision_time": pl.col("train_first_kickoff") >= dt,
        "test row counts inconsistent": (pl.col("test_rows_primary")
                                         + pl.col("test_rows_outcome_admitted")
                                         != pl.col("test_rows")),
        "train primary rows exceed train rows": pl.col("train_rows_primary") > pl.col("train_rows"),
    })
