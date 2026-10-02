"""Phase 4A deterministic baselines: features (+ Phase 3 eligible history) -> point predictions.

Prediction generation never reads labels: the five feature baselines select only key and
feature columns, and EWMA uses the same eligible history as the feature store (`_window`:
available_at <= decision_time, target season or the one before, player appearances only).
Formulas and versioning: docs/adr/0004.
"""

from collections.abc import Mapping

import polars as pl

from lineseek.contracts import PROP_MARKETS, validate_frame
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.features.player_game import SeasonHistory, _window

BASELINE_SET_VERSION = "1.0.0"
EWMA_SPAN = 5
EWMA_ALPHA = 2 / (EWMA_SPAN + 1)  # 1/3, fixed; never tuned on evaluation results
# Frozen names, in reporting order. Value: (feature column, history-count expression).
FEATURE_BASELINES: dict[str, tuple[str, pl.Expr]] = {
    "prev_game": ("market_prev", pl.min_horizontal(pl.col("prior_games_n"), pl.lit(1))),
    "roll3_mean": ("market_roll3_mean", pl.col("market_roll3_n")),
    "roll5_mean": ("market_roll5_mean", pl.col("market_roll5_n")),
    "roll8_mean": ("market_roll8_mean", pl.col("market_roll8_n")),
    "season_mean": ("market_season_mean", pl.col("market_season_games")),
}
EWMA_BASELINE = "ewma_span5"
BASELINES = (*FEATURE_BASELINES, EWMA_BASELINE)
KEY = ("game_id", "player_id", "market")
ROW_COLUMNS = ("game_id", "player_id", "season", "week", "team", "opponent", "market",
               "kickoff_utc", "decision_time", "prior_games_n", "feature_set_version")
FEATURE_INPUTS = (*ROW_COLUMNS, *(c for c, _ in FEATURE_BASELINES.values()), "market_roll3_n",
                  "market_roll5_n", "market_roll8_n", "market_season_games")
I32, F64 = pl.Int32, pl.Float64


class BaselineIntegrityError(ValueError):
    pass


def _rows(df: pl.DataFrame, baseline: str, prediction: pl.Expr, n: pl.Expr) -> pl.DataFrame:
    return df.select(
        *ROW_COLUMNS[:9], pl.lit(baseline).alias("baseline"),
        prediction.cast(F64).alias("prediction"), n.cast(I32).alias("history_n"),
        "prior_games_n", "feature_set_version",
        pl.lit(BASELINE_SET_VERSION).alias("baseline_set_version"))


def predict_feature_baselines(features: pl.DataFrame) -> pl.DataFrame:
    """B1-B5: copies of audited Phase 3 columns. Reads only FEATURE_INPUTS (no label columns)."""
    f = features.select(FEATURE_INPUTS)
    return pl.concat([_rows(f, name, pl.col(col), n)
                      for name, (col, n) in FEATURE_BASELINES.items()])


def ewma_history(features: pl.DataFrame, histories: Mapping[int, SeasonHistory]) -> pl.DataFrame:
    """B6 per (game_id, player_id, market): EWMA over the Phase 3 eligible history, oldest first.

    EWMA_1 = v_1; EWMA_t = alpha * v_t + (1 - alpha) * EWMA_(t-1); the prediction is the value
    after the most recent eligible game. Also returns games used and the most recent value.
    """
    targets = features.select("game_id", "player_id", "season", "decision_time").unique()
    seasons = sorted(set(targets["season"]))
    need = {s for t in seasons for s in (t - 1, t) if s in histories}
    missing = [t for t in seasons if t not in histories]
    if missing:
        raise BaselineIntegrityError(f"EWMA needs the history of target season(s) {missing}")
    app = pl.concat([histories[s].appearances for s in sorted(need)])
    stats = {m.stat_column for m in PROP_MARKETS.values()}
    order = ["h_kickoff_utc", "h_game_id"]
    agg = [pl.len().cast(I32).alias("_n")]
    for c in sorted(stats):
        v = pl.col(c).cast(F64).sort_by(order)
        agg += [v.ewm_mean(alpha=EWMA_ALPHA, adjust=False).last().alias(f"{c}__ewma"),
                v.last().alias(f"{c}__last")]
    group = ["game_id", "player_id"]
    wide = _window(targets, app, "player_id", group).group_by(group).agg(agg)
    return pl.concat([
        wide.select(*group, pl.lit(k).alias("market"), "_n",
                    pl.col(f"{m.stat_column}__ewma").alias("ewma"),
                    pl.col(f"{m.stat_column}__last").alias("last"))
        for k, m in PROP_MARKETS.items()
    ])


def predict_baselines(features: pl.DataFrame, histories: Mapping[int, SeasonHistory]
                      ) -> pl.DataFrame:
    """All six baselines in long form, validated against the baseline_predictions contract.

    EWMA must agree with the feature row it is attached to: same number of eligible games as
    prior_games_n and same most recent value as market_prev, else the histories are not the
    ones the features were built from.
    """
    f = features.select(FEATURE_INPUTS)
    e = f.join(ewma_history(f, histories), on=list(KEY), how="left")
    bad = e.filter((pl.col("_n").fill_null(0) != pl.col("prior_games_n"))
                   | ~pl.col("last").eq_missing(pl.col("market_prev")))
    if bad.height:
        raise BaselineIntegrityError(
            f"{bad.height} row(s) whose EWMA history disagrees with the feature row "
            f"(prior_games_n/market_prev): {bad.select(*KEY).head(5).rows()}")
    out = pl.concat([predict_feature_baselines(f),
                     _rows(e, EWMA_BASELINE, pl.col("ewma"), pl.col("_n").fill_null(0))])
    out = out.select(list(ARTIFACT_SCHEMAS["baseline_predictions"])).sort(
        "season", "week", "game_id", "team", "player_id", "market", "baseline")
    return validate_frame(out, "baseline_predictions")
