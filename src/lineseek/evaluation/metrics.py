"""Point-forecast error metrics, participation tracks and grouped uncertainty; docs/adr/0004.

error = prediction - actual (positive mean_error = overprediction). Metrics are computed per
market only; there is no cross-market score and no MAPE (zero outcomes are legitimate).
"""

import numpy as np
import polars as pl

from lineseek.contracts import Track, resolve_actual

# participant: played rows only, actual as labelled (a participant's 0 is a real 0).
# forecast_lock: every row; actual resolved by the Phase 0 forecast-track rule
# (`resolve_actual(..., Track.FORECAST)`: a non-participant scores 0).
TRACKS = ("participant", "forecast_lock")
HISTORY_BUCKETS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-7", 5, 7),
                   ("8+", 8, None))
P90_QUANTILE = 0.9  # linear interpolation between order statistics (numpy/R type 7)
METRIC_COLUMNS = ("n_rows", "n_eligible", "n_predicted", "coverage_rate", "mae", "rmse",
                  "median_ae", "mean_error", "p90_ae", "max_ae")


def track_actuals(labels: pl.DataFrame) -> pl.DataFrame:
    """Labels + actual_participant / actual_forecast_lock (null = not scored on that track).

    Values come from the contract's resolve_actual, evaluated once per distinct
    (actual, played) pair; a null actual is never turned into 0 except by that rule.
    """
    pairs = labels.select("actual", "played").unique()
    resolved = pairs.with_columns(
        pl.Series("actual_forecast_lock",
                  [resolve_actual(a, Track.FORECAST) for a, _ in pairs.iter_rows()],
                  dtype=pl.Float64),
        pl.Series("_market",
                  [resolve_actual(a, Track.MARKET, p) for a, p in pairs.iter_rows()],
                  dtype=pl.Float64),
    )
    out = labels.join(resolved, on=["actual", "played"], how="left", nulls_equal=True)
    # Participant accuracy = market-track point values (non-participants void, participants
    # graded); a played row must have a recorded actual.
    bad = out.filter(pl.col("played") & pl.col("actual").is_null())
    if bad.height:
        raise ValueError(f"{bad.height} played label row(s) without an actual")
    return out.rename({"_market": "actual_participant"})


def history_bucket() -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for name, lo, hi in reversed(HISTORY_BUCKETS):
        cond = pl.col("prior_games_n") >= lo
        if hi is not None:
            cond &= pl.col("prior_games_n") <= hi
        e = pl.when(cond).then(pl.lit(name)).otherwise(e)
    return e.alias("history_bucket")


def error_metrics(df: pl.DataFrame, by: list[str], actual: str) -> pl.DataFrame:
    """Per group: rows, rows eligible on the track (actual not null), rows predicted among them,
    coverage, and MAE/RMSE/median AE/mean error/P90 AE/max AE over predicted eligible rows."""
    eligible = pl.col(actual).is_not_null()
    scored = eligible & pl.col("prediction").is_not_null()
    err = (pl.col("prediction") - pl.col(actual)).filter(scored)
    n_elig, n_pred = eligible.sum().cast(pl.Int64), scored.sum().cast(pl.Int64)
    return df.group_by(by).agg(
        pl.len().cast(pl.Int64).alias("n_rows"),
        n_elig.alias("n_eligible"),
        n_pred.alias("n_predicted"),
        pl.when(n_elig > 0).then(n_pred / n_elig).alias("coverage_rate"),
        err.abs().mean().alias("mae"),
        err.pow(2).mean().sqrt().alias("rmse"),
        err.abs().median().alias("median_ae"),
        err.mean().alias("mean_error"),
        err.abs().quantile(P90_QUANTILE, interpolation="linear").alias("p90_ae"),
        err.abs().max().cast(pl.Float64).alias("max_ae"),
    ).sort(by)


def cluster_bootstrap_diff(df: pl.DataFrame, actual: str, cluster: str, reps: int,
                           seed: int) -> tuple[float, float, float]:
    """Paired MAE difference (prediction - reference) with a percentile interval from
    resampling whole clusters (games), never individual rows.

    `df` has columns prediction, reference, `actual`, `cluster`; rows with any null dropped.
    """
    d = df.drop_nulls(["prediction", "reference", actual]).group_by(cluster).agg(
        ((pl.col("prediction") - pl.col(actual)).abs()
         - (pl.col("reference") - pl.col(actual)).abs()).sum().alias("s"),
        pl.len().alias("n")).sort(cluster)
    s, n = d["s"].to_numpy(), d["n"].to_numpy()
    point = float(s.sum() / n.sum())
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)
