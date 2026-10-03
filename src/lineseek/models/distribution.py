"""Phase 4C distributional models (docs/adr/0006).

Per market and fold, on the exact Phase 4B training population and feature matrix:

- five LightGBM quantile models (objective "quantile", alpha = 0.10/0.25/0.50/0.75/0.90),
- a squared-error LightGBM mean comparator ("regression"),
- a LightGBM classifier for P(actual == 0 | X, participated) ("binary").

Every parameter equals the audited Phase 4B LightGBM configuration except the objective (and
alpha for quantiles). Quantiles are repaired by row-wise monotone rearrangement (sorting;
Chernozhukov, Fernandez-Val & Galichon 2010) only after the raw crossing rate is recorded; count
markets are then clipped at 0. The empirical baseline uses the player's Phase 3 eligible
history (same `_window` as the feature store), or a market-level fallback from the fold's
training targets when the player has none.
"""

from collections.abc import Mapping
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from lineseek.contracts import PROP_MARKETS
from lineseek.evaluation.scoring import QCOLS, QUANTILES
from lineseek.features.player_game import SeasonHistory, _window
from lineseek.models.learned import COUNT_MARKETS, LGBM_PARAMS

DISTRIBUTION_SET_VERSION = "1.0.0"
COMPONENTS = (*QCOLS, "mean", "zero")
_BASE = {k: v for k, v in LGBM_PARAMS.items() if k != "objective"}
QUANTILE_PARAMS: dict[str, dict[str, Any]] = {
    c: {**_BASE, "objective": "quantile", "alpha": q} for c, q in zip(QCOLS, QUANTILES,
                                                                       strict=True)}
MEAN_PARAMS: dict[str, Any] = {**_BASE, "objective": "regression"}
ZERO_PARAMS: dict[str, Any] = {**_BASE, "objective": "binary"}
EMPIRICAL_RULE = ("player's Phase 3 eligible history E (available_at <= decision_time, target "
                  "season or the one before, appearances only); quantiles by linear "
                  "interpolation between order statistics (numpy/R type 7); mean = mean(E); "
                  "p_zero = share of zeros in E. If E is empty: the same statistics of the "
                  "fold's training targets for the market (market-level fallback).")


class DistributionError(ValueError):
    pass


def params(component: str) -> dict[str, Any]:
    if component in QUANTILE_PARAMS:
        return QUANTILE_PARAMS[component]
    return MEAN_PARAMS if component == "mean" else ZERO_PARAMS


def fit_predict_component(component: str, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray) -> np.ndarray:
    """Fit on training rows only; for "zero" the target is 1{y == 0} and the output P(y == 0)."""
    if len(x_train) != len(y_train) or np.isnan(y_train).any():
        raise DistributionError("training targets must be complete and aligned")
    if component == "zero":
        z = (y_train == 0).astype(int)
        if z.min() == z.max():  # one class only: the probability is that class
            return np.full(len(x_test), float(z[0]))
        model = lgb.LGBMClassifier(**ZERO_PARAMS).fit(x_train, z)
        return model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    return np.asarray(lgb.LGBMRegressor(**params(component)).fit(x_train, y_train)
                      .predict(x_test), dtype=np.float64)


def repair_and_clip(market: str, raw: np.ndarray) -> dict[str, np.ndarray]:
    """raw: (n, 5) quantiles in QUANTILES order. Sort each row (monotone rearrangement), then
    clip count markets at 0. Returns final quantiles plus crossing/repair/clipping records."""
    crossed = (np.diff(raw, axis=1) < 0).any(axis=1)
    repaired = np.sort(raw, axis=1)
    repair_max = np.abs(repaired - raw).max(axis=1)
    if market in COUNT_MARKETS:
        final = np.maximum(repaired, 0.0)
    else:
        final = repaired
    return {"final": final, "repaired": repaired, "crossed": crossed, "repair_max": repair_max,
            "clipped": (final != repaired).sum(axis=1)}


def empirical_history(targets: pl.DataFrame, histories: Mapping[int, SeasonHistory]
                      ) -> pl.DataFrame:
    """Per (game_id, player_id, market): empirical quantiles, mean, zero share and n of E."""
    seasons = sorted(set(targets["season"]))
    need = sorted({s for t in seasons for s in (t - 1, t) if s in histories})
    app = pl.concat([histories[s].appearances for s in need])
    t = targets.select("game_id", "player_id", "season", "decision_time").unique()
    w = _window(t, app, "player_id", ["game_id", "player_id"])
    parts = []
    for key, m in PROP_MARKETS.items():
        v = pl.col(m.stat_column).cast(pl.Float64)
        parts.append(w.group_by("game_id", "player_id", maintain_order=True).agg(
            *[v.quantile(q, interpolation="linear").alias(c) for c, q in
              zip(QCOLS, QUANTILES, strict=True)],
            v.mean().alias("mean_prediction"), (v == 0).mean().alias("p_zero"),
            pl.len().cast(pl.Int32).alias("history_n")).with_columns(pl.lit(key).alias("market")))
    return pl.concat(parts)


def fallback_distribution(y_train: np.ndarray) -> dict[str, float]:
    """Market-level distribution of the fold's training targets (cold-start rows only)."""
    qs = np.quantile(y_train, QUANTILES, method="linear")
    return {**dict(zip(QCOLS, qs.tolist(), strict=True)), "mean_prediction": float(y_train.mean()),
            "p_zero": float((y_train == 0).mean())}
