"""Phase 4C.1 discrete-aware CDF (docs/adr/0007): P(Y <= k | X, participated) at integer k.

One LightGBM binary classifier per market, threshold k and fold. It uses the audited Phase 4B
feature matrix, training population and availability rule, and the Phase 4C zero-model
parameters (Phase 4B configuration, objective "binary"). For count markets the k = 0
classifier has exactly the Phase 4C zero model's target I(y == 0), so CDF(0) equals P(Y = 0)
by construction. Raw per-threshold probabilities are made monotone per row after recording
violations: rearrangement (sorting across ascending thresholds), anchored at CDF(0) for count
markets so the zero model's probability is never altered.
"""

from collections.abc import Mapping

import lightgbm as lgb
import numpy as np
import polars as pl

from lineseek.contracts import PROP_MARKETS
from lineseek.features.player_game import SeasonHistory, _window
from lineseek.models.distribution import ZERO_PARAMS

CDF_SET_VERSION = "1.0.0"
CDF_PARAMS = ZERO_PARAMS  # Phase 4B LightGBM configuration with objective "binary"
GRID_LEVELS = (0.01, 0.025, 0.05, *(i / 100 for i in range(10, 95, 5)), 0.95, 0.975, 0.99)
CENTRAL = (0.25, 0.90)  # dense central range of the nonzero warm-up outcomes
YARDAGE_STEP = 5
GRID_RULE = (
    "One global grid per market from warm-up (2018-2020) participant primary-cohort outcomes, "
    "all available before the first evaluation fold. Z = nonzero outcomes. Grid = {0} U "
    "{floor(Q_p(Z)) : p in GRID_LEVELS (linear quantiles)} U (count markets: every integer in "
    "[floor(Q_.25(Z)), floor(Q_.90(Z))]) U (yardage markets: {-1} U multiples of 5 in that "
    "range).")


class CDFError(ValueError):
    pass


def threshold_grid(market: str, warmup_outcomes: np.ndarray) -> list[int]:
    """Pre-registered integer thresholds for one market (see GRID_RULE)."""
    z = warmup_outcomes[warmup_outcomes != 0]
    if not len(z):
        raise CDFError(f"{market}: no nonzero warm-up outcomes")
    grid = {0, *np.floor(np.quantile(z, GRID_LEVELS, method="linear")).astype(int).tolist()}
    lo, hi = np.floor(np.quantile(z, CENTRAL, method="linear")).astype(int).tolist()
    if PROP_MARKETS[market].nonnegative:
        grid |= set(range(lo, hi + 1))
    else:
        grid |= {-1, *(k for k in range(lo, hi + 1) if k % YARDAGE_STEP == 0)}
    return sorted(grid)


def fit_predict_le(thresholds: list[int], x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray) -> np.ndarray:
    """(n_test, len(thresholds)) raw P(Y <= k) from one binary classifier per threshold."""
    if len(x_train) != len(y_train) or np.isnan(y_train).any():
        raise CDFError("training targets must be complete and aligned")
    out = np.empty((len(x_test), len(thresholds)))
    for j, k in enumerate(thresholds):
        z = (y_train <= k).astype(int)
        if z.min() == z.max():  # one class only: the probability is that class
            out[:, j] = float(z[0])
            continue
        model = lgb.LGBMClassifier(**CDF_PARAMS).fit(x_train, z)
        out[:, j] = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    return out


def monotone_repair(raw: np.ndarray, anchor_first: bool = False) -> dict[str, np.ndarray]:
    """Monotone CDF across ascending thresholds; raw violations are recorded first.

    Plain rearrangement sorts each row. With `anchor_first` (count markets, whose first
    threshold is k = 0) the first column is kept, so CDF(0) stays equal to the calibrated
    zero model P(Y = 0); the remaining columns are sorted and floored at it."""
    viol = np.diff(raw, axis=1) < 0
    if anchor_first:
        rest = np.maximum(np.sort(raw[:, 1:], axis=1), raw[:, :1])
        repaired = np.concatenate([raw[:, :1], rest], axis=1)
    else:
        repaired = np.sort(raw, axis=1)
    return {"repaired": repaired, "violations": viol.sum(axis=1),
            "changed": repaired != raw, "repair_max": np.abs(repaired - raw).max(axis=1)}


def pmf(cdf: np.ndarray) -> dict[str, np.ndarray]:
    """Interval masses from a monotone CDF on the grid: lower tail P(Y <= k_0), interior
    P(k_{i-1} < Y <= k_i), upper tail P(Y > k_max). They sum to 1 by construction."""
    return {"lower": cdf[:, 0], "interior": np.diff(cdf, axis=1), "upper": 1 - cdf[:, -1]}


def half_point(cdf_row: Mapping[int, float], line: float) -> tuple[float, float]:
    """(P(under), P(over)) for a half-point line k + 0.5 with k on the grid."""
    k = line - 0.5
    if k != int(k) or int(k) not in cdf_row:
        raise CDFError(f"line {line} is not k + 0.5 for a grid threshold k")
    under = float(cdf_row[int(k)])
    return under, 1.0 - under


def empirical_cdf(targets: pl.DataFrame, histories: Mapping[int, SeasonHistory],
                  grids: Mapping[str, list[int]]) -> pl.DataFrame:
    """Per (game_id, player_id, market, threshold): share of the player's eligible history
    E <= k (same `_window` as the feature store) and |E|."""
    seasons = sorted(set(targets["season"]))
    need = sorted({s for t in seasons for s in (t - 1, t) if s in histories})
    app = pl.concat([histories[s].appearances for s in need])
    t = targets.select("game_id", "player_id", "season", "decision_time").unique()
    w = _window(t, app, "player_id", ["game_id", "player_id"])
    parts = []
    for market, m in PROP_MARKETS.items():
        v = pl.col(m.stat_column)
        parts.append(w.group_by("game_id", "player_id", maintain_order=True).agg(
            pl.len().cast(pl.Int32).alias("history_n"),
            *[(v <= k).mean().alias(f"k{k}") for k in grids[market]]).unpivot(
            index=["game_id", "player_id", "history_n"], variable_name="threshold",
            value_name="p_le").with_columns(
            pl.col("threshold").str.slice(1).cast(pl.Int32), pl.lit(market).alias("market")))
    return pl.concat(parts)


def quantile_approx_cdf(q: np.ndarray, k: float) -> np.ndarray:
    """Phase 4C diagnostic CDF from five sorted quantiles (rows of Q10..Q90): F = 0.05 below
    Q10, 0.95 above Q90; at or between knots, the level of the last knot <= k (the largest
    level among tied knots) interpolated linearly towards the next distinct knot."""
    taus = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
    idx = (q <= k).sum(axis=1) - 1
    out = np.full(len(q), 0.95)
    out[idx < 0] = 0.05
    inner = (idx >= 0) & (idx < len(taus) - 1)
    i = idx[inner]
    rows = q[inner]
    left, right = rows[np.arange(len(i)), i], rows[np.arange(len(i)), i + 1]
    frac = np.where(right > left, (k - left) / np.where(right > left, right - left, 1), 0.0)
    out[inner] = taus[i] + (taus[i + 1] - taus[i]) * frac
    at_top = idx == len(taus) - 1
    out[at_top & (q[:, -1] == k)] = taus[-1]
    return out
