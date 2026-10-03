"""Proper scores and calibration for quantile forecasts (Phase 4C; docs/adr/0006).

- Pinball (quantile) loss: rho_tau(y, q) = (1{y <= q} - tau) * (q - y)  (>= 0).
- Interval score (Bracher et al. 2021, after Gneiting & Raftery 2007), central (1 - a) interval
  [l, u]: IS_a = (u - l) + 2/a (l - y) 1{y < l} + 2/a (y - u) 1{y > u}.
- Weighted interval score, K intervals + median m (Bracher et al. 2021, eq. 1-2):
  WIS = 1/(K + 1/2) * (w0 |y - m| + sum_k w_k IS_{a_k}), w0 = 1/2, w_k = a_k / 2.
  Equivalent to 2/(2K + 1) * sum of pinball losses over the 2K + 1 quantiles (eq. 4).
- Calibration for integer-valued outcomes (Czado, Gneiting & Held 2009): with ties a quantile's
  coverage is the bracket [P(y < Q), P(y <= Q)]; the error is the distance from the nominal
  level to that bracket (0 inside). Intervals likewise use [P(l < y < u), P(l <= y <= u)].
"""

from itertools import pairwise

import numpy as np
import polars as pl

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QCOLS = ("q10", "q25", "q50", "q75", "q90")
INTERVALS = {"50": ("q25", "q75", 0.50), "80": ("q10", "q90", 0.20)}  # name: (l, u, alpha)
# Zero-probability calibration bins, fixed before any result.
ZERO_BINS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0)
LOG_LOSS_EPS = 1e-15


def pinball(y: pl.Expr, q: pl.Expr, tau: float) -> pl.Expr:
    return (pl.when(y <= q).then(1.0).otherwise(0.0) - tau) * (q - y)


def interval_score(y: pl.Expr, lo: pl.Expr, hi: pl.Expr, alpha: float) -> pl.Expr:
    return ((hi - lo) + 2 / alpha * pl.when(y < lo).then(lo - y).otherwise(0.0)
            + 2 / alpha * pl.when(y > hi).then(y - hi).otherwise(0.0))


def wis(y: pl.Expr, cols: dict[str, pl.Expr]) -> pl.Expr:
    """WIS over the median and the 50% / 80% central intervals (K = 2)."""
    k = len(INTERVALS)
    total = 0.5 * (y - cols["q50"]).abs()
    for lo, hi, alpha in INTERVALS.values():
        total = total + alpha / 2 * interval_score(y, cols[lo], cols[hi], alpha)
    return total / (k + 0.5)


def score_columns(actual: str = "actual", prefix: str = "") -> list[pl.Expr]:
    """Per-row scores: pinball per quantile, mean pinball, WIS, interval hits (open/closed)."""
    y = pl.col(actual)
    c = {q: pl.col(f"{prefix}{q}") for q in QCOLS}
    out = [pinball(y, c[q], t).alias(f"pinball_{q}") for q, t in zip(QCOLS, QUANTILES,
                                                                       strict=True)]
    out.append(pl.mean_horizontal(*[pinball(y, c[q], t) for q, t in zip(QCOLS, QUANTILES,
                                                                         strict=True)])
               .alias("pinball_mean"))
    out.append(wis(y, c).alias("wis"))
    for name, (lo, hi, _) in INTERVALS.items():
        out += [((y >= c[lo]) & (y <= c[hi])).alias(f"in{name}_closed"),
                ((y > c[lo]) & (y < c[hi])).alias(f"in{name}_open"),
                (c[hi] - c[lo]).alias(f"width{name}")]
    for q in QCOLS:
        out += [(y <= c[q]).alias(f"le_{q}"), (y < c[q]).alias(f"lt_{q}")]
    return out


def bracket_error(target: float, low: float, high: float) -> float:
    """Distance from `target` to [low, high]; 0 if inside (discrete-outcome calibration)."""
    if low <= target <= high:
        return 0.0
    return low - target if target < low else target - high


def zero_scores(p: pl.Expr, is_zero: pl.Expr) -> list[pl.Expr]:
    pc = p.clip(LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)
    z = is_zero.cast(pl.Float64)
    return [((p - z) ** 2).alias("brier"),
            (-(z * pc.log() + (1 - z) * (1 - pc).log())).alias("log_loss")]


def zero_bin() -> pl.Expr:
    edges = ZERO_BINS
    e = pl.lit(None, dtype=pl.String)
    for lo, hi in reversed(list(pairwise(edges))):
        cond = (pl.col("p_zero") >= lo) & ((pl.col("p_zero") < hi) if hi < 1 else
                                            (pl.col("p_zero") <= hi))
        e = pl.when(cond).then(pl.lit(f"[{lo:.2f},{hi:.2f}{']' if hi == 1 else ')'}")).otherwise(e)
    return e.alias("p_zero_bin")


def cluster_bootstrap_mean(df: pl.DataFrame, value: str, cluster: str, reps: int, seed: int
                           ) -> tuple[float, float, float]:
    """Mean of `value` with a 95% percentile interval resampling whole clusters (games)."""
    g = df.group_by(cluster, maintain_order=True).agg(pl.col(value).sum().alias("s"),
                                                      pl.len().alias("n")).sort(cluster)
    s, n = g["s"].to_numpy(), g["n"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(reps, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(s.sum() / n.sum()), float(lo), float(hi)
