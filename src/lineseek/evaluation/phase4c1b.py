"""Phase 4C.1b run: threshold-conditioned monotone CDF, the final Phase 4C gate (docs/adr/0008).

Reuses the validated Phase 4A run (110 folds, cohorts), the audited Phase 4B training population
and 42 features, the Phase 4C run (zero model, for consistency only) and the frozen Phase 4C.1
run: its threshold grid (asserted equal to the warm-up rule), and its independent-classifier
CDF and empirical CDF predictions as comparators on identical rows. Phase 4C.1 artifacts are
read, never rewritten. Predictions are serialised and checksummed before labels are joined.
Scoring: primary cohort, participant rows, not outcome-admitted (DNP is never a zero).
"""

import hashlib
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from lineseek.contracts import CONTRACT_VERSION, PROP_MARKETS, validate_frame
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c as p4c
from lineseek.evaluation import phase4c1 as p4c1
from lineseek.evaluation.phase4a import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    F3_NOTE,
    POINT_IN_TIME_WARNING,
    UNIVERSE_WARNING,
    WARMUP_SEASONS,
    Inputs,
    _modelling_rows,
    parquet_bytes,
    runtime_versions,
)
from lineseek.evaluation.scoring import LOG_LOSS_EPS, cluster_bootstrap_mean
from lineseek.evaluation.universe import qb_role
from lineseek.evaluation.walkforward import walk_forward_folds
from lineseek.features.player_game import FEATURE_SET_VERSION
from lineseek.models.baselines import BASELINE_SET_VERSION
from lineseek.models.cdf import CDF_SET_VERSION, GRID_RULE, CDFError, pmf
from lineseek.models.cdf_tc import (
    FP_TOL,
    TC_CDF_SET_VERSION,
    TC_FEATURES,
    TC_PARAMS,
    check_monotone,
    fit_predict_tc,
)
from lineseek.models.distribution import DISTRIBUTION_SET_VERSION
from lineseek.models.learned import FEATURES, MODEL_SET_VERSION, SEED

log = logging.getLogger("lineseek.cdf_tc")

RUN_DIR = "phase4c1b"
KEY = ["game_id", "player_id", "market"]
TC, INDEP, EMP = "cdf_tc", "cdf_indep", "empirical"  # INDEP = Phase 4C.1 "cdf_lgbm" (repaired)
PREDICTORS = (TC, INDEP, EMP)
WORKERS = 2        # LightGBM is single-threaded; two processes keep the 8 GB machine out of swap
FOLD_CHUNK = 22    # folds per worker task
N_BINS = 20        # reliability bins of width 0.05 (as Phase 4C.1)
# Pre-registered before any Phase 4C.1b result (ADR 0008).
INFO_MIN, INFO_RATE = 200, (0.05, 0.95)  # informative: >= 200 each side, observed rate in range
CAL_OK, CAL_CONCERN = 0.03, 0.05         # mean informative ECE; worst informative threshold
PERSIST_BIAS, PERSIST_SEASONS = 0.05, 3  # |season bias| >= 5 pp, same sign, in >= 3 seasons
MIN_SEASONS_BETTER = 4                   # Brier better than empirical in >= 4 of 5 seasons
PMF_TOL = 1e-9
NOISE_DRAWS = 200                        # ECE noise floor (diagnostic only)
INTERVALS = ((0.25, 0.75), (0.10, 0.90))
READY, CONDITIONAL, NOT_READY = "READY", "CONDITIONAL", "NOT READY"


def _check(msg: str, df: pl.DataFrame) -> None:
    if df.height:
        raise CDFError(f"{df.height} {msg}: {df.head(5).rows()}")


# ---------------------------------------------------------------------------- inputs

@dataclass
class Phase4C1:
    path: Path
    manifest: dict
    grids: dict[str, list[int]]


def load_phase4c1(path: Path, p4a: p4b.Phase4A, p4cc: p4c1.Phase4C) -> Phase4C1:
    """Frozen Phase 4C.1 run: checksums, lineage (same Phase 4A and Phase 4C runs) and grid.
    Its predictions are read only at scoring time (see `phase4c1_predictions`)."""
    m = json.loads((path / "manifest.json").read_text())
    for k, meta in m["files"].items():
        if hashlib.sha256((path / f"{k}.parquet").read_bytes()).hexdigest() != meta["sha256"]:
            raise CDFError(f"Phase 4C.1 {k}.parquet: checksum does not match its manifest")
    if m["phase4a"]["content_sha256"] != p4a.manifest["content_sha256"]:
        raise CDFError("Phase 4C.1 run was not built on this Phase 4A run")
    if m["phase4c"]["content_sha256"] != p4cc.manifest["content_sha256"]:
        raise CDFError("Phase 4C.1 run was not built on this Phase 4C run")
    g = pl.read_parquet(path / "threshold_grid.parquet").sort("market", "threshold")
    grids = {mk: g.filter(pl.col("market") == mk)["threshold"].to_list() for mk in PROP_MARKETS}
    return Phase4C1(path, m, grids)


def phase4c1_predictions(p: Phase4C1) -> pl.DataFrame:
    return pl.scan_parquet(p.path / "cdf_predictions.parquet").filter(
        pl.col("predictor").is_in(["cdf_lgbm", "empirical"])).select(
        *KEY, "decision_time", "threshold", "p_le",
        pl.col("predictor").replace({"cdf_lgbm": INDEP})).collect()


def frozen_grids(table: pl.DataFrame, folds: pl.DataFrame, eval_seasons: tuple[int, ...],
                 p: Phase4C1) -> dict[str, list[int]]:
    """The warm-up grid rule must reproduce the frozen Phase 4C.1 grid exactly."""
    grids = p4c1.grids_from_warmup(table, folds, eval_seasons)
    if grids != p.grids:
        raise CDFError("threshold grid differs from the frozen Phase 4C.1 grid")
    return grids


# ---------------------------------------------------------------------------- training

@dataclass(frozen=True)
class TCTask:
    market: str
    thresholds: tuple[int, ...]
    x: np.ndarray
    y: np.ndarray
    available: np.ndarray
    decision: np.ndarray
    folds: tuple[tuple[int, int], ...]


def run_tc(task: TCTask) -> dict:
    """One threshold-conditioned fit per fold. Returns test-row indices and probabilities."""
    idx, out, meta = [], [], []
    ok = ~np.isnan(task.y)
    for fold_id, dt in task.folds:
        test = np.flatnonzero(task.decision == dt)
        if not len(test):
            continue
        train = ok & (task.available <= dt)
        if not train.any() or train[test].any() or task.available[train].max() > dt:
            raise CDFError(f"fold {fold_id}: training rows not available by the lock")
        idx.append(test)
        out.append(fit_predict_tc(list(task.thresholds), task.x[train], task.y[train],
                                  task.x[test]))
        yt = task.y[train]
        meta.append({"fold_id": fold_id, "market": task.market, "train_rows": int(train.sum()),
                     "train_stacked_rows": int(train.sum()) * len(task.thresholds),
                     "thresholds": len(task.thresholds), "test_rows": len(test),
                     "train_rate_k0": float((yt <= 0).mean()),
                     "train_last_available_us": int(task.available[train].max())})
    return {"market": task.market, "idx": idx, "pred": out, "meta": meta}


def build_tc_tasks(table: pl.DataFrame, folds: pl.DataFrame, grids: dict[str, list[int]]
                   ) -> list[TCTask]:
    base = p4b.build_tasks(table, folds, ("lightgbm",))
    if any(t.x.shape[1] != len(FEATURES) for t in base):
        raise CDFError("model matrices must be exactly the frozen feature list")
    tasks = [TCTask(t.market, tuple(grids[t.market]), t.x, t.y, t.available, t.decision,
                    t.folds[i:i + FOLD_CHUNK])
             for t in base for i in range(0, len(t.folds), FOLD_CHUNK)]

    def cost(t: TCTask) -> int:  # stacked training rows: largest tasks first (balance)
        ok = ~np.isnan(t.y)
        return sum(int((ok & (t.available <= d)).sum()) for _, d in t.folds) * len(t.thresholds)
    return sorted(tasks, key=lambda t: (-cost(t), t.market, t.folds[0][0]))


META = ["game_id", "player_id", "season", "week", "team", "opponent", "market", "kickoff_utc",
        "decision_time", "fold_id"]


def predict_tc(table: pl.DataFrame, folds: pl.DataFrame, grids: dict[str, list[int]],
               eval_seasons: tuple[int, ...], workers: int = WORKERS
               ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Threshold-conditioned CDF for every evaluation row and grid threshold, the fold metadata
    and the monotonicity record. No label is read except fit-eligible training targets."""
    tasks = build_tc_tasks(table, folds, grids)
    results = []
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for i, r in enumerate(ex.map(run_tc, tasks), 1):
                log.info("task %d/%d done (%s, %d fits)", i, len(tasks), r["market"],
                         len(r["meta"]))
                results.append(r)
    else:
        results = [run_tc(t) for t in tasks]
    common = {"feature_set_version": FEATURE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
              "distribution_set_version": DISTRIBUTION_SET_VERSION,
              "cdf_set_version": TC_CDF_SET_VERSION}
    parts, mono = [], []
    for market, grid in grids.items():
        m = table.filter(pl.col("market") == market).select(*META, "prior_games_n")
        raw = np.full((m.height, len(grid)), np.nan)
        train_n = np.zeros(m.height, dtype=np.int64)
        for r in (r for r in results if r["market"] == market):
            for idx, pred, meta in zip(r["idx"], r["pred"], r["meta"], strict=True):
                raw[idx] = pred
                train_n[idx] = meta["train_rows"]
        keep = m["season"].is_in(list(eval_seasons)).to_numpy()
        raw, train_n = raw[keep], train_n[keep]
        if np.isnan(raw).any():
            raise CDFError(f"{market}: evaluation rows without a prediction")
        mc = check_monotone(raw)
        mono.append({"market": market, "rows": len(raw), "cells": raw.size,
                     "raw_decreasing_cells": int(mc["violations"].sum()),
                     "raw_decreasing_rows": int((mc["violations"] > 0).sum()),
                     "max_raw_decrease": mc["max_decrease"],
                     "fp_cleanup_cells": int(mc["changed"].sum()),
                     "fp_tolerance": FP_TOL})
        mm = m.filter(pl.Series(keep)).with_columns(
            pl.col("prior_games_n").cast(pl.Int32).alias("history_n"),
            pl.Series("train_rows", train_n).cast(pl.Int32)).drop("prior_games_n")
        n, t = raw.shape
        parts.append(pl.DataFrame({
            "row": np.repeat(np.arange(n), t),
            "threshold": np.tile(np.array(grid, dtype=np.int32), n),
            "p_le_raw": raw.reshape(-1), "p_le": mc["final"].reshape(-1),
            "monotonic_repair": mc["changed"].reshape(-1)}).join(
            mm.with_row_index("row").with_columns(pl.col("row").cast(pl.Int64)), on="row"))
    cols = list(ARTIFACT_SCHEMAS["cdf_predictions"])
    out = pl.concat(parts).with_columns(
        pl.lit(TC).alias("predictor"), *[pl.lit(v).alias(k) for k, v in common.items()]).select(
        cols).sort("season", "week", "game_id", "team", "player_id", "market", "threshold")
    validate_frame(out, "cdf_predictions")
    meta = pl.DataFrame([x for r in results for x in r["meta"]]).with_columns(
        pl.from_epoch("train_last_available_us", "us").dt.replace_time_zone("UTC")
        .alias("train_last_available_at")).drop("train_last_available_us").join(
        folds.select("fold_id", "decision_time"), on="fold_id").sort("fold_id", "market")
    _check("fits trained on data unavailable at the decision time",
           meta.filter(pl.col("train_last_available_at") > pl.col("decision_time")))
    return out, meta, pl.DataFrame(mono)


# ---------------------------------------------------------------------------- scoring

def scored_rows(inputs: Inputs, p4a: p4b.Phase4A, eval_seasons: tuple[int, ...]
                ) -> pl.DataFrame:
    """Primary-cohort participant rows (not outcome-admitted) with actual and qb_role. qb_role
    is an evaluation cohort only; it never reaches a model."""
    lab = pl.concat([inputs.labels[s] for s in eval_seasons]).select(
        *KEY, "season", "actual", "played")
    qb = pl.concat([qb_role(inputs.labels[s], inputs.label_raw[s]["schedules"],
                            inputs.label_raw[s]["rosters_weekly"]) for s in eval_seasons])
    rows = (lab.filter(pl.col("played")).join(
        p4a.cohorts.select("game_id", "player_id", "outcome_admitted"),
        on=["game_id", "player_id"]).filter(~pl.col("outcome_admitted"))
        .join(qb, on=["game_id", "player_id"]))
    _check("participant rows without an actual", rows.filter(pl.col("actual").is_null()))
    return rows.select(*KEY, "season", "actual", "qb_role").sort(KEY)


def _matrix(p: pl.DataFrame, grid: list[int], rows: pl.DataFrame) -> np.ndarray:
    """(len(rows), T) p_le aligned to `rows` (sorted by KEY) for one predictor and market."""
    p = p.sort(*KEY, "threshold")
    keys = p.filter(pl.col("threshold") == grid[0]).select(KEY).with_row_index("i")
    if p.height != keys.height * len(grid) or not keys.select(KEY).is_unique().all():
        raise CDFError("predictions are not one row per key and grid threshold")
    if not (p["threshold"].to_numpy().reshape(-1, len(grid)) == np.array(grid)).all():
        raise CDFError("predictions do not cover exactly the grid thresholds")
    j = rows.select(KEY).join(keys, on=KEY, how="left", maintain_order="left")
    if j["i"].null_count():
        raise CDFError(f"{j['i'].null_count()} scored rows without a prediction")
    return p["p_le"].to_numpy().reshape(-1, len(grid))[j["i"].to_numpy()]


@dataclass
class Market:
    rows: pl.DataFrame               # KEY, season, actual, qb_role (sorted by KEY)
    grid: list[int]
    y: np.ndarray                    # (n, T) 1{actual <= k}
    P: dict[str, np.ndarray]         # predictor -> (n, T)
    p_zero: np.ndarray               # Phase 4C zero model P(Y = 0)


def build_markets(tc: pl.DataFrame, other: pl.DataFrame, rows: pl.DataFrame,
                  grids: dict[str, list[int]], p4cc: p4c1.Phase4C) -> dict[str, Market]:
    """Every predictor on the identical scored rows (asserted by key alignment)."""
    out = {}
    for market, grid in grids.items():
        r = rows.filter(pl.col("market") == market)
        P = {TC: _matrix(tc.filter(pl.col("market") == market), grid, r)}
        for name in (INDEP, EMP):
            P[name] = _matrix(other.filter(pl.col("market") == market,
                                           pl.col("predictor") == name), grid, r)
        z = r.select(KEY).join(p4cc.dist.select(*KEY, "p_zero"), on=KEY, how="left",
                               maintain_order="left")["p_zero"]
        if z.null_count():
            raise CDFError(f"{market}: scored rows without a Phase 4C zero probability")
        y = (r["actual"].to_numpy()[:, None] <= np.array(grid)[None, :]).astype(np.float64)
        out[market] = Market(r, grid, y, P, z.to_numpy())
    return out


def ece(p: np.ndarray, y: np.ndarray) -> float:
    """Binned expected calibration error: sum_b n_b/n |mean p_b - rate_b| (20 equal bins)."""
    if not len(p):
        return float("nan")
    b = np.clip(np.floor(p * N_BINS), 0, N_BINS - 1).astype(int)
    return float(np.abs(np.bincount(b, p, N_BINS) - np.bincount(b, y, N_BINS)).sum() / len(p))


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    pc = np.clip(p, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)  # scoring only; stored p unchanged
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def informative(y: np.ndarray) -> np.ndarray:
    """Pre-registered informative-threshold rule (outcome counts only) per column of y."""
    pos = y.sum(axis=0)
    neg = len(y) - pos
    rate = pos / max(len(y), 1)
    return (pos >= INFO_MIN) & (neg >= INFO_MIN) & (rate >= INFO_RATE[0]) & (rate <= INFO_RATE[1])


def cohorts(mk: Market, market: str) -> list[tuple[str, np.ndarray]]:
    c = [("primary", np.ones(len(mk.rows), dtype=bool))]
    if market == "pass_attempts":
        c.append(("qb_role", mk.rows["qb_role"].to_numpy()))
    return c


def noise_floor(p: np.ndarray, seed: list[int]) -> float:
    """Mean ECE of a perfectly calibrated forecaster with these probabilities: y* ~ Bern(p)."""
    rng = np.random.default_rng(seed)
    return float(np.mean([ece(p, (rng.random(len(p)) < p).astype(np.float64))
                          for _ in range(NOISE_DRAWS)]))


def threshold_metrics(markets: dict[str, Market], eval_seasons: tuple[int, ...]
                      ) -> pl.DataFrame:
    """Per cohort x period x predictor x market x threshold. `informative` is fixed per cohort,
    market and threshold from all evaluation seasons; noise floor for cdf_tc, all seasons."""
    all_ = f"{min(eval_seasons)}-{max(eval_seasons)}"
    out = []
    for mi, (market, mk) in enumerate(markets.items()):
        season = mk.rows["season"].to_numpy()
        for ci, (cohort, cm) in enumerate(cohorts(mk, market)):
            info = informative(mk.y[cm])
            for period, pm in [(all_, cm)] + [(str(s), cm & (season == s)) for s in eval_seasons]:
                y = mk.y[pm]
                for name in PREDICTORS:
                    P = mk.P[name][pm]
                    for j, k in enumerate(mk.grid):
                        p, yj = P[:, j], y[:, j]
                        floor = (noise_floor(p, [SEED, mi, ci, j]) if name == TC and
                                 period == all_ and info[j] else None)
                        out.append({
                            "cohort": cohort, "period": period, "predictor": name,
                            "market": market, "threshold": k, "n": len(yj),
                            "n_pos": int(yj.sum()), "observed": float(yj.mean()) if len(yj)
                            else None, "informative": bool(info[j]),
                            "mean_pred": float(p.mean()) if len(p) else None,
                            "bias": float(p.mean() - yj.mean()) if len(p) else None,
                            "ece": ece(p, yj), "brier": float(((p - yj) ** 2).mean())
                            if len(p) else None, "log_loss": float(log_loss(p, yj).mean())
                            if len(p) else None, "noise_floor": floor})
    return pl.DataFrame(out, infer_schema_length=None).with_columns(
        pl.col("ece").fill_nan(None))


def _inf(tm: pl.DataFrame) -> pl.DataFrame:
    return tm.filter(pl.col("informative"))


def market_metrics(tm: pl.DataFrame) -> pl.DataFrame:
    """Per cohort x period x predictor x market, over informative thresholds (all thresholds
    for the reference Brier)."""
    g = ["cohort", "period", "predictor", "market"]
    allk = tm.group_by(g, maintain_order=True).agg(
        pl.len().alias("thresholds"), pl.col("n").max().alias("rows"),
        pl.col("brier").mean().alias("brier_all_thresholds"))
    inf = _inf(tm).group_by(g, maintain_order=True).agg(
        pl.len().alias("informative_thresholds"), pl.col("brier").mean().alias("brier"),
        pl.col("log_loss").mean(), pl.col("ece").mean().alias("mean_ece"),
        pl.col("ece").max().alias("worst_ece"),
        pl.col("threshold").sort_by("ece").last().alias("worst_threshold"),
        pl.col("bias").abs().mean().alias("mean_abs_bias"),
        pl.col("bias").abs().max().alias("worst_abs_bias"),
        (pl.col("ece") > CAL_CONCERN).sum().alias("over_5pp"),
        pl.col("noise_floor").mean().alias("mean_noise_floor"))
    return allk.join(inf, on=g, how="left").with_columns(
        pl.col("informative_thresholds").fill_null(0)).sort(g)


def persistent_bias(tm: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """Per cohort x predictor x market x informative threshold: seasons with bias >= +5 pp and
    <= -5 pp; persistent if either count reaches PERSIST_SEASONS."""
    s = _inf(tm).filter(pl.col("period") != f"{min(eval_seasons)}-{max(eval_seasons)}")
    return s.group_by("cohort", "predictor", "market", "threshold", maintain_order=True).agg(
        (pl.col("bias") >= PERSIST_BIAS).sum().alias("seasons_over"),
        (pl.col("bias") <= -PERSIST_BIAS).sum().alias("seasons_under"),
        pl.col("bias").abs().max().alias("worst_season_abs_bias")).with_columns(
        (pl.max_horizontal("seasons_over", "seasons_under") >= PERSIST_SEASONS)
        .alias("persistent")).sort("cohort", "predictor", "market", "threshold")


def _row_score(mk: Market, name: str, cm: np.ndarray, j: list[int], metric: str) -> np.ndarray:
    """Per-row mean over thresholds j of the Brier score or log loss of one predictor."""
    p, y = mk.P[name][cm][:, j], mk.y[cm][:, j]
    return ((p - y) ** 2 if metric == "brier" else log_loss(p, y)).mean(axis=1)


def comparisons(markets: dict[str, Market], tm: pl.DataFrame) -> pl.DataFrame:
    """Per-row mean over informative thresholds of Brier / log loss, cdf_tc minus comparator,
    game-clustered 95% bootstrap interval (Phase 4A reps and seed)."""
    all_ = tm.filter(pl.col("period").str.contains("-"))["period"][0]
    out = []
    for market, mk in markets.items():
        for cohort, cm in cohorts(mk, market):
            info = (_inf(tm).filter(pl.col("cohort") == cohort, pl.col("market") == market,
                                    pl.col("period") == all_, pl.col("predictor") == TC)
                    ["threshold"].to_list())
            j = [mk.grid.index(k) for k in info]
            for other in (EMP, INDEP):
                row = {"cohort": cohort, "market": market, "comparator": other,
                       "informative_thresholds": len(j)}
                for metric in ("brier", "log_loss"):
                    if not j or not cm.any():
                        row |= {f"{metric}_delta": None, f"{metric}_ci_low": None,
                                f"{metric}_ci_high": None}
                        continue
                    d = pl.DataFrame({"game_id": mk.rows["game_id"].filter(pl.Series(cm)),
                                      "d": _row_score(mk, TC, cm, j, metric)
                                      - _row_score(mk, other, cm, j, metric)})
                    pt, lo, hi = cluster_bootstrap_mean(d, "d", "game_id", BOOTSTRAP_REPS,
                                                        BOOTSTRAP_SEED)
                    row |= {f"{metric}_delta": pt, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi}
                out.append(row)
    return pl.DataFrame(out, infer_schema_length=None)


def interval_coverage(markets: dict[str, Market]) -> pl.DataFrame:
    """Central 50% / 80% grid intervals (a, b]: a = largest grid k with F(k) <= lo, b = smallest
    with F(k) >= hi. Predicted mass F(b) - F(a) vs observed I(a < Y <= b). Rows without such a
    and b on the grid are dropped and counted."""
    out = []
    for market, mk in markets.items():
        g = np.array(mk.grid)
        act = mk.rows["actual"].to_numpy()
        for cohort, cm in cohorts(mk, market):
            for lo, hi in INTERVALS:
                for name in (TC, INDEP):
                    F = mk.P[name][cm]
                    T = F.shape[1]
                    ok_a, ok_b = F <= lo + 1e-12, F >= hi - 1e-12
                    has = ok_a.any(1) & ok_b.any(1)
                    ia = np.where(ok_a, np.arange(T), -1).max(1)[has]
                    ib = np.where(ok_b, np.arange(T), T).min(1)[has]
                    r = np.flatnonzero(has)
                    mass = F[r, ib] - F[r, ia]
                    a = act[cm][has]
                    cov = (a > g[ia]) & (a <= g[ib])
                    out.append({"cohort": cohort, "market": market,
                                "interval": f"{round(100 * (hi - lo))}%", "predictor": name,
                                "rows_used": int(has.sum()), "rows_dropped": int((~has).sum()),
                                "predicted_mass": float(mass.mean()) if len(r) else None,
                                "observed_coverage": float(cov.mean()) if len(r) else None,
                                "coverage_gap": float(cov.mean() - mass.mean()) if len(r)
                                else None})
    return pl.DataFrame(out, infer_schema_length=None)


def zero_consistency(markets: dict[str, Market]) -> pl.DataFrame:
    """Counts: CDF(0) vs the Phase 4C zero model. Yardage: CDF(0) - CDF(-1) (= P(Y = 0)) vs the
    zero model, plus CDF(0) bias against observed P(Y <= 0) (yardage coherence)."""
    out = []
    for market, mk in markets.items():
        count = PROP_MARKETS[market].nonnegative
        act = mk.rows["actual"].to_numpy()
        z = (act == 0).astype(np.float64)
        for name in (TC, INDEP):
            F = mk.P[name]
            f0 = F[:, mk.grid.index(0)]
            imp = f0 if count else f0 - F[:, mk.grid.index(-1)]
            d = np.abs(imp - mk.p_zero)
            out.append({"market": market, "predictor": name,
                        "compared": "CDF(0)" if count else "CDF(0)-CDF(-1)",
                        "mean_abs_diff": float(d.mean()), "p95_abs_diff": float(np.quantile(
                            d, .95)), "max_abs_diff": float(d.max()),
                        "implied_p0_mean": float(imp.mean()), "observed_zero": float(z.mean()),
                        "implied_p0_bias": float(imp.mean() - z.mean()),
                        "implied_p0_brier": float(((imp - z) ** 2).mean()),
                        "zero_model_brier": float(((mk.p_zero - z) ** 2).mean()),
                        "implied_p0_ece": ece(np.clip(imp, 0, 1), z),
                        "cdf0_bias_vs_observed_le0": float(f0.mean() - (act <= 0).mean())})
    return pl.DataFrame(out)


def pmf_report(preds: pl.DataFrame, markets: dict[str, Market]) -> pl.DataFrame:
    """PMF validity of cdf_tc on every prediction row; tails against scored participant rows."""
    out = []
    for market, mk in markets.items():
        g = mk.grid
        mat = preds.filter(pl.col("market") == market).sort(
            "game_id", "player_id", "threshold")["p_le"].to_numpy().reshape(-1, len(g))
        m, s = pmf(mat), pmf(mk.P[TC])
        act = mk.rows["actual"].to_numpy()
        out.append({"market": market, "rows": len(mat), "thresholds": len(g), "k_min": g[0],
                    "k_max": g[-1], "negative_masses": int((m["interior"] < 0).sum()),
                    "min_interior_mass": float(m["interior"].min()),
                    "min_cdf": float(mat.min()), "max_cdf": float(mat.max()),
                    "max_total_mass_error": float(np.abs(m["lower"] + m["interior"].sum(1)
                                                         + m["upper"] - 1).max()),
                    "mean_lower_tail": float(s["lower"].mean()),
                    "observed_lower_tail": float((act <= g[0]).mean()),
                    "mean_upper_tail": float(s["upper"].mean()),
                    "observed_upper_tail": float((act > g[-1]).mean())})
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------- readiness

def classify(mm: pl.DataFrame, comp: pl.DataFrame, pers: pl.DataFrame, mono: pl.DataFrame,
             pmfr: pl.DataFrame, eval_seasons: tuple[int, ...], integrity_ok: bool
             ) -> pl.DataFrame:
    """Pre-registered Phase 4C.1b readiness rule (ADR 0008)."""
    all_ = f"{min(eval_seasons)}-{max(eval_seasons)}"
    rows = []

    def get(cohort, period, predictor, market):
        r = mm.filter(pl.col("cohort") == cohort, pl.col("period") == period,
                      pl.col("predictor") == predictor, pl.col("market") == market)
        return r.row(0, named=True) if r.height else {}

    def le(v, bound):
        return v is not None and not np.isnan(v) and v <= bound
    for market in PROP_MARKETS:
        mod = get("primary", all_, TC, market)
        c = comp.filter(pl.col("cohort") == "primary", pl.col("market") == market,
                        pl.col("comparator") == EMP).row(0, named=True)
        emp = get("primary", all_, EMP, market)
        seasons = [(get("primary", str(s), TC, market), get("primary", str(s), EMP, market))
                   for s in eval_seasons]
        better = sum(1 for a, b in seasons if a.get("brier") is not None
                     and b.get("brier") is not None and a["brier"] < b["brier"])
        season_ece = [a.get("mean_ece") for a, _ in seasons]
        worst_season_ece = max((v for v in season_ece if v is not None), default=None)
        n_pers = pers.filter(pl.col("cohort") == "primary", pl.col("predictor") == TC,
                             pl.col("market") == market, pl.col("persistent")).height
        mo = mono.filter(pl.col("market") == market).row(0, named=True)
        pm = pmfr.filter(pl.col("market") == market).row(0, named=True)
        has_info = (mod.get("informative_thresholds") or 0) > 0
        checks = {
            "beats_empirical": (has_info and c["brier_ci_high"] is not None
                                and mod["brier"] < emp["brier"] and c["brier_ci_high"] < 0),
            "mean_ece_ok": has_info and le(mod.get("mean_ece"), CAL_OK),
            "worst_threshold_ok": has_info and le(mod.get("worst_ece"), CAL_CONCERN),
            "monotone": mo["max_raw_decrease"] <= FP_TOL,
            "seasons_stable": (better >= MIN_SEASONS_BETTER and n_pers == 0
                               and None not in season_ece and le(worst_season_ece, CAL_CONCERN)),
            "leak_free": integrity_ok,
            "pmf_valid": pm["negative_masses"] == 0 and pm["max_total_mass_error"] <= PMF_TOL,
        }
        qb = {}
        if market == "pass_attempts":
            q = get("qb_role", all_, TC, market)
            q_pers = pers.filter(pl.col("cohort") == "qb_role", pl.col("predictor") == TC,
                                 pl.col("persistent")).height
            q_info = (q.get("informative_thresholds") or 0) > 0
            checks |= {"qb_mean_ece_ok": q_info and le(q.get("mean_ece"), CAL_OK),
                       "qb_worst_threshold_ok": q_info and le(q.get("worst_ece"), CAL_CONCERN),
                       "qb_no_persistent_bias": q_info and q_pers == 0}
            qb = {"qb_mean_ece": q.get("mean_ece"), "qb_worst_ece": q.get("worst_ece"),
                  "qb_worst_threshold": q.get("worst_threshold"),
                  "qb_persistent_thresholds": q_pers}
        if all(checks.values()):
            status = READY
        elif (integrity_ok and checks["monotone"] and checks["pmf_valid"]
              and checks["beats_empirical"] and le(mod.get("mean_ece"), CAL_CONCERN)
              and (market != "pass_attempts" or le(qb.get("qb_mean_ece"), CAL_CONCERN))):
            status = CONDITIONAL
        else:
            status = NOT_READY
        rows.append({"market": market, "status": status,
                     "informative_thresholds": mod.get("informative_thresholds"),
                     "brier_tc": mod.get("brier"), "brier_empirical": emp.get("brier"),
                     "brier_delta": c["brier_delta"], "brier_delta_ci_high": c["brier_ci_high"],
                     "log_loss_tc": mod.get("log_loss"), "log_loss_empirical": emp.get("log_loss"),
                     "mean_ece": mod.get("mean_ece"), "worst_ece": mod.get("worst_ece"),
                     "worst_threshold": mod.get("worst_threshold"),
                     "seasons_better": better, "worst_season_mean_ece": worst_season_ece,
                     "persistent_thresholds": n_pers, "max_raw_decrease": mo["max_raw_decrease"],
                     "qb_mean_ece": qb.get("qb_mean_ece"), "qb_worst_ece": qb.get("qb_worst_ece"),
                     "qb_worst_threshold": qb.get("qb_worst_threshold"),
                     "qb_persistent_thresholds": qb.get("qb_persistent_thresholds"),
                     **{k: bool(v) for k, v in checks.items()}})
    return pl.DataFrame(rows, infer_schema_length=None)


def phase5_decision(status: pl.DataFrame) -> str:
    """Frozen Phase 4C.1 mapping: GO if all READY, CONDITIONAL GO if at least one, else NO-GO."""
    return p4c1.phase5_decision(status)


# ---------------------------------------------------------------------------- run

@dataclass
class Run4C1b:
    grids: dict[str, list[int]]
    predictions: pl.DataFrame
    fold_metadata: pl.DataFrame
    threshold_metrics: pl.DataFrame
    market_metrics: pl.DataFrame
    persistent_bias: pl.DataFrame
    comparisons: pl.DataFrame
    interval_coverage: pl.DataFrame
    zero: pl.DataFrame
    pmf: pl.DataFrame
    monotonic: pl.DataFrame
    status: pl.DataFrame
    decision: str
    predictions_sha256: str


def evaluate(inputs: Inputs, p4a: p4b.Phase4A, p4cc: p4c1.Phase4C, p41: Phase4C1,
             eval_seasons: tuple[int, ...], workers: int = WORKERS) -> Run4C1b:
    p4b.check_same_inputs(inputs, p4a)
    rows = _modelling_rows(inputs.features, inputs.labels, p4a.cohorts)
    if not walk_forward_folds(rows, eval_seasons).equals(p4a.folds):
        raise CDFError("recomputed folds differ from the frozen Phase 4A folds")
    table = p4b.modelling_table(inputs, p4a)
    grids = frozen_grids(table, p4a.folds, eval_seasons, p41)
    preds, meta, mono = predict_tc(table, p4a.folds, grids, eval_seasons, workers)
    del table
    frozen = parquet_bytes(preds)
    preds = pl.read_parquet(frozen)  # score exactly the serialised predictions
    markets = build_markets(preds, phase4c1_predictions(p41),
                            scored_rows(inputs, p4a, eval_seasons), grids, p4cc)
    tm = threshold_metrics(markets, eval_seasons)
    mm = market_metrics(tm)
    pers = persistent_bias(tm, eval_seasons)
    comp = comparisons(markets, tm)
    pmfr = pmf_report(preds, markets)
    integrity_ok = meta.filter(pl.col("train_last_available_at")
                               > pl.col("decision_time")).height == 0
    status = classify(mm, comp, pers, mono, pmfr, eval_seasons, integrity_ok)
    return Run4C1b(grids, preds, meta, tm, mm, pers, comp, interval_coverage(markets),
                   zero_consistency(markets), pmfr, mono, status, phase5_decision(status),
                   hashlib.sha256(frozen).hexdigest())


FILES = ("cdf_predictions", "threshold_grid", "fold_metadata", "threshold_metrics",
         "market_metrics", "persistent_bias", "comparisons", "interval_coverage",
         "zero_consistency", "pmf", "monotonicity", "status")
RESEARCH_SOURCES = (
    ("Foresi & Peracchi (1995), JASA; Chernozhukov, Fernandez-Val & Melly (2013), "
     "Econometrica: distribution regression, binary models of 1{Y <= y} over thresholds"),
    ("Ke et al. (2017), LightGBM, NeurIPS; LightGBM docs: monotone_constraints, "
     "monotone_constraints_method 'basic' (each tree monotone in the constrained feature)"),
    ("Gneiting & Raftery (2007), JASA: Brier score and log loss are strictly proper; "
     "threshold-averaged Brier approximates the ranked probability score on the grid"),
)


def parameters(eval_seasons, grids) -> dict:
    return {
        "eval_seasons": list(eval_seasons), "warmup_seasons": list(WARMUP_SEASONS),
        "features": list(TC_FEATURES), "tc_params": TC_PARAMS, "seed": SEED,
        "grid_rule": GRID_RULE, "grids": grids, "grid_source": "frozen Phase 4C.1 run (asserted "
        "equal to the warm-up rule)",
        "training_population": "Phase 4B: kickoff + 36h <= decision_time, primary cohort, "
                               "played; each row repeated per grid threshold k; target "
                               "1{actual <= k}",
        "monotone": f"monotone constraint +1 on threshold; no repair; decreases > {FP_TOL} are "
                    "an error, smaller ones removed by a running maximum and recorded",
        "workers": "2 processes, LightGBM n_jobs=1 (results independent of the worker count)",
        "comparators": {INDEP: "Phase 4C.1 cdf_lgbm (repaired), frozen run",
                        EMP: "Phase 4C.1 empirical CDF, frozen run"},
        "scoring": {"population": "primary cohort, participant rows, not outcome-admitted",
                    "informative": f"n(Y<=k) >= {INFO_MIN}, n(Y>k) >= {INFO_MIN}, observed "
                                   f"rate in {list(INFO_RATE)}; per cohort, all seasons",
                    "ece": f"{N_BINS} equal-width bins; sum_b n_b/n |mean p_b - rate_b|",
                    "log_loss": f"p clipped to [{LOG_LOSS_EPS}, 1-{LOG_LOSS_EPS}] for scoring",
                    "noise_floor": f"diagnostic: mean ECE of y* ~ Bern(p), {NOISE_DRAWS} draws"},
        "readiness_rule": {
            "READY (all)": [
                ("mean informative Brier < empirical and game-clustered 95% CI of the "
                 "per-row difference < 0"),
                f"mean informative ECE <= {CAL_OK}", f"every informative ECE <= {CAL_CONCERN}",
                f"monotone: no raw decrease > {FP_TOL}",
                (f"stable: Brier better than empirical in >= {MIN_SEASONS_BETTER}/5 seasons, "
                 f"every season's mean informative ECE <= {CAL_CONCERN}, no informative "
                 f"threshold with |bias| >= {PERSIST_BIAS} of one sign in >= {PERSIST_SEASONS} "
                 "seasons"),
                "leak-free: no fit trained on data unavailable at its decision time",
                f"PMF valid: no negative mass, total mass error <= {PMF_TOL}"],
            "READY (pass_attempts, additionally, QB-role cohort)": [
                f"QB mean informative ECE <= {CAL_OK}",
                f"QB worst informative ECE <= {CAL_CONCERN}",
                "no persistent QB seasonal bias (same definition)"],
            "CONDITIONAL": ("leak-free, monotone, PMF valid, beats empirical, mean informative "
                            f"ECE <= {CAL_CONCERN} (pass_attempts: QB mean too)"),
            "NOT READY": "otherwise",
            "phase5": "GO if all markets READY; CONDITIONAL GO if at least one; else NO-GO"},
        "final_gate": "Phase 4C.1b is the final Phase 4C experiment; no Phase 4C.1c",
        "research_sources": list(RESEARCH_SOURCES),
    }


def write_run(run: Run4C1b, inputs: Inputs, p4a: p4b.Phase4A, p4cc: p4c1.Phase4C,
              p41: Phase4C1, eval_seasons, root: Path | None = None) -> Path:
    grid_df = pl.DataFrame([{"market": m, "threshold": k} for m, g in run.grids.items()
                            for k in g], schema={"market": pl.String, "threshold": pl.Int32})
    frames = {"cdf_predictions": run.predictions, "threshold_grid": grid_df,
              "fold_metadata": run.fold_metadata, "threshold_metrics": run.threshold_metrics,
              "market_metrics": run.market_metrics, "persistent_bias": run.persistent_bias,
              "comparisons": run.comparisons, "interval_coverage": run.interval_coverage,
              "zero_consistency": run.zero, "pmf": run.pmf, "monotonicity": run.monotonic,
              "status": run.status}
    blobs = {k: parquet_bytes(v) for k, v in frames.items()}
    shas = {k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()}
    if shas["cdf_predictions"] != run.predictions_sha256:
        raise CDFError("predictions changed after they were frozen")
    content = hashlib.sha256("".join(shas[k] for k in FILES).encode()).hexdigest()
    built_at = datetime.now(UTC)
    rel = Path(RUN_DIR) / f"{built_at:%Y%m%dT%H%M%S%fZ}_{content[:12]}"
    target = (root or p4c.evaluation_root()) / rel
    target.mkdir(parents=True, exist_ok=False)
    for k, b in blobs.items():
        with open(target / f"{k}.parquet", "xb") as f:
            f.write(b)
    manifest = {
        "artifact": "phase4c1b_threshold_conditioned_cdf", "path": rel.as_posix(),
        "built_at": built_at.isoformat(), "content_sha256": content,
        "contract_version": CONTRACT_VERSION, "feature_set_version": FEATURE_SET_VERSION,
        "baseline_set_version": BASELINE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
        "distribution_set_version": DISTRIBUTION_SET_VERSION,
        "cdf_set_version": TC_CDF_SET_VERSION, "phase4c1_cdf_set_version": CDF_SET_VERSION,
        "files": {k: {"sha256": shas[k], "rows": frames[k].height} for k in FILES},
        "phase4a": {"path": str(p4a.path), "content_sha256": p4a.manifest["content_sha256"],
                    "folds_sha256": p4a.manifest["files"]["folds"]["sha256"]},
        "phase4c": {"path": str(p4cc.path), "content_sha256": p4cc.manifest["content_sha256"]},
        "phase4c1": {"path": str(p41.path), "content_sha256": p41.manifest["content_sha256"],
                     "cdf_predictions_sha256": p41.manifest["files"]["cdf_predictions"]["sha256"],
                     "official_status": p41.manifest["summary"]["status"],
                     "official_phase5": p41.manifest["summary"]["phase5"]},
        "data_roots": {k: os.environ.get(v, d) for k, v, d in (
            ("raw", "LINESEEK_DATA_ROOT", "data/raw"),
            ("labels", "LINESEEK_LABELS_ROOT", "data/labels"),
            ("features", "LINESEEK_FEATURES_ROOT", "data/features"),
            ("evaluation", "LINESEEK_EVALUATION_ROOT", "data/evaluation"))},
        "inputs": inputs.provenance,
        "parameters": parameters(eval_seasons, run.grids),
        "runtime": runtime_versions() | {p: p4c._v(p) for p in ("scikit-learn", "lightgbm")},
        "warnings": [POINT_IN_TIME_WARNING, UNIVERSE_WARNING, F3_NOTE, p4c.RECONSTRUCTED_WARNING],
        "summary": summarize(run),
    }
    with open(target / "manifest.json", "x") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    return target


def summarize(run: Run4C1b) -> dict:
    return {
        "folds": run.fold_metadata["fold_id"].n_unique(),
        "fits": run.fold_metadata.height,
        "thresholds": {m: len(g) for m, g in run.grids.items()},
        "prediction_rows": run.predictions.height,
        "temporal_violations": run.fold_metadata.filter(
            pl.col("train_last_available_at") > pl.col("decision_time")).height,
        "max_raw_decrease": dict(run.monotonic.select("market", "max_raw_decrease").iter_rows()),
        "status": dict(run.status.select("market", "status").iter_rows()),
        "phase5": run.decision,
    }


def validate_run(path: Path) -> dict:
    m = json.loads((path / "manifest.json").read_text())
    for k, meta in m["files"].items():
        blob = (path / f"{k}.parquet").read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            raise CDFError(f"{k}.parquet: checksum does not match the manifest")
    p = validate_frame(pl.read_parquet(path / "cdf_predictions.parquet"), "cdf_predictions")
    if set(p["predictor"]) != {TC}:
        raise CDFError("a Phase 4C.1b run holds only cdf_tc predictions")
    if m["summary"]["temporal_violations"]:
        raise CDFError("temporal violations recorded")
    if POINT_IN_TIME_WARNING not in m["warnings"]:
        raise CDFError("manifest lacks the point-in-time warning")
    return m
