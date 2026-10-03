"""Phase 4C.1 run: discrete CDF P(Y <= k) on the frozen protocol (docs/adr/0007).

Reuses the validated Phase 4A run (folds, cohorts), the audited Phase 4B training population
and features, and the Phase 4C run (zero model, quantiles) for consistency diagnostics. The
threshold grid is fixed from warm-up outcomes before any fold. Predictions (CDF LightGBM,
empirical CDF) are serialised and checksummed before labels are joined. Scoring: primary
cohort, participant rows (conditional on playing).
"""

import hashlib
import json
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
from lineseek.evaluation.phase4a import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    F3_NOTE,
    POINT_IN_TIME_WARNING,
    UNIVERSE_WARNING,
    WARMUP_SEASONS,
    Inputs,
    _modelling_rows,
    histories_for,
    parquet_bytes,
    runtime_versions,
)
from lineseek.evaluation.scoring import LOG_LOSS_EPS, cluster_bootstrap_mean
from lineseek.evaluation.universe import qb_role
from lineseek.evaluation.walkforward import walk_forward_folds
from lineseek.features.player_game import FEATURE_SET_VERSION
from lineseek.models.baselines import BASELINE_SET_VERSION
from lineseek.models.cdf import (
    CDF_PARAMS,
    CDF_SET_VERSION,
    GRID_RULE,
    CDFError,
    empirical_cdf,
    fit_predict_le,
    monotone_repair,
    pmf,
    quantile_approx_cdf,
    threshold_grid,
)
from lineseek.models.distribution import DISTRIBUTION_SET_VERSION
from lineseek.models.learned import FEATURES, MODEL_SET_VERSION, SEED

RUN_DIR = "phase4c1"
KEY = ["game_id", "player_id", "market"]
ROW = [*KEY, "decision_time"]
PREDICTORS = ("cdf_lgbm", "empirical", "quantile_approx")
CHUNK = 4  # thresholds per worker task
N_BINS = 20  # reliability bins of width 0.05, fixed before any result
# Readiness rule and tolerances, frozen before any Phase 4C.1 result (ADR 0007).
CAL_STRONG, CAL_OK, CAL_CONCERN = 0.02, 0.03, 0.05
IMPORTANT_OBS = (0.05, 0.95)   # a threshold is "important" if its observed rate is inside
LOW_SAMPLE_NEFF = 200          # n * min(rate, 1 - rate) below this: low-sample exception
MONO_SEVERE = 0.25             # raw monotonicity-violating rows above this share: severe
ZERO_TOL = 0.01                # count markets: mean |P0 - CDF(0)| tolerance
MIN_SEASONS_BETTER = 4         # Brier better than empirical in >= 4 of 5 seasons
READY, CONDITIONAL, NOT_READY = "READY", "CONDITIONAL", "NOT READY"


def _check(msg: str, df: pl.DataFrame) -> None:
    if df.height:
        raise CDFError(f"{df.height} {msg}: {df.head(5).rows()}")


# ---------------------------------------------------------------------------- inputs

@dataclass
class Phase4C:
    path: Path
    manifest: dict
    dist: pl.DataFrame  # quantile_lgbm rows: q10..q90, p_zero


def load_phase4c(path: Path, p4a: p4b.Phase4A) -> Phase4C:
    m = p4c.validate_run(path)
    if m["phase4a"]["content_sha256"] != p4a.manifest["content_sha256"]:
        raise CDFError("Phase 4C run was not built on this Phase 4A run")
    d = pl.read_parquet(path / "distribution_predictions.parquet").filter(
        pl.col("predictor") == "quantile_lgbm").select(*KEY, "q10", "q25", "q50", "q75", "q90",
                                                       "p_zero")
    return Phase4C(path, m, d)


def grids_from_warmup(table: pl.DataFrame, folds: pl.DataFrame, eval_seasons: tuple[int, ...]
                      ) -> dict[str, list[int]]:
    """Grid from warm-up outcomes: every season before the first evaluation season (2018-2020
    for the frozen protocol), fit-eligible rows only, all available before the first fold."""
    w = table.filter(pl.col("season") < min(eval_seasons), pl.col("y").is_not_null())
    first = folds["decision_time"].min()
    if w["outcome_available_at"].max() > first:
        raise CDFError("warm-up outcomes not all available before the first fold")
    return {m: threshold_grid(m, w.filter(pl.col("market") == m)["y"].to_numpy())
            for m in PROP_MARKETS}


# ---------------------------------------------------------------------------- training

@dataclass(frozen=True)
class CDFTask:
    market: str
    thresholds: tuple[int, ...]
    x: np.ndarray
    y: np.ndarray
    available: np.ndarray
    decision: np.ndarray
    folds: tuple[tuple[int, int], ...]


def run_cdf(task: CDFTask) -> dict:
    pred = np.full((len(task.x), len(task.thresholds)), np.nan)
    train_n = np.zeros(len(task.x), dtype=np.int64)
    meta = []
    ok = ~np.isnan(task.y)
    for fold_id, dt in task.folds:
        test = task.decision == dt
        if not test.any():
            continue
        train = ok & (task.available <= dt)
        if not train.any() or (train & test).any() or task.available[train].max() > dt:
            raise CDFError(f"fold {fold_id}: training rows not available by the lock")
        pred[test] = fit_predict_le(list(task.thresholds), task.x[train], task.y[train],
                                    task.x[test])
        train_n[test] = train.sum()
        meta += [{"fold_id": fold_id, "market": task.market, "threshold": k,
                  "train_rows": int(train.sum()),
                  "train_rate": float((task.y[train] <= k).mean()),
                  "train_last_available_us": int(task.available[train].max())}
                 for k in task.thresholds]
    return {"market": task.market, "thresholds": task.thresholds, "pred": pred,
            "train_rows": train_n, "meta": meta}


META = ["game_id", "player_id", "season", "week", "team", "opponent", "market", "kickoff_utc",
        "decision_time", "fold_id"]


def _long(m: pl.DataFrame, grid: list[int], raw: np.ndarray, rep: np.ndarray,
          changed: np.ndarray) -> pl.DataFrame:
    n, t = raw.shape
    return pl.DataFrame({
        "row": np.repeat(np.arange(n), t), "threshold": np.tile(np.array(grid, dtype=np.int32), n),
        "p_le_raw": raw.reshape(-1), "p_le": rep.reshape(-1),
        "monotonic_repair": changed.reshape(-1)}).join(
        m.with_row_index("row").with_columns(pl.col("row").cast(pl.Int64)), on="row").drop("row")


def predict_cdf(table: pl.DataFrame, folds: pl.DataFrame, inputs: Inputs,
                grids: dict[str, list[int]], eval_seasons: tuple[int, ...],
                workers: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """CDF LightGBM + empirical CDF for every evaluation row and grid threshold, plus the
    monotonicity record. No label is read except fit-eligible training targets inside folds."""
    base = {t.market: t for t in p4b.build_tasks(table, folds, ("lightgbm",))}
    if any(t.x.shape[1] != len(FEATURES) for t in base.values()):
        raise CDFError("model matrices must be exactly the frozen feature list")
    tasks = [CDFTask(mk, tuple(g[i:i + CHUNK]), t.x, t.y, t.available, t.decision, t.folds)
             for mk, t in base.items() for g in [grids[mk]] for i in range(0, len(g), CHUNK)]
    workers = min(len(tasks), os.cpu_count() or 1) if workers is None else workers
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(run_cdf, tasks))
    else:
        results = [run_cdf(t) for t in tasks]
    common = {"feature_set_version": FEATURE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
              "distribution_set_version": DISTRIBUTION_SET_VERSION,
              "cdf_set_version": CDF_SET_VERSION}
    parts, mono = [], []
    for market in PROP_MARKETS:
        rs = [r for r in results if r["market"] == market]
        m = table.filter(pl.col("market") == market).select(*META, "prior_games_n")
        keep = m["season"].is_in(list(eval_seasons)).to_numpy()
        raw = np.column_stack([r["pred"] for r in rs])[keep]
        rep = monotone_repair(raw, anchor_first=PROP_MARKETS[market].nonnegative)
        if PROP_MARKETS[market].nonnegative and grids[market][0] != 0:
            raise CDFError(f"{market}: count-market grid must start at 0 for the zero anchor")
        mm = m.filter(pl.Series(keep)).with_columns(
            pl.col("prior_games_n").alias("history_n"),
            pl.Series("train_rows", rs[0]["train_rows"][keep]).cast(pl.Int32)).drop(
            "prior_games_n")
        mono.append(pl.DataFrame({"market": market, "violations": rep["violations"],
                                  "repair_max": rep["repair_max"]}))
        parts.append(_long(mm, grids[market], raw, rep["repaired"], rep["changed"])
                     .with_columns(pl.lit("cdf_lgbm").alias("predictor")))
    model = pl.concat(parts).with_columns(*[pl.lit(v).alias(k) for k, v in common.items()])
    # Empirical CDF: eligible history, else the fold's market-level training share <= k.
    feats = pl.concat([inputs.features[s] for s in eval_seasons])
    hist = histories_for(inputs.feature_raw)
    emp = pl.concat([empirical_cdf(feats.filter(pl.col("season") == s), hist[s], grids)
                     for s in eval_seasons])
    fb = []
    for mk, t in base.items():
        ok = ~np.isnan(t.y)
        for fid, dt in t.folds:
            yy = t.y[ok & (t.available <= dt)]
            fb += [{"market": mk, "fold_id": fid, "threshold": k, "fb": float((yy <= k).mean())}
                   for k in grids[mk]]
    fbd = pl.DataFrame(fb, schema={"market": pl.String, "fold_id": pl.Int32,
                                   "threshold": pl.Int32, "fb": pl.Float64})
    e = (model.select(*META, "threshold", "train_rows", pl.col("history_n").alias("_n"))
         .join(emp.rename({"history_n": "_hn"}), on=[*KEY[:2], "market", "threshold"], how="left")
         .join(fbd, on=["market", "fold_id", "threshold"], how="left"))
    _check("empirical history length differs from prior_games_n",
           e.filter(pl.col("_hn").fill_null(0) != pl.col("_n")))
    e = e.with_columns(pl.coalesce("p_le", "fb").alias("p_le"), pl.col("_n").alias("history_n"),
                       pl.lit("empirical").alias("predictor")).with_columns(
        pl.col("p_le").alias("p_le_raw"), pl.lit(False).alias("monotonic_repair"),
        *[pl.lit(v).alias(k) for k, v in common.items()])
    cols = list(ARTIFACT_SCHEMAS["cdf_predictions"])
    out = pl.concat([model.select(cols), e.select(cols)]).sort(
        "season", "week", "game_id", "team", "player_id", "market", "predictor", "threshold")
    validate_frame(out, "cdf_predictions")
    meta = pl.DataFrame([x for r in results for x in r["meta"]]).with_columns(
        pl.from_epoch("train_last_available_us", "us").dt.replace_time_zone("UTC")
        .alias("train_last_available_at")).drop("train_last_available_us").join(
        folds.select("fold_id", "decision_time"), on="fold_id").sort("fold_id", "market",
                                                                      "threshold")
    _check("fits trained on data unavailable at the decision time",
           meta.filter(pl.col("train_last_available_at") > pl.col("decision_time")))
    return out, meta, pl.concat(mono)


# ---------------------------------------------------------------------------- scoring

def scored_frame(preds: pl.DataFrame, inputs: Inputs, p4a: p4b.Phase4A, p4cc: Phase4C,
                 grids: dict[str, list[int]], eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """Primary-cohort participant rows x thresholds for the CDF, empirical and Phase 4C
    quantile-approximation predictors, with the outcome indicator I(actual <= k)."""
    lab = pl.concat([inputs.labels[s] for s in eval_seasons]).select(*KEY, "actual", "played")
    qb = pl.concat([qb_role(inputs.labels[s], inputs.label_raw[s]["schedules"],
                            inputs.label_raw[s]["rosters_weekly"]) for s in eval_seasons])
    rows = (lab.filter(pl.col("played")).join(
        p4a.cohorts.select("game_id", "player_id", "outcome_admitted"),
        on=["game_id", "player_id"]).filter(~pl.col("outcome_admitted"))
        .join(qb, on=["game_id", "player_id"]))
    _check("participant rows without an actual", rows.filter(pl.col("actual").is_null()))
    p = preds.select(*KEY, "season", "predictor", "threshold", "p_le", "p_le_raw").join(
        rows.select(*KEY, "actual", "qb_role"), on=KEY)
    qa = []
    d = p4cc.dist.join(rows.select(KEY), on=KEY)
    seasons = preds.select(*KEY, "season").unique()
    d = d.join(seasons, on=KEY)
    for market, grid in grids.items():
        x = d.filter(pl.col("market") == market)
        qm = x.select("q10", "q25", "q50", "q75", "q90").to_numpy()
        for k in grid:
            qa.append(x.select(*KEY, "season").with_columns(
                pl.lit("quantile_approx").alias("predictor"), pl.lit(k, pl.Int32).alias("threshold"),
                pl.Series("p_le", quantile_approx_cdf(qm, k))).with_columns(
                pl.col("p_le").alias("p_le_raw")))
    qa = pl.concat(qa).join(rows.select(*KEY, "actual", "qb_role"), on=KEY)
    out = pl.concat([p, qa.select(p.columns)]).with_columns(
        (pl.col("actual") <= pl.col("threshold")).cast(pl.Float64).alias("y_le")).sort(
        *KEY, "predictor", "threshold")  # fixed row order -> bit-stable float aggregates
    return out.with_columns(
        ((pl.col("p_le") - pl.col("y_le")) ** 2).alias("brier"),
        ((pl.col("p_le_raw") - pl.col("y_le")) ** 2).alias("brier_raw"),
        _logloss("p_le").alias("log_loss"),
        (pl.col("p_le") * N_BINS).floor().clip(0, N_BINS - 1).cast(pl.Int8).alias("bin"))


def _logloss(col: str) -> pl.Expr:
    p = pl.col(col).clip(LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)  # scoring only; stored p unchanged
    y = pl.col("y_le")
    return -(y * p.log() + (1 - y) * (1 - p).log())


def cohorts(s: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    return [("primary", s), ("qb_role", s.filter(pl.col("market") == "pass_attempts",
                                                pl.col("qb_role")))]


def threshold_metrics(s: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """Per cohort x period x predictor x market x threshold: Brier, log loss, observed rate,
    mean prediction, calibration-in-the-large and binned ECE (20 bins)."""
    all_ = f"{min(eval_seasons)}-{max(eval_seasons)}"
    keys = ["predictor", "market", "threshold"]
    out = []
    for cohort, c in cohorts(s):
        for period, cond in [(all_, pl.lit(True))] + [(str(y), pl.col("season") == y)
                                                      for y in eval_seasons]:
            d = c.filter(cond)
            base = d.group_by(keys, maintain_order=True).agg(
                pl.len().alias("n"), pl.col("brier").mean(), pl.col("brier_raw").mean(),
                pl.col("log_loss").mean(), pl.col("p_le").mean().alias("mean_pred"),
                pl.col("y_le").mean().alias("observed"))
            ece = d.group_by(*keys, "bin", maintain_order=True).agg(
                pl.len().alias("nb"), pl.col("p_le").mean().alias("pb"),
                pl.col("y_le").mean().alias("yb")).group_by(keys, maintain_order=True).agg(
                ((pl.col("pb") - pl.col("yb")).abs() * pl.col("nb")).sum().alias("_e"),
                pl.col("nb").sum().alias("_n"))
            out.append(base.join(ece, on=keys).with_columns(
                (pl.col("_e") / pl.col("_n")).alias("ece"),
                (pl.col("mean_pred") - pl.col("observed")).abs().alias("citl"),
                pl.lit(cohort).alias("cohort"), pl.lit(period).alias("period")).drop("_e", "_n"))
    t = pl.concat(out)
    lo, hi = IMPORTANT_OBS
    return t.with_columns(
        pl.col("observed").is_between(lo, hi).alias("important"),
        (pl.col("n") * pl.min_horizontal("observed", 1 - pl.col("observed"))).alias("n_eff"),
    ).select("cohort", "period", *keys, pl.exclude("cohort", "period", *keys)).sort(
        "cohort", "period", *keys)


def market_metrics(s: pl.DataFrame, tm: pl.DataFrame) -> pl.DataFrame:
    """Aggregate over thresholds: mean Brier/log loss (equal weight per threshold), mean and
    worst ECE, worst important-threshold ECE, pooled ECE (all row-threshold pairs)."""
    g = ["cohort", "period", "predictor", "market"]
    agg = tm.group_by(g, maintain_order=True).agg(
        pl.len().alias("thresholds"), pl.col("n").max().alias("rows"),
        pl.col("brier").mean(), pl.col("brier_raw").mean(), pl.col("log_loss").mean(),
        pl.col("ece").mean().alias("mean_ece"), pl.col("ece").max().alias("worst_ece"),
        pl.col("threshold").sort_by("ece").last().alias("worst_threshold"),
        pl.col("citl").mean().alias("mean_citl"),
        pl.col("ece").filter(pl.col("important")).max().alias("worst_important_ece"),
        pl.col("threshold").filter(pl.col("important")).sort_by(
            pl.col("ece").filter(pl.col("important"))).last().alias("worst_important_threshold"),
        (pl.col("important") & (pl.col("ece") > CAL_CONCERN)
         & (pl.col("n_eff") >= LOW_SAMPLE_NEFF)).sum().alias("important_over_5pp"))
    pooled = []
    all_ = tm.filter(pl.col("period").str.contains("-"))["period"][0]
    for cohort, c in cohorts(s):
        for period, cond in [(all_, pl.lit(True))] + [
                (p, pl.col("season") == int(p)) for p in sorted(set(tm["period"])) if p != all_]:
            d = c.filter(cond).group_by("predictor", "market", "bin", maintain_order=True).agg(
                pl.len().alias("nb"), pl.col("p_le").mean().alias("pb"),
                pl.col("y_le").mean().alias("yb")).group_by("predictor", "market",
                                                            maintain_order=True).agg(
                (((pl.col("pb") - pl.col("yb")).abs() * pl.col("nb")).sum()
                 / pl.col("nb").sum()).alias("pooled_ece"))
            pooled.append(d.with_columns(pl.lit(cohort).alias("cohort"),
                                         pl.lit(period).alias("period")))
    return agg.join(pl.concat(pooled), on=g, how="left").sort(g)


def reliability(s: pl.DataFrame) -> pl.DataFrame:
    """Pooled reliability table per predictor x market x bin (primary, all seasons)."""
    return s.group_by("predictor", "market", "bin", maintain_order=True).agg(
        pl.len().alias("n"), pl.col("p_le").mean().alias("mean_pred"),
        pl.col("y_le").mean().alias("observed")).with_columns(
        (pl.col("observed") - pl.col("mean_pred")).alias("gap")).sort("predictor", "market",
                                                                      "bin")


def zero_consistency(preds: pl.DataFrame, s: pl.DataFrame, p4cc: Phase4C) -> pl.DataFrame:
    """Counts: CDF(0) vs Phase 4C P(Y = 0) (identical target). Yardage: CDF(0) - CDF(-1)
    (= P(Y = 0)) vs Phase 4C P(Y = 0); CDF(0) itself also includes negative outcomes."""
    c = preds.filter(pl.col("predictor") == "cdf_lgbm", pl.col("threshold").is_in([-1, 0])).pivot(
        "threshold", index=KEY, values="p_le").rename({"0": "cdf0", "-1": "cdf_m1"}, strict=False)
    if "cdf_m1" not in c.columns:
        c = c.with_columns(pl.lit(None, pl.Float64).alias("cdf_m1"))
    c = c.with_columns(pl.when(pl.col("market").is_in(
        [k for k, m in PROP_MARKETS.items() if m.nonnegative])).then(pl.col("cdf0"))
        .otherwise(pl.col("cdf0") - pl.col("cdf_m1")).alias("cdf_p0"))
    a = s.filter(pl.col("predictor") == "cdf_lgbm", pl.col("threshold") == 0).select(
        *KEY, "actual")
    j = c.join(p4cc.dist.select(*KEY, "p_zero"), on=KEY).join(a, on=KEY).sort(KEY)
    z = (pl.col("actual") == 0).cast(pl.Float64)
    return j.group_by("market", maintain_order=True).agg(
        pl.len().alias("n"),
        (pl.col("p_zero") - pl.col("cdf_p0")).abs().mean().alias("mean_abs_diff"),
        (pl.col("p_zero") - pl.col("cdf_p0")).abs().max().alias("max_abs_diff"),
        ((pl.col("p_zero") - z) ** 2).mean().alias("p0_brier"),
        ((pl.col("cdf_p0") - z) ** 2).mean().alias("cdf_p0_brier"),
        pl.col("p_zero").mean().alias("p0_mean"), pl.col("cdf_p0").mean().alias("cdf_p0_mean"),
        z.mean().alias("observed_zero"),
        ((pl.col("cdf0") - (pl.col("actual") <= 0).cast(pl.Float64)) ** 2).mean()
        .alias("cdf0_brier"), (pl.col("actual") <= 0).cast(pl.Float64).mean()
        .alias("observed_le0")).sort("market")


def pmf_report(preds: pl.DataFrame, s: pl.DataFrame, grids: dict[str, list[int]]) -> pl.DataFrame:
    """PMF validity on the repaired CDF: nonnegative interval masses, explicit tails and their
    observed frequencies (participant rows)."""
    rows = []
    act = s.filter(pl.col("predictor") == "cdf_lgbm").select(*KEY, "actual").unique(
        maintain_order=True)
    for market, grid in grids.items():
        p = preds.filter(pl.col("predictor") == "cdf_lgbm", pl.col("market") == market).sort(
            *ROW, "threshold")
        mat = p["p_le"].to_numpy().reshape(-1, len(grid))
        m = pmf(mat)
        keys = p.filter(pl.col("threshold") == grid[0]).select(KEY)
        lower = keys.with_columns(pl.Series("lower", m["lower"]), pl.Series("upper", m["upper"]))
        j = lower.join(act, on=KEY).sort(KEY)
        rows.append({"market": market, "thresholds": len(grid), "k_min": grid[0],
                     "k_max": grid[-1], "min_interior_mass": float(m["interior"].min()),
                     "negative_masses": int((m["interior"] < 0).sum()),
                     "max_total_mass_error": float(np.abs(m["lower"] + m["interior"].sum(1)
                                                          + m["upper"] - 1).max()),
                     "mean_lower_tail": float(j["lower"].mean()),
                     "observed_lower_tail": float((j["actual"] <= grid[0]).mean()),
                     "mean_upper_tail": float(j["upper"].mean()),
                     "observed_upper_tail": float((j["actual"] > grid[-1]).mean())})
    return pl.DataFrame(rows)


def quantile_consistency(preds: pl.DataFrame, p4cc: Phase4C, grids: dict[str, list[int]]
                         ) -> pl.DataFrame:
    """Implied quantile from the discrete CDF (smallest grid k with CDF >= tau) vs Phase 4C."""
    rows = []
    for market, grid in grids.items():
        p = preds.filter(pl.col("predictor") == "cdf_lgbm", pl.col("market") == market).sort(
            *ROW, "threshold")
        mat = p["p_le"].to_numpy().reshape(-1, len(grid))
        keys = p.filter(pl.col("threshold") == grid[0]).select(KEY)
        q = keys.join(p4cc.dist, on=KEY, how="left", maintain_order="left")  # positional
        for qc, tau in (("q10", .1), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9)):
            idx = (mat < tau).sum(axis=1)
            implied = np.where(idx < len(grid), np.array(grid)[np.minimum(idx, len(grid) - 1)],
                               np.nan)
            old = q[qc].to_numpy()
            ok = ~np.isnan(implied)
            rows.append({"market": market, "quantile": qc, "rows": len(old),
                         "above_grid": int((~ok).sum()),
                         "mean_abs_diff": float(np.abs(implied[ok] - old[ok]).mean()),
                         "share_implied_equals_ceil": float((implied[ok] == np.ceil(old[ok]))
                                                            .mean()),
                         "share_implied_equals_floor": float((implied[ok] == np.floor(old[ok]))
                                                             .mean())})
    return pl.DataFrame(rows)


def monotonic_report(mono: pl.DataFrame, mm: pl.DataFrame, all_: str) -> pl.DataFrame:
    b = mm.filter(pl.col("cohort") == "primary", pl.col("period") == all_,
                  pl.col("predictor") == "cdf_lgbm").select("market", "brier_raw", "brier")
    return mono.group_by("market", maintain_order=True).agg(
        pl.len().alias("rows"), (pl.col("violations") > 0).sum().alias("violating_rows"),
        (pl.col("violations") > 0).mean().alias("violating_share"),
        pl.col("violations").sum().alias("threshold_violations"),
        pl.col("repair_max").filter(pl.col("violations") > 0).mean().alias("mean_repair"),
        pl.col("repair_max").max().alias("max_repair")).join(b, on="market").sort("market")


def classify(s: pl.DataFrame, mm: pl.DataFrame, mono: pl.DataFrame, zero: pl.DataFrame,
             eval_seasons: tuple[int, ...], integrity_ok: bool) -> pl.DataFrame:
    """Frozen Phase 5 readiness rule per market (primary cohort, participant rows)."""
    all_ = f"{min(eval_seasons)}-{max(eval_seasons)}"
    prim = mm.filter(pl.col("cohort") == "primary")
    rows = []
    for market in PROP_MARKETS:
        g = prim.filter(pl.col("market") == market)
        mod = g.filter(pl.col("period") == all_, pl.col("predictor") == "cdf_lgbm").row(
            0, named=True)
        emp = g.filter(pl.col("period") == all_, pl.col("predictor") == "empirical").row(
            0, named=True)
        d = s.filter(pl.col("market") == market, pl.col("predictor").is_in(
            ["cdf_lgbm", "empirical"])).group_by(*KEY, "predictor", maintain_order=True).agg(
            pl.col("brier").mean()).pivot("predictor", index=KEY, values="brier").with_columns(
            (pl.col("cdf_lgbm") - pl.col("empirical")).alias("d")).sort(KEY)
        point, lo, hi = cluster_bootstrap_mean(d, "d", "game_id", BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        seas = g.filter(pl.col("period") != all_).pivot("predictor", index="period",
                                                        values=["brier", "mean_ece"])
        seasons_better = int((seas["brier_cdf_lgbm"] < seas["brier_empirical"]).sum())
        season_ece = float(seas["mean_ece_cdf_lgbm"].max())
        mono_share = float(mono.filter(pl.col("market") == market)["violating_share"].item())
        zr = zero.filter(pl.col("market") == market).row(0, named=True)
        zero_ok = (zr["mean_abs_diff"] <= ZERO_TOL if PROP_MARKETS[market].nonnegative
                   else abs(zr["cdf_p0_mean"] - zr["observed_zero"]) <= CAL_OK)
        checks = {
            "beats_empirical": mod["brier"] < emp["brier"] and hi < 0,
            "mean_ece_ok": mod["mean_ece"] <= CAL_OK,
            "important_thresholds_ok": mod["important_over_5pp"] == 0,
            "monotone_clean": mono_share <= MONO_SEVERE,
            "zero_consistent": zero_ok,
            "seasons_stable": seasons_better >= MIN_SEASONS_BETTER and season_ece <= CAL_CONCERN,
            "integrity_ok": integrity_ok,
        }
        if all(checks.values()):
            status = READY
        elif (integrity_ok and checks["beats_empirical"] and mod["mean_ece"] <= CAL_CONCERN
              and mono_share <= MONO_SEVERE):
            status = CONDITIONAL
        else:
            status = NOT_READY
        band = ("strong" if mod["mean_ece"] <= CAL_STRONG else "acceptable"
                if mod["mean_ece"] <= CAL_OK else "concerning" if mod["mean_ece"] <= CAL_CONCERN
                else "not ready")
        rows.append({"market": market, "status": status, "calibration_band": band,
                     "brier_cdf": mod["brier"], "brier_empirical": emp["brier"],
                     "brier_delta": point, "brier_delta_ci_low": lo, "brier_delta_ci_high": hi,
                     "log_loss_cdf": mod["log_loss"], "log_loss_empirical": emp["log_loss"],
                     "mean_ece": mod["mean_ece"], "worst_ece": mod["worst_ece"],
                     "worst_threshold": mod["worst_threshold"],
                     "worst_important_ece": mod["worst_important_ece"],
                     "important_over_5pp": mod["important_over_5pp"],
                     "seasons_better": seasons_better, "worst_season_mean_ece": season_ece,
                     "monotone_violating_share": mono_share, **checks})
    return pl.DataFrame(rows)


def phase5_decision(status: pl.DataFrame) -> str:
    ready = int((status["status"] == READY).sum())
    return "GO" if ready == len(status) else "CONDITIONAL GO" if ready else "NO-GO"


# ---------------------------------------------------------------------------- run

@dataclass
class Run4C1:
    grids: dict[str, list[int]]
    predictions: pl.DataFrame
    fold_metadata: pl.DataFrame
    scored: pl.DataFrame
    threshold_metrics: pl.DataFrame
    market_metrics: pl.DataFrame
    reliability: pl.DataFrame
    zero: pl.DataFrame
    pmf: pl.DataFrame
    quantiles: pl.DataFrame
    monotonic: pl.DataFrame
    status: pl.DataFrame
    decision: str
    predictions_sha256: str


def evaluate(inputs: Inputs, p4a: p4b.Phase4A, p4cc: Phase4C, eval_seasons: tuple[int, ...],
             workers: int | None = None) -> Run4C1:
    p4b.check_same_inputs(inputs, p4a)
    rows = _modelling_rows(inputs.features, inputs.labels, p4a.cohorts)
    if not walk_forward_folds(rows, eval_seasons).equals(p4a.folds):
        raise CDFError("recomputed folds differ from the frozen Phase 4A folds")
    table = p4b.modelling_table(inputs, p4a)
    grids = grids_from_warmup(table, p4a.folds, eval_seasons)
    preds, meta, mono = predict_cdf(table, p4a.folds, inputs, grids, eval_seasons, workers)
    frozen = parquet_bytes(preds)
    preds = pl.read_parquet(frozen)  # score exactly the serialised predictions
    s = scored_frame(preds, inputs, p4a, p4cc, grids, eval_seasons)
    tm = threshold_metrics(s, eval_seasons)
    mm = market_metrics(s, tm)
    all_ = f"{min(eval_seasons)}-{max(eval_seasons)}"
    monor = monotonic_report(mono, mm, all_)
    zero = zero_consistency(preds, s, p4cc)
    integrity_ok = meta.filter(pl.col("train_last_available_at")
                               > pl.col("decision_time")).height == 0
    status = classify(s, mm, monor, zero, eval_seasons, integrity_ok)
    return Run4C1(grids, preds, meta, s, tm, mm,
                  reliability(s), zero,
                  pmf_report(preds, s, grids), quantile_consistency(preds, p4cc, grids), monor,
                  status, phase5_decision(status), hashlib.sha256(frozen).hexdigest())


FILES = ("cdf_predictions", "threshold_grid", "fold_metadata", "cdf_metrics",
         "cdf_calibration", "cdf_reliability", "zero_consistency", "cdf_pmf",
         "quantile_consistency", "monotonicity", "status")
RESEARCH_SOURCES = (
    ("Foresi & Peracchi (1995), The conditional distribution of excess returns, JASA; "
     "Chernozhukov, Fernandez-Val & Melly (2013), Inference on counterfactual distributions, "
     "Econometrica: distribution regression, one binary model of 1{Y <= y} per threshold"),
    ("Chernozhukov, Fernandez-Val & Galichon (2010), Econometrica: monotone rearrangement of "
     "estimated distribution functions (sorting) cannot increase estimation error"),
    ("Gneiting & Raftery (2007), JASA: the Brier score and log loss are strictly proper; "
     "Brier score of 1{Y <= k} averaged over thresholds approximates the ranked probability "
     "score (Epstein 1969) on the grid"),
    "LightGBM Parameters documentation: objective 'binary'; deterministic settings",
)


def parameters(eval_seasons, grids) -> dict:
    return {
        "eval_seasons": list(eval_seasons), "warmup_seasons": list(WARMUP_SEASONS),
        "features": list(FEATURES), "cdf_params": CDF_PARAMS, "seed": SEED,
        "grid_rule": GRID_RULE, "grids": grids,
        "training_population": "Phase 4B: kickoff + 36h <= decision_time, primary cohort, "
                               "played; target 1{actual <= k}",
        "monotone_repair": "record raw violations; row-wise sort across ascending thresholds; "
                           "count markets anchored: CDF(0) kept (= zero model), the rest "
                           "sorted and floored at CDF(0)",
        "empirical_baseline": "share of the player's eligible history <= k; no history: the "
                              "fold's market-level training share <= k",
        "quantile_approx": "Phase 4C quantiles -> F(k): 0.05 below Q10, 0.95 above Q90, "
                           "largest tied level at a knot, linear between knots (diagnostic)",
        "scoring": {"brier": "(p - 1{y<=k})^2, mean over rows then equal weight per threshold",
                    "log_loss": f"p clipped to [{LOG_LOSS_EPS}, 1-{LOG_LOSS_EPS}] for scoring",
                    "ece": f"{N_BINS} equal-width bins; sum_b n_b/n |mean p_b - rate_b|",
                    "population": "primary cohort, participant rows (DNP never a zero)"},
        "readiness_rule": {
            "READY": ["Brier < empirical with game-clustered 95% CI < 0",
                      f"mean ECE <= {CAL_OK}",
                      (f"no important threshold (observed rate in {IMPORTANT_OBS}) with ECE > "
                       f"{CAL_CONCERN} unless n_eff < {LOW_SAMPLE_NEFF}"),
                      f"raw monotonicity-violating rows <= {MONO_SEVERE}",
                      (f"zero consistency: counts mean |P0 - CDF(0)| <= {ZERO_TOL}; yardage "
                       f"|mean CDF-implied P0 - observed| <= {CAL_OK}"),
                      (f"Brier better in >= {MIN_SEASONS_BETTER}/5 seasons and every season "
                       f"mean ECE <= {CAL_CONCERN}"), "no integrity violation"],
            "CONDITIONAL": f"beats empirical, mean ECE <= {CAL_CONCERN}, monotone not severe",
            "NOT READY": "otherwise",
            "phase5": "GO if all markets READY; CONDITIONAL GO if at least one; else NO-GO"},
        "calibration_bands": {"strong": CAL_STRONG, "acceptable": CAL_OK,
                              "concerning": CAL_CONCERN},
        "research_sources": list(RESEARCH_SOURCES),
    }


def write_run(run: Run4C1, inputs: Inputs, p4a: p4b.Phase4A, p4cc: Phase4C, eval_seasons,
              root: Path | None = None) -> Path:
    grid_df = pl.DataFrame([{"market": m, "threshold": k} for m, g in run.grids.items()
                            for k in g], schema={"market": pl.String, "threshold": pl.Int32})
    frames = {"cdf_predictions": run.predictions, "threshold_grid": grid_df,
              "fold_metadata": run.fold_metadata, "cdf_metrics": run.market_metrics,
              "cdf_calibration": run.threshold_metrics, "cdf_reliability": run.reliability,
              "zero_consistency": run.zero, "cdf_pmf": run.pmf,
              "quantile_consistency": run.quantiles, "monotonicity": run.monotonic,
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
        "artifact": "phase4c1_discrete_cdf", "path": rel.as_posix(),
        "built_at": built_at.isoformat(), "content_sha256": content,
        "contract_version": CONTRACT_VERSION, "feature_set_version": FEATURE_SET_VERSION,
        "baseline_set_version": BASELINE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
        "distribution_set_version": DISTRIBUTION_SET_VERSION, "cdf_set_version": CDF_SET_VERSION,
        "files": {k: {"sha256": shas[k], "rows": frames[k].height} for k in FILES},
        "phase4a": {"path": str(p4a.path), "content_sha256": p4a.manifest["content_sha256"],
                    "folds_sha256": p4a.manifest["files"]["folds"]["sha256"]},
        "phase4c": {"path": str(p4cc.path), "content_sha256": p4cc.manifest["content_sha256"],
                    "distribution_predictions_sha256":
                        p4cc.manifest["files"]["distribution_predictions"]["sha256"]},
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


def summarize(run: Run4C1) -> dict:
    return {
        "folds": run.fold_metadata["fold_id"].n_unique(),
        "fits": run.fold_metadata.height,
        "thresholds": {m: len(g) for m, g in run.grids.items()},
        "prediction_rows": run.predictions.height,
        "temporal_violations": run.fold_metadata.filter(
            pl.col("train_last_available_at") > pl.col("decision_time")).height,
        "monotone_violating_share": dict(run.monotonic.select("market", "violating_share")
                                         .iter_rows()),
        "status": dict(run.status.select("market", "status").iter_rows()),
        "phase5": run.decision,
    }


def validate_run(path: Path) -> dict:
    m = json.loads((path / "manifest.json").read_text())
    for k, meta in m["files"].items():
        blob = (path / f"{k}.parquet").read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            raise CDFError(f"{k}.parquet: checksum does not match the manifest")
    validate_frame(pl.read_parquet(path / "cdf_predictions.parquet"), "cdf_predictions")
    if m["summary"]["temporal_violations"]:
        raise CDFError("temporal violations recorded")
    if POINT_IN_TIME_WARNING not in m["warnings"]:
        raise CDFError("manifest lacks the point-in-time warning")
    return m
