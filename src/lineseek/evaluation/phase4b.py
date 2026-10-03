"""Phase 4B run: walk-forward Ridge/LightGBM point forecasts on the frozen Phase 4A protocol.

Folds, cohorts, benchmark choices and benchmark predictions are read from a validated Phase 4A
run (checksums in its manifest) and never rebuilt. Per fold and market a model is fitted on
rows whose outcome was available by the fold's decision_time, in the primary cohort, with the
player having played; it predicts every test row of that decision_time. Predictions are
serialised before labels are joined for scoring with the Phase 4A metric code. docs/adr/0005.
"""

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from lineseek.contracts import CONTRACT_VERSION, PROP_MARKETS, validate_frame
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.contracts.temporal import POSTGAME_STATS_LAG
from lineseek.evaluation.metrics import (
    TRACKS,
    cluster_bootstrap_diff,
    error_metrics,
    history_bucket,
    track_actuals,
)
from lineseek.evaluation.phase4a import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    COHORTS,
    F3_NOTE,
    POINT_IN_TIME_WARNING,
    UNIVERSE_WARNING,
    Inputs,
    _modelling_rows,
    compute_metrics,
    evaluation_root,
    parquet_bytes,
    runtime_versions,
)
from lineseek.evaluation.phase4a import (
    validate_run as validate_phase4a_run,
)
from lineseek.evaluation.universe import qb_role
from lineseek.evaluation.walkforward import walk_forward_folds
from lineseek.features.player_game import FEATURE_SET_VERSION
from lineseek.models.baselines import BASELINE_SET_VERSION, BASELINES
from lineseek.models.learned import (
    COUNT_MARKETS,
    FEATURES,
    HISTORY_ONLY_FEATURES,
    IMPUTER_PARAMS,
    LGBM_PARAMS,
    MODEL_SET_VERSION,
    MODELS,
    RIDGE_PARAMS,
    SEED,
    clip,
    feature_matrix,
    fit_predict,
    lightgbm_gain,
    ridge_coefficients,
)

RUN_DIR = "phase4b"
PHASE4A_RUN_DIR = "phase4a"
DIAGNOSTIC_MODELS = ("lightgbm_history_only",)  # ablation, not part of the model set
BENCHMARK = "benchmark"
PREDICTORS = (BENCHMARK, *MODELS, *DIAGNOSTIC_MODELS)
MODEL_FEATURES = {"ridge": FEATURES, "lightgbm": FEATURES,
                  "lightgbm_history_only": HISTORY_ONLY_FEATURES}
KEY = ["game_id", "player_id", "market"]
BENCHMARK_CRITERION = "best_mae_common_support"
HEADLINE = {"cohort": "primary", "track": "participant", "view": "common_support"}

# Promotion rule (ADR 0005), frozen before any Phase 4B result was computed.
SEASON_TIE_BAND = 0.005      # |season MAE delta| <= 0.5% of benchmark season MAE -> tie
MIN_SEASONS_WON = 3          # of 5
P90_MATERIAL = 0.05          # P90 > 5% worse: cannot PROMOTE (tail trade-off flagged)
P90_MAJOR = 0.10             # P90 > 10% worse: REJECT
COVERAGE_TOLERANCE = 0.01    # native coverage may not drop > 1 point vs benchmark
PROMOTE, PROMISING, NO_IMPROVEMENT, REJECT = "PROMOTE", "PROMISING", "NO IMPROVEMENT", "REJECT"


class Phase4BIntegrityError(ValueError):
    pass


def _check(msg: str, df: pl.DataFrame) -> None:
    if df.height:
        raise Phase4BIntegrityError(f"{df.height} {msg}: {df.head(5).rows()}")


# ---------------------------------------------------------------------------- frozen Phase 4A

@dataclass
class Phase4A:
    path: Path
    manifest: dict
    folds: pl.DataFrame
    predictions: pl.DataFrame
    cohorts: pl.DataFrame
    benchmarks: pl.DataFrame
    metrics: pl.DataFrame


def latest_phase4a(root: Path | None = None) -> Path:
    runs = sorted((root or evaluation_root()).glob(f"{PHASE4A_RUN_DIR}/*/manifest.json"))
    if not runs:
        raise FileNotFoundError("no Phase 4A run; run `lineseek-baselines build` first")
    return runs[-1].parent


def load_phase4a(path: Path) -> Phase4A:
    m = validate_phase4a_run(path)  # checksums, contracts, invariants
    read = {k: pl.read_parquet(path / f"{k}.parquet") for k in
            ("folds", "baseline_predictions", "cohorts", "benchmarks", "baseline_metrics")}
    return Phase4A(path, m, read["folds"], read["baseline_predictions"], read["cohorts"],
                   read["benchmarks"], read["baseline_metrics"])


def benchmark_baselines(p4a: Phase4A) -> dict[str, str]:
    """Frozen per-market benchmark: primary cohort, participant track, best MAE on common
    support (the Phase 4A selection, read from benchmarks.parquet)."""
    b = p4a.benchmarks.filter(pl.col("cohort") == "primary", pl.col("track") == "participant",
                              pl.col("criterion") == BENCHMARK_CRITERION)
    out = dict(b.select("market", "baseline").iter_rows())
    if set(out) != set(PROP_MARKETS) or not set(out.values()) <= set(BASELINES):
        raise Phase4BIntegrityError(f"unexpected Phase 4A benchmarks: {out}")
    return out


def check_same_inputs(inputs: Inputs, p4a: Phase4A) -> None:
    """Phase 4B must read exactly the feature/label artifacts Phase 4A was scored on."""
    def key(rows):
        return sorted((r["artifact"], r["season"], r["sha256"]) for r in rows)
    if key(inputs.provenance) != key(p4a.manifest["inputs"]):
        raise Phase4BIntegrityError("feature/label inputs differ from the Phase 4A run's")


# ---------------------------------------------------------------------------- training data

def modelling_table(inputs: Inputs, p4a: Phase4A) -> pl.DataFrame:
    """Every feature row (all seasons) with its fold, availability and training target.

    y is the labelled actual only for rows eligible for fitting (primary cohort, played); it is
    null otherwise. Whether a row is used is decided per fold by availability.
    """
    rows = _modelling_rows(inputs.features, inputs.labels, p4a.cohorts)
    feats = pl.concat([inputs.features[s] for s in sorted(inputs.features)])
    lab = pl.concat([inputs.labels[s] for s in sorted(inputs.labels)]).select(
        *KEY, "played", "actual")
    t = (feats.join(rows.select(*KEY, "outcome_admitted"), on=KEY)
         .join(lab, on=KEY)
         .join(p4a.folds.select("decision_time", "fold_id"), on="decision_time", how="left")
         .with_columns(
             (pl.col("kickoff_utc") + POSTGAME_STATS_LAG).alias("outcome_available_at"),
             pl.when(pl.col("played") & ~pl.col("outcome_admitted")).then(pl.col("actual"))
             .alias("y")))
    _check("fit-eligible rows without an actual",
           t.filter(pl.col("played") & ~pl.col("outcome_admitted") & pl.col("y").is_null()))
    return t.drop("played", "actual").sort("season", "week", "game_id", "team", "player_id",
                                           "market")


@dataclass(frozen=True)
class MarketTask:
    market: str
    model: str
    x: np.ndarray            # all rows of the market, MODEL_FEATURES[model]
    y: np.ndarray            # NaN = never fit-eligible
    available: np.ndarray    # outcome_available_at, int64 microseconds
    decision: np.ndarray     # decision_time, int64 microseconds
    kickoff: np.ndarray
    player: np.ndarray       # integer codes (only for the train player count)
    folds: tuple[tuple[int, int], ...]  # (fold_id, decision_time int64 us)


def _us(s: pl.Series) -> np.ndarray:
    return s.dt.cast_time_unit("us").cast(pl.Int64).to_numpy()


def run_market(task: MarketTask) -> dict:
    """Walk the frozen folds for one market and model. Estimators see X_train, y_train, X_test."""
    pred = np.full(len(task.x), np.nan)
    train_n = np.zeros(len(task.x), dtype=np.int64)
    meta, extra = [], []
    fit_ok = ~np.isnan(task.y)
    for fold_id, dt in task.folds:
        test = task.decision == dt
        if not test.any():
            continue
        train = fit_ok & (task.available <= dt)
        if not train.any():
            raise Phase4BIntegrityError(f"fold {fold_id}: no training rows")
        if (train & test).any() or task.available[train].max() > dt:
            raise Phase4BIntegrityError(f"fold {fold_id}: training rows not available by lock")
        t0 = time.perf_counter()
        raw, model = fit_predict(task.model, task.x[train], task.y[train], task.x[test])
        seconds = time.perf_counter() - t0
        pred[test], train_n[test] = raw, train.sum()
        meta.append({"fold_id": fold_id, "market": task.market, "model": task.model,
                     "train_rows": int(train.sum()),
                     "train_players": len(np.unique(task.player[train])),
                     "train_first_kickoff_us": int(task.kickoff[train].min()),
                     "train_last_available_us": int(task.available[train].max()),
                     "test_rows": int(test.sum()), "seconds": seconds})
        feats = MODEL_FEATURES[task.model]
        detail = (ridge_coefficients(model, feats) if task.model == "ridge"
                  else lightgbm_gain(model, feats))
        extra += [{"fold_id": fold_id, "market": task.market, "model": task.model,
                   "feature": f, "value": v} for f, v in detail.items()]
    return {"market": task.market, "model": task.model, "pred": pred, "train_rows": train_n,
            "meta": meta, "detail": extra}


def build_tasks(table: pl.DataFrame, folds: pl.DataFrame, models: tuple[str, ...]
                ) -> list[MarketTask]:
    fl = tuple((int(f), int(d)) for f, d in zip(folds["fold_id"], _us(folds["decision_time"]),
                                                  strict=True))
    tasks = []
    for market in PROP_MARKETS:
        m = table.filter(pl.col("market") == market)
        codes = m["player_id"].cast(pl.Categorical).to_physical().to_numpy()
        y = m["y"].cast(pl.Float64).to_numpy()
        for model in models:
            tasks.append(MarketTask(market, model, feature_matrix(m, MODEL_FEATURES[model]),
                                    y,
                                    _us(m["outcome_available_at"]), _us(m["decision_time"]),
                                    _us(m["kickoff_utc"]), codes, fl))
    return tasks


def predict_models(table: pl.DataFrame, folds: pl.DataFrame, eval_seasons: tuple[int, ...],
                   models: tuple[str, ...] = (*MODELS, *DIAGNOSTIC_MODELS),
                   workers: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame,
                                                        pl.DataFrame, pl.DataFrame]:
    """Model predictions for every evaluation-season row; no label column leaves this function.

    `workers` > 1 runs markets in separate processes (results do not depend on it: each
    task is a pure function of its own data with single-threaded, seeded estimators).
    """
    tasks = build_tasks(table, folds, models)
    workers = min(len(tasks), os.cpu_count() or 1) if workers is None else workers
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            results = list(ex.map(run_market, tasks))
    else:
        results = [run_market(t) for t in tasks]
    meta_cols = ["game_id", "player_id", "season", "week", "team", "opponent", "market",
                 "kickoff_utc", "decision_time", "fold_id"]
    parts, meta, detail = [], [], []
    for r in results:
        m = table.filter(pl.col("market") == r["market"]).select(meta_cols).with_columns(
            pl.Series("prediction_raw", r["pred"]), pl.Series("train_rows", r["train_rows"]))
        m = m.filter(pl.col("season").is_in(list(eval_seasons)))
        p, c = clip(r["market"], m["prediction_raw"].to_numpy())
        parts.append(m.with_columns(
            pl.lit(r["model"]).alias("model"), pl.Series("prediction", p),
            pl.Series("clipped", c), pl.col("train_rows").cast(pl.Int32),
            pl.lit(len(MODEL_FEATURES[r["model"]]), pl.Int32).alias("feature_count"),
            pl.lit(FEATURE_SET_VERSION).alias("feature_set_version"),
            pl.lit(MODEL_SET_VERSION).alias("model_set_version")))
        meta += r["meta"]
        detail += r["detail"]
    preds = pl.concat(parts).select(list(ARTIFACT_SCHEMAS["model_predictions"])).sort(
        "season", "week", "game_id", "team", "player_id", "market", "model")
    validate_frame(preds, "model_predictions")
    meta_df = pl.DataFrame(meta).with_columns(
        pl.from_epoch("train_first_kickoff_us", "us").dt.replace_time_zone("UTC")
        .alias("train_first_kickoff"),
        pl.from_epoch("train_last_available_us", "us").dt.replace_time_zone("UTC")
        .alias("train_last_available_at")).drop("train_first_kickoff_us",
                                                "train_last_available_us")
    meta_df = meta_df.join(folds.select("fold_id", "decision_time"), on="fold_id").sort(
        "fold_id", "market", "model")
    _check("fits whose training data was not available by the decision time",
           meta_df.filter(pl.col("train_last_available_at") > pl.col("decision_time")))
    timing = meta_df.select("fold_id", "market", "model", "seconds")
    meta_df = meta_df.drop("seconds").with_columns(
        pl.col("model").replace_strict(
            {m: json.dumps(model_params(m), sort_keys=True) for m in models},
            return_dtype=pl.String).alias("params"),
        pl.col("model").replace_strict({m: ",".join(MODEL_FEATURES[m]) for m in models},
                                       return_dtype=pl.String).alias("features"),
        pl.lit(SEED).alias("seed"))
    return preds, meta_df, timing, pl.DataFrame(detail).sort("model", "market", "fold_id",
                                                             "feature")


def model_params(model: str) -> dict:
    if model == "ridge":
        return {"pipeline": "FeatureUnion[SimpleImputer, MissingIndicator(all)] -> "
                            "StandardScaler -> Ridge", "imputer": IMPUTER_PARAMS,
                "missing_indicator": "features=all (fixed width)", "scaler": "StandardScaler()",
                "ridge": RIDGE_PARAMS}
    return {"lightgbm": LGBM_PARAMS, "missing": "native (NaN)"}


# ---------------------------------------------------------------------------- scoring

def scored_frame(preds: pl.DataFrame, p4a: Phase4A, inputs: Inputs, bench: dict[str, str],
                 qb: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """Frozen model + benchmark predictions joined with labels, cohorts, QB role and the
    Phase 4A common-support flag (all six baselines predict)."""
    b = p4a.predictions
    support = b.group_by(KEY).agg(pl.col("prediction").is_not_null().all().alias("common_support"))
    bench_rows = b.join(pl.DataFrame({"market": list(bench), "baseline": list(bench.values())}),
                        on=["market", "baseline"]).select(
        *KEY, "season", "decision_time", pl.lit(BENCHMARK).alias("model"), "prediction")
    allp = pl.concat([preds.select(*KEY, "season", "decision_time", "model", "prediction"),
                      bench_rows])
    feats = pl.concat([inputs.features[s] for s in eval_seasons]).select(*KEY, "prior_games_n")
    lab = track_actuals(pl.concat([inputs.labels[s] for s in eval_seasons])).select(
        *KEY, "played", *(f"actual_{t}" for t in TRACKS))
    out = (allp.join(feats, on=KEY, how="left").join(lab, on=KEY, how="left")
           .join(p4a.cohorts.select("game_id", "player_id", "outcome_admitted"),
                 on=["game_id", "player_id"], how="left")
           .join(qb, on=["game_id", "player_id"], how="left")
           .join(support, on=KEY, how="left")
           .with_columns(history_bucket())
           .sort(*KEY, "model"))  # fixed row order -> bit-stable aggregates
    _check("predictions without label/cohort/support", out.filter(
        pl.any_horizontal(pl.col(c).is_null() for c in
                          ("played", "outcome_admitted", "qb_role", "common_support"))))
    per = out.group_by(KEY).agg(pl.col("model").n_unique())
    _check("keys without every predictor", per.filter(pl.col("model") != len(PREDICTORS)))
    return out


def check_benchmark_reproduced(metrics: pl.DataFrame, p4a: Phase4A, bench: dict[str, str]) -> None:
    """Benchmark metrics recomputed here must equal the frozen Phase 4A metrics exactly."""
    a = metrics.filter(pl.col("model") == BENCHMARK).drop("model")
    f = p4a.metrics.join(pl.DataFrame({"market": list(bench), "baseline": list(bench.values())}),
                         on=["market", "baseline"]).drop("baseline")
    j = a.join(f, on=["cohort", "track", "view", "market", "period", "history_bucket"],
               suffix="_frozen")
    if j.height != a.height:
        raise Phase4BIntegrityError("benchmark metric slices missing from Phase 4A")
    for c in ("n_predicted", "mae", "rmse", "p90_ae"):
        _check(f"benchmark {c} differs from the frozen Phase 4A metric",
               j.filter(~pl.col(c).eq_missing(pl.col(f"{c}_frozen"))
                        & ((pl.col(c) - pl.col(f"{c}_frozen")).abs() > 1e-9)))


def compare_models(scored: pl.DataFrame, metrics: pl.DataFrame, eval_seasons: tuple[int, ...]
                   ) -> pl.DataFrame:
    """Each model vs the frozen benchmark per cohort x track x market (common support for
    accuracy, native for coverage), with season stability and a game-clustered MAE-delta CI."""
    period_all = f"{min(eval_seasons)}-{max(eval_seasons)}"
    m = metrics.filter(pl.col("history_bucket") == "all")
    rows = []
    for cohort in COHORTS:
        for track in TRACKS:
            a = f"actual_{track}"
            sc = scored.filter(pl.col("common_support"))
            if cohort == "primary":
                sc = sc.filter(~pl.col("outcome_admitted"))
            for market in PROP_MARKETS:
                sel = m.filter(pl.col("cohort") == cohort, pl.col("track") == track,
                               pl.col("market") == market)
                cs = sel.filter(pl.col("view") == "common_support")
                nat = sel.filter(pl.col("view") == "native", pl.col("period") == period_all)

                def get(df, model, col, period=period_all):
                    return df.filter(pl.col("model") == model,
                                     pl.col("period") == period)[col].item()
                ref = sc.filter(pl.col("market") == market, pl.col("model") == BENCHMARK).select(
                    *KEY, pl.col("prediction").alias("reference"))
                for model in (*MODELS, *DIAGNOSTIC_MODELS):
                    d = sc.filter(pl.col("market") == market, pl.col("model") == model).join(
                        ref, on=KEY)
                    point, lo, hi = cluster_bootstrap_diff(d, a, "game_id", BOOTSTRAP_REPS,
                                                           BOOTSTRAP_SEED)
                    won = tied = worse = 0
                    for s in eval_seasons:
                        bm, mm = get(cs, BENCHMARK, "mae", str(s)), get(cs, model, "mae", str(s))
                        if abs(mm - bm) <= SEASON_TIE_BAND * bm:
                            tied += 1
                        elif mm < bm:
                            won += 1
                        else:
                            worse += 1
                    r = {"cohort": cohort, "track": track, "market": market, "model": model}
                    for col in ("mae", "rmse", "median_ae", "p90_ae", "mean_error", "max_ae",
                                "n_predicted"):
                        r[col] = get(cs, model, col)
                        r[f"benchmark_{col}"] = get(cs, BENCHMARK, col)
                    r |= {"delta_mae": r["mae"] - r["benchmark_mae"],
                          "delta_rmse": r["rmse"] - r["benchmark_rmse"],
                          "delta_p90": r["p90_ae"] - r["benchmark_p90_ae"],
                          "p90_ratio": r["p90_ae"] / r["benchmark_p90_ae"]
                          if r["benchmark_p90_ae"] else None,
                          "delta_abs_bias": abs(r["mean_error"]) - abs(r["benchmark_mean_error"]),
                          "delta_mae_bootstrap_point": point, "delta_mae_ci_low": lo,
                          "delta_mae_ci_high": hi,
                          "coverage_native": get(nat, model, "coverage_rate"),
                          "benchmark_coverage_native": get(nat, BENCHMARK, "coverage_rate"),
                          "seasons_won": won, "seasons_tied": tied, "seasons_worse": worse}
                    r["coverage_diff"] = r["coverage_native"] - r["benchmark_coverage_native"]
                    rows.append(r)
    return pl.DataFrame(rows)


def classify(comparison: pl.DataFrame, integrity_ok: bool) -> pl.DataFrame:
    """Frozen promotion rule on the headline view (primary, participant, common support);
    rule 7 also requires the improvement to hold on the full label universe."""
    head = comparison.filter(pl.col("cohort") == "primary", pl.col("track") == "participant")
    full = comparison.filter(pl.col("cohort") == "full_label_universe",
                             pl.col("track") == "participant").select(
        "market", "model", pl.col("delta_mae").alias("full_universe_delta_mae"))
    out = []
    for r in head.join(full, on=["market", "model"]).iter_rows(named=True):
        better = r["delta_mae"] < 0
        checks = {
            "mae_lower": better,
            "ci_below_zero": r["delta_mae_ci_high"] < 0,
            "seasons_won_ok": r["seasons_won"] >= MIN_SEASONS_WON,
            "coverage_ok": r["coverage_diff"] >= -COVERAGE_TOLERANCE,
            "p90_ok": r["p90_ratio"] is not None and r["p90_ratio"] <= 1 + P90_MATERIAL,
            "integrity_ok": integrity_ok,
            "not_outcome_admitted_dependent": r["full_universe_delta_mae"] < 0,
        }
        major_tail = r["p90_ratio"] is not None and r["p90_ratio"] > 1 + P90_MAJOR
        if not integrity_ok or major_tail:
            status = REJECT
        elif all(checks.values()):
            status = PROMOTE
        elif better:
            status = PROMISING
        else:
            status = NO_IMPROVEMENT
        out.append({"market": r["market"], "model": r["model"], "status": status,
                    "major_tail_degradation": major_tail, **checks})
    return pl.DataFrame(out)


def select_point_models(status: pl.DataFrame, comparison: pl.DataFrame,
                        bench: dict[str, str]) -> pl.DataFrame:
    """Per market: the promoted model with the lowest headline MAE, else the benchmark."""
    head = comparison.filter(pl.col("cohort") == "primary", pl.col("track") == "participant")
    rows = []
    for market in PROP_MARKETS:
        s = status.filter(pl.col("market") == market, pl.col("status") == PROMOTE,
                          pl.col("model").is_in(list(MODELS)))
        cand = head.join(s.select("market", "model"), on=["market", "model"]).sort("mae")
        if cand.height:
            rows.append({"market": market, "selected_point_model": cand["model"][0],
                         "basis": PROMOTE})
        else:
            rows.append({"market": market,
                         "selected_point_model": f"phase4a:{bench[market]}",
                         "basis": "no learned model promoted"})
    return pl.DataFrame(rows)


def qb_diagnostic(scored: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """pass_attempts metrics on the full primary cohort and on its QB-role subset."""
    period_all = f"{min(eval_seasons)}-{max(eval_seasons)}"
    pa = scored.filter(pl.col("market") == "pass_attempts", ~pl.col("outcome_admitted"))
    parts = []
    for cohort, df in (("primary", pa), ("qb_role", pa.filter(pl.col("qb_role")))):
        for view in ("native", "common_support"):
            v = df if view == "native" else df.with_columns(
                pl.when(pl.col("common_support")).then(pl.col("prediction")).alias("prediction"))
            for track in TRACKS:
                for per, col in ((period_all, pl.lit(period_all)),
                                 (None, pl.col("season").cast(pl.String))):
                    e = error_metrics(v.with_columns(col.alias("period")), ["model", "period"],
                                      f"actual_{track}")
                    parts.append(e.with_columns(pl.lit(cohort).alias("cohort"),
                                                pl.lit(view).alias("view"),
                                                pl.lit(track).alias("track")))
    return pl.concat(parts).select("cohort", "view", "track", "model", "period",
                                   *[c for c in parts[0].columns
                                     if c not in ("cohort", "view", "track", "model", "period")]
                                   ).sort("cohort", "view", "track", "model", "period")


def clipping_report(preds: pl.DataFrame, scored: pl.DataFrame) -> pl.DataFrame:
    """Per model x count market: raw negatives, clipped rows, headline MAE raw vs clipped."""
    p = preds.filter(pl.col("market").is_in(list(COUNT_MARKETS)))
    s = scored.filter(pl.col("common_support"), ~pl.col("outcome_admitted"), pl.col("played"),
                      pl.col("model") != BENCHMARK).join(
        p.select(*KEY, "model", "prediction_raw"), on=[*KEY, "model"]).sort(*KEY, "model")
    # maintain_order: Polars' bare mean fast path is not bit-stable across runs.
    mae = s.group_by("model", "market", maintain_order=True).agg(
        (pl.col("prediction_raw") - pl.col("actual_participant")).abs().mean().alias("mae_raw"),
        (pl.col("prediction") - pl.col("actual_participant")).abs().mean().alias("mae_clipped"))
    counts = p.group_by("model", "market", maintain_order=True).agg(
        pl.len().alias("rows"), (pl.col("prediction_raw") < 0).sum().alias("raw_negative"),
        pl.col("clipped").sum().alias("clipped"))
    return counts.join(mae, on=["model", "market"]).with_columns(
        (pl.col("mae_clipped") - pl.col("mae_raw")).alias("mae_effect")).sort("model", "market")


def summarise_detail(detail: pl.DataFrame) -> pl.DataFrame:
    """Across folds: LightGBM gain share and Ridge standardised coefficient stability."""
    d = detail.sort("model", "market", "fold_id", "feature").with_columns(
        (pl.col("value") / pl.col("value").abs().sum().over("model", "market", "fold_id"))
        .alias("share"))
    return d.group_by("model", "market", "feature", maintain_order=True).agg(
        pl.col("value").mean().alias("mean"), pl.col("value").std().alias("std"),
        pl.col("share").mean().alias("mean_share"),
        (pl.col("value") > 0).mean().alias("frac_positive"), pl.len().alias("folds"),
    ).sort("model", "market", pl.col("mean_share").abs(), "feature",
           descending=[False, False, True, False]).with_columns(
        pl.int_range(1, pl.len() + 1).over("model", "market").cast(pl.Int32).alias("rank"))


# ---------------------------------------------------------------------------- run

@dataclass
class Run4B:
    predictions: pl.DataFrame
    fold_metadata: pl.DataFrame
    timing: pl.DataFrame
    detail: pl.DataFrame
    scored: pl.DataFrame
    metrics: pl.DataFrame
    comparison: pl.DataFrame
    status: pl.DataFrame
    selection: pl.DataFrame
    qb: pl.DataFrame
    clipping: pl.DataFrame
    importance: pl.DataFrame
    benchmarks: pl.DataFrame
    predictions_sha256: str
    folds_reused: bool


def evaluate(inputs: Inputs, p4a: Phase4A, eval_seasons: tuple[int, ...],
             workers: int | None = None) -> Run4B:
    check_same_inputs(inputs, p4a)
    bench = benchmark_baselines(p4a)
    rows = _modelling_rows(inputs.features, inputs.labels, p4a.cohorts)
    folds_reused = walk_forward_folds(rows, eval_seasons).equals(p4a.folds)
    if not folds_reused:
        raise Phase4BIntegrityError("recomputed folds differ from the frozen Phase 4A folds")
    table = modelling_table(inputs, p4a)
    preds, meta, timing, detail = predict_models(table, p4a.folds, eval_seasons,
                                                 workers=workers)
    frozen = parquet_bytes(preds)
    preds = pl.read_parquet(frozen)  # score exactly the serialised predictions
    qb = pl.concat([qb_role(inputs.labels[s], inputs.label_raw[s]["schedules"],
                            inputs.label_raw[s]["rosters_weekly"]) for s in eval_seasons])
    scored = scored_frame(preds, p4a, inputs, bench, qb, eval_seasons)
    metrics = compute_metrics(scored, eval_seasons, names=PREDICTORS, name_col="model")
    check_benchmark_reproduced(metrics, p4a, bench)
    comparison = compare_models(scored, metrics, eval_seasons)
    integrity_ok = folds_reused and meta.filter(
        pl.col("train_last_available_at") > pl.col("decision_time")).height == 0
    status = classify(comparison.filter(pl.col("model").is_in(list(MODELS))), integrity_ok)
    benchmarks = p4a.benchmarks.filter(pl.col("criterion") == BENCHMARK_CRITERION).with_columns(
        pl.lit(p4a.manifest["files"]["benchmarks"]["sha256"]).alias("phase4a_benchmarks_sha256"))
    return Run4B(preds, meta, timing, detail, scored, metrics, comparison, status,
                 select_point_models(status, comparison, bench), qb_diagnostic(scored, eval_seasons),
                 clipping_report(preds, scored), summarise_detail(detail), benchmarks,
                 hashlib.sha256(frozen).hexdigest(), folds_reused)


FILES = ("model_predictions", "model_fold_metadata", "model_metrics", "model_comparison",
         "model_status", "point_model_selection", "qb_diagnostic", "clipping",
         "feature_importance", "fold_feature_detail", "benchmarks")


def promotion_rule() -> dict:
    return {
        "view": HEADLINE | {"period": "eval seasons combined", "folds": "frozen Phase 4A"},
        "PROMOTE": ["common-support MAE lower than the frozen benchmark",
                    "game-clustered 95% CI of the MAE delta entirely below 0",
                    (f"MAE better (beyond a {SEASON_TIE_BAND:.1%} tie band) in >= "
                     f"{MIN_SEASONS_WON} of 5 seasons"),
                    f"native coverage not more than {COVERAGE_TOLERANCE:.0%} below the benchmark",
                    f"P90 AE not more than {P90_MATERIAL:.0%} worse",
                    "no temporal or protocol violation",
                    "MAE also lower on the full label universe (not outcome-admitted dependent)"],
        "PROMISING": "MAE lower, other PROMOTE conditions not all met",
        "NO IMPROVEMENT": "MAE equal or higher",
        "REJECT": f"integrity failure or P90 AE more than {P90_MAJOR:.0%} worse",
        "selection": "lowest-MAE PROMOTE model per market, else the frozen Phase 4A benchmark",
    }


def write_run(run: Run4B, inputs: Inputs, p4a: Phase4A, eval_seasons: tuple[int, ...],
              warmup: tuple[int, ...], root: Path | None = None) -> Path:
    frames = {"model_predictions": run.predictions, "model_fold_metadata": run.fold_metadata,
              "model_metrics": run.metrics, "model_comparison": run.comparison,
              "model_status": run.status, "point_model_selection": run.selection,
              "qb_diagnostic": run.qb, "clipping": run.clipping,
              "feature_importance": run.importance, "fold_feature_detail": run.detail,
              "benchmarks": run.benchmarks}
    blobs = {k: parquet_bytes(v) for k, v in frames.items()}
    shas = {k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()}
    if shas["model_predictions"] != run.predictions_sha256:
        raise Phase4BIntegrityError("predictions changed after they were frozen")
    content = hashlib.sha256("".join(shas[k] for k in FILES).encode()).hexdigest()
    built_at = datetime.now(UTC)
    rel = Path(RUN_DIR) / f"{built_at:%Y%m%dT%H%M%S%fZ}_{content[:12]}"
    target = (root or evaluation_root()) / rel
    target.mkdir(parents=True, exist_ok=False)
    for k, b in blobs.items():
        with open(target / f"{k}.parquet", "xb") as f:
            f.write(b)
    # Wall-clock fit times: kept for the record, excluded from the content hash.
    run.timing.write_parquet(target / "model_fold_timing.parquet")
    manifest = {
        "artifact": "phase4b_model_evaluation", "path": rel.as_posix(),
        "built_at": built_at.isoformat(), "content_sha256": content,
        "contract_version": CONTRACT_VERSION, "feature_set_version": FEATURE_SET_VERSION,
        "baseline_set_version": BASELINE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
        "files": {k: {"sha256": shas[k], "rows": frames[k].height} for k in FILES},
        "nondeterministic_files": ["model_fold_timing.parquet"],
        "phase4a": {"path": str(p4a.path), "content_sha256": p4a.manifest["content_sha256"],
                    "contract_version": p4a.manifest["contract_version"],
                    "folds_sha256": p4a.manifest["files"]["folds"]["sha256"],
                    "benchmarks_sha256": p4a.manifest["files"]["benchmarks"]["sha256"],
                    "baseline_predictions_sha256":
                        p4a.manifest["files"]["baseline_predictions"]["sha256"],
                    "cohorts_sha256": p4a.manifest["files"]["cohorts"]["sha256"],
                    "folds_reused_exactly": run.folds_reused},
        "inputs": inputs.provenance,
        "parameters": {
            "eval_seasons": list(eval_seasons), "warmup_seasons": list(warmup),
            "models": {m: model_params(m) for m in (*MODELS, *DIAGNOSTIC_MODELS)},
            "features": {m: list(MODEL_FEATURES[m]) for m in (*MODELS, *DIAGNOSTIC_MODELS)},
            "excluded_columns": "keys, ids, timestamps, provenance, versions, labels, cohorts, "
                                "fold ids (learned.FORBIDDEN); season and week",
            "training_population": "rows with kickoff_utc + 36h <= fold decision_time, primary "
                                   "cohort (not outcome_only), played == true; target = actual",
            "prediction_population": "every row of the fold's decision_time (DNP included); "
                                     "primary metrics exclude outcome_only",
            "clipping": "pass_attempts, rush_attempts, receptions: prediction = max(0, raw); "
                        "yards unclipped; no rounding",
            "qb_diagnostic": "pass_attempts rows whose game-day (ACT/INA) roster position is "
                             "QB; evaluation only, never a feature",
            "benchmarks": {"criterion": BENCHMARK_CRITERION, "per_market":
                           benchmark_baselines(p4a)},
            "promotion_rule": promotion_rule(),
            "seed": SEED, "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                                        "cluster": "game_id"},
            "research_sources": list(RESEARCH_SOURCES),
        },
        "runtime": runtime_versions() | {"scikit-learn": _v("scikit-learn"),
                                         "lightgbm": _v("lightgbm")},
        "warnings": [POINT_IN_TIME_WARNING, UNIVERSE_WARNING, F3_NOTE],
        "summary": summarize(run),
    }
    with open(target / "manifest.json", "x") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    return target


RESEARCH_SOURCES = (
    ("LightGBM Parameters documentation: objective regression_l1 (alias l1); "
     "deterministic=true with force_row_wise; seed; num_threads"),
    ("scikit-learn SimpleImputer documentation: add_indicator only marks features missing at "
     "fit (variable width); all-missing columns are dropped unless keep_empty_features=True"),
    "scikit-learn MissingIndicator(features='all') and Pipeline/FeatureUnion documentation",
)


def _v(pkg: str) -> str:
    from importlib.metadata import version
    return version(pkg)


def summarize(run: Run4B) -> dict:
    s = run.scored.filter(pl.col("model") == "ridge")
    return {
        "folds": run.predictions["fold_id"].n_unique(),
        "model_market_fits": run.fold_metadata.height,
        "prediction_rows": run.predictions.height,
        "temporal_violations": run.fold_metadata.filter(
            pl.col("train_last_available_at") > pl.col("decision_time")).height,
        "primary_rows": s.filter(~pl.col("outcome_admitted")).height,
        "participant_rows": s.filter(~pl.col("outcome_admitted"), pl.col("played")).height,
        "dnp_rows": s.filter(~pl.col("played")).height,
        "qb_diagnostic_rows": s.filter(pl.col("market") == "pass_attempts",
                                       ~pl.col("outcome_admitted"), pl.col("qb_role")).height,
        "clipped": dict(run.predictions.group_by("model").agg(pl.col("clipped").sum())
                        .sort("model").iter_rows()),
        "status": {f"{m}/{k}": v for m, k, v in run.status.select("market", "model", "status")
                   .iter_rows()},
        "selected_point_model": dict(run.selection.select("market", "selected_point_model")
                                     .iter_rows()),
    }


def validate_run(path: Path) -> dict:
    m = json.loads((path / "manifest.json").read_text())
    frames = {}
    for k, meta in m["files"].items():
        blob = (path / f"{k}.parquet").read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            raise Phase4BIntegrityError(f"{k}.parquet: checksum does not match the manifest")
        frames[k] = pl.read_parquet(blob)
    p = validate_frame(frames["model_predictions"], "model_predictions")
    if set(p["model"]) != {*MODELS, *DIAGNOSTIC_MODELS}:
        raise Phase4BIntegrityError(f"unexpected models {sorted(set(p['model']))}")
    _check("fits trained on data unavailable at the decision time",
           frames["model_fold_metadata"].filter(
               pl.col("train_last_available_at") > pl.col("decision_time")))
    if not m["phase4a"]["folds_reused_exactly"]:
        raise Phase4BIntegrityError("run did not reuse the Phase 4A folds")
    if POINT_IN_TIME_WARNING not in m["warnings"]:
        raise Phase4BIntegrityError("manifest lacks the point-in-time warning")
    return m
