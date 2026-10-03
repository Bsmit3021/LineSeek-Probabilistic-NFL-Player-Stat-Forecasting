"""Phase 4C run: distributional forecasts on the frozen Phase 4A/4B protocol (docs/adr/0006).

Reuses the validated Phase 4A run (folds, cohorts) and the audited Phase 4B training population
and feature matrix (`phase4b.modelling_table`/`build_tasks`). Predictions (quantile LightGBM,
mean LightGBM, zero classifier, empirical baseline) are serialised and checksummed before labels
are joined. Scoring is on the primary cohort, participant rows: the distribution is conditional
on participation; did-not-play rows are never zeros.
"""

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from lineseek.contracts import CONTRACT_VERSION, PROP_MARKETS, validate_frame
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation.phase4a import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    F3_NOTE,
    POINT_IN_TIME_WARNING,
    UNIVERSE_WARNING,
    Inputs,
    _modelling_rows,
    evaluation_root,
    histories_for,
    parquet_bytes,
    runtime_versions,
)
from lineseek.evaluation.scoring import (
    INTERVALS,
    QCOLS,
    QUANTILES,
    ZERO_BINS,
    bracket_error,
    cluster_bootstrap_mean,
    score_columns,
    zero_bin,
    zero_scores,
)
from lineseek.evaluation.universe import qb_role
from lineseek.evaluation.walkforward import walk_forward_folds
from lineseek.features.player_game import FEATURE_SET_VERSION
from lineseek.models.baselines import BASELINE_SET_VERSION
from lineseek.models.distribution import (
    COMPONENTS,
    DISTRIBUTION_SET_VERSION,
    EMPIRICAL_RULE,
    MEAN_PARAMS,
    QUANTILE_PARAMS,
    ZERO_PARAMS,
    DistributionError,
    empirical_history,
    fallback_distribution,
    fit_predict_component,
    repair_and_clip,
)
from lineseek.models.learned import COUNT_MARKETS, FEATURES, MODEL_SET_VERSION, SEED

RUN_DIR = "phase4c"
PREDICTORS = ("quantile_lgbm", "empirical")
KEY = ["game_id", "player_id", "market"]
# Calibration tolerances and promotion rule, frozen before any Phase 4C result (ADR 0006).
CAL_STRONG, CAL_ACCEPT, CAL_SEASON, CAL_REJECT = 0.03, 0.05, 0.07, 0.10
CROSSING_SEVERE = 0.10        # raw crossing rate above this -> REJECT (invalid distributions)
MIN_PINBALL_WINS = 4          # quantiles (of 5) where the model's pinball loss beats the baseline
PROMOTE, PROMISING, NO_IMPROVEMENT, REJECT = "PROMOTE", "PROMISING", "NO IMPROVEMENT", "REJECT"
RECONSTRUCTED_WARNING = ("Ten 2024-2025 raw snapshot manifests were reconstructed on 2026-10-03 "
                         "after macOS /tmp cleanup; their data checksums are verified against the "
                         "feature/label manifests, but git_revision/contract_version/"
                         "schema_variant/temporal_quality in them are unverified.")


def _check(msg: str, df: pl.DataFrame) -> None:
    if df.height:
        raise DistributionError(f"{df.height} {msg}: {df.head(5).rows()}")


# ---------------------------------------------------------------------------- frozen inputs

@dataclass
class Phase4B:
    path: Path
    manifest: dict
    l1: pl.DataFrame  # Phase 4B LightGBM-L1 predictions (the conditional-median point model)


def latest_run(kind: str, root: Path | None = None) -> Path:
    runs = sorted((root or evaluation_root()).glob(f"{kind}/*/manifest.json"))
    if not runs:
        raise FileNotFoundError(f"no {kind} run under the evaluation root")
    return runs[-1].parent


def load_phase4b(path: Path, p4a: p4b.Phase4A) -> Phase4B:
    m = p4b.validate_run(path)
    if m["phase4a"]["content_sha256"] != p4a.manifest["content_sha256"]:
        raise DistributionError("Phase 4B run was not built on this Phase 4A run")
    l1 = pl.read_parquet(path / "model_predictions.parquet").filter(
        pl.col("model") == "lightgbm").select(*KEY, pl.col("prediction").alias("l1_median"))
    return Phase4B(path, m, l1)


# ---------------------------------------------------------------------------- training

def run_component(task: p4b.MarketTask) -> dict:
    """Walk the frozen folds for one market and component (task.model = component)."""
    pred = np.full(len(task.x), np.nan)
    train_n = np.zeros(len(task.x), dtype=np.int64)
    meta = []
    fit_ok = ~np.isnan(task.y)
    for fold_id, dt in task.folds:
        test = task.decision == dt
        if not test.any():
            continue
        train = fit_ok & (task.available <= dt)
        if not train.any() or (train & test).any() or task.available[train].max() > dt:
            raise DistributionError(f"fold {fold_id}: training rows not available by the lock")
        pred[test] = fit_predict_component(task.model, task.x[train], task.y[train],
                                           task.x[test])
        train_n[test] = train.sum()
        meta.append({"fold_id": fold_id, "market": task.market, "component": task.model,
                     "train_rows": int(train.sum()),
                     "train_zero_rate": float((task.y[train] == 0).mean()),
                     "train_last_available_us": int(task.available[train].max()),
                     "test_rows": int(test.sum())})
    return {"market": task.market, "component": task.model, "pred": pred, "train_rows": train_n,
            "meta": meta}


def fallbacks(task: p4b.MarketTask) -> dict[int, dict[str, float]]:
    """Per fold: market-level distribution of the fold's training targets (cold-start rows)."""
    ok = ~np.isnan(task.y)
    return {fid: fallback_distribution(task.y[ok & (task.available <= dt)])
            for fid, dt in task.folds}


META_COLS = ["game_id", "player_id", "season", "week", "team", "opponent", "market",
             "kickoff_utc", "decision_time", "fold_id"]


def predict_distributions(table: pl.DataFrame, folds: pl.DataFrame, inputs: Inputs,
                          eval_seasons: tuple[int, ...], workers: int | None = None
                          ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Model and empirical distributions for every evaluation-season row. No label is read
    except the fit-eligible training targets (`y`) of available rows inside each fold."""
    base = {t.market: t for t in p4b.build_tasks(table, folds, ("lightgbm",))}
    tasks = [replace(t, model=c) for t in base.values() for c in COMPONENTS]
    if any(t.x.shape[1] != len(FEATURES) for t in tasks):
        raise DistributionError("model matrices must be exactly the frozen feature list")
    workers = min(len(tasks), os.cpu_count() or 1) if workers is None else workers
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(run_component, tasks))
    else:
        results = [run_component(t) for t in tasks]
    by = {(r["market"], r["component"]): r for r in results}
    common = {"feature_set_version": FEATURE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
              "distribution_set_version": DISTRIBUTION_SET_VERSION}
    parts = []
    for market in PROP_MARKETS:
        m = table.filter(pl.col("market") == market).select(*META_COLS, "prior_games_n")
        keep = (m["season"].is_in(list(eval_seasons))).to_numpy()
        raw = np.column_stack([by[(market, q)]["pred"] for q in QCOLS])[keep]
        rc = repair_and_clip(market, raw)
        mean_raw = by[(market, "mean")]["pred"][keep]
        mean = np.maximum(mean_raw, 0.0) if market in COUNT_MARKETS else mean_raw
        mm = m.filter(pl.Series(keep))
        parts.append(mm.with_columns(
            pl.lit("quantile_lgbm").alias("predictor"),
            *[pl.Series(c, rc["final"][:, i]) for i, c in enumerate(QCOLS)],
            *[pl.Series(f"{c}_raw", raw[:, i]) for i, c in enumerate(QCOLS)],
            pl.Series("crossed_raw", rc["crossed"]), pl.Series("repair_max", rc["repair_max"]),
            pl.Series("clipped_quantiles", rc["clipped"]).cast(pl.Int32),
            pl.Series("mean_prediction", mean), pl.Series("mean_raw", mean_raw),
            pl.Series("p_zero", by[(market, "zero")]["pred"][keep]),
            pl.col("prior_games_n").alias("history_n"),
            pl.Series("train_rows", by[(market, "q50")]["train_rows"][keep]).cast(pl.Int32),
            *[pl.lit(v).alias(k) for k, v in common.items()]))
    model = pl.concat(parts)
    # Empirical baseline: the player's eligible history, else the fold's market fallback.
    feats = pl.concat([inputs.features[s] for s in eval_seasons])
    hist = histories_for(inputs.feature_raw)
    emp = pl.concat([empirical_history(feats.filter(pl.col("season") == s), hist[s])
                     for s in eval_seasons])
    fb = pl.DataFrame([{"market": mk, "fold_id": fid, **{f"fb_{k}": v for k, v in d.items()}}
                       for mk, t in base.items() for fid, d in fallbacks(t).items()])
    e = (model.select(*META_COLS, "prior_games_n", "train_rows")
         .join(emp, on=KEY, how="left").join(fb, on=["market", "fold_id"], how="left"))
    _check("empirical history length differs from prior_games_n",
           e.filter(pl.col("history_n").fill_null(0) != pl.col("prior_games_n")))
    cold = pl.col("history_n").is_null()
    stats = [*QCOLS, "mean_prediction", "p_zero"]
    e = e.with_columns(*[pl.when(cold).then(pl.col(f"fb_{c}")).otherwise(pl.col(c)).alias(c)
                         for c in stats], pl.col("history_n").fill_null(0))
    e = e.with_columns(
        pl.lit("empirical").alias("predictor"),
        *[pl.col(c).alias(f"{c}_raw") for c in QCOLS], pl.lit(False).alias("crossed_raw"),
        pl.lit(0.0).alias("repair_max"), pl.lit(0, pl.Int32).alias("clipped_quantiles"),
        pl.col("mean_prediction").alias("mean_raw"),
        *[pl.lit(v).alias(k) for k, v in common.items()])
    out = pl.concat([model.drop("prior_games_n"),
                     e.select(model.drop("prior_games_n").columns)], how="vertical")
    out = out.select(list(ARTIFACT_SCHEMAS["distribution_predictions"])).sort(
        "season", "week", "game_id", "team", "player_id", "market", "predictor")
    validate_frame(out, "distribution_predictions")
    meta = pl.DataFrame([x for r in results for x in r["meta"]]).with_columns(
        pl.from_epoch("train_last_available_us", "us").dt.replace_time_zone("UTC")
        .alias("train_last_available_at")).drop("train_last_available_us").join(
        folds.select("fold_id", "decision_time"), on="fold_id").sort("fold_id", "market",
                                                                      "component")
    _check("fits trained on data unavailable at the decision time",
           meta.filter(pl.col("train_last_available_at") > pl.col("decision_time")))
    return out, meta


# ---------------------------------------------------------------------------- scoring

def scored_frame(preds: pl.DataFrame, inputs: Inputs, p4a: p4b.Phase4A, p4bb: Phase4B,
                 eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """Primary-cohort participant rows: frozen predictions + labels + QB role + L1 median."""
    lab = pl.concat([inputs.labels[s] for s in eval_seasons]).select(*KEY, "actual", "played")
    qb = pl.concat([qb_role(inputs.labels[s], inputs.label_raw[s]["schedules"],
                            inputs.label_raw[s]["rosters_weekly"]) for s in eval_seasons])
    s = (preds.join(lab, on=KEY).join(p4a.cohorts.select("game_id", "player_id",
                                                         "outcome_admitted"),
                                      on=["game_id", "player_id"])
         .join(qb, on=["game_id", "player_id"]).join(p4bb.l1, on=KEY, how="left"))
    _check("predictions without label/cohort/L1", s.filter(pl.any_horizontal(
        pl.col(c).is_null() for c in ("played", "outcome_admitted", "qb_role", "l1_median"))))
    s = s.filter(~pl.col("outcome_admitted"), pl.col("played"))
    _check("participant rows without an actual", s.filter(pl.col("actual").is_null()))
    return (s.with_columns(*score_columns(), *zero_scores(pl.col("p_zero"),
                                                          pl.col("actual") == 0),
                           _raw_wis().alias("wis_raw"), _unclipped_wis().alias("wis_unclipped"),
                           pl.col("history_n").gt(0).alias("has_history"))
            .sort(*KEY, "predictor"))


def _wis_from_pinball(cols: list[pl.Expr]) -> pl.Expr:
    """WIS = 2/(2K+1) * sum of pinball losses (Bracher et al. eq. 4); defined even when the
    quantiles cross, so it scores the raw (pre-repair) quantiles."""
    y = pl.col("actual")
    total = sum(((pl.when(y <= c).then(1.0).otherwise(0.0) - t) * (c - y))
                for c, t in zip(cols, QUANTILES, strict=True))
    return 2 / (2 * len(INTERVALS) + 1) * total


def _raw_wis() -> pl.Expr:
    return _wis_from_pinball([pl.col(f"{c}_raw") for c in QCOLS])


def _unclipped_wis() -> pl.Expr:
    """Repaired (sorted) but unclipped quantiles."""
    raw = [pl.col(f"{c}_raw") for c in QCOLS]
    srt = pl.concat_list(raw).list.sort()
    return _wis_from_pinball([srt.list.get(i) for i in range(len(QCOLS))])


def cohorts(s: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    qb = s.filter(pl.col("market") == "pass_attempts", pl.col("qb_role"))
    return [("primary", s), ("qb_role", qb)]


def periods(eval_seasons: tuple[int, ...]) -> tuple[str, list[tuple[str, pl.Expr]]]:
    all_ = f"{min(eval_seasons)}-{max(eval_seasons)}"
    return all_, [(all_, pl.lit(True))] + [(str(y), pl.col("season") == y) for y in eval_seasons]


def distribution_metrics(s: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    _, per = periods(eval_seasons)
    rows = []
    for cohort, c in cohorts(s):
        for view, v in (("all", c), ("has_history", c.filter(pl.col("has_history")))):
            for period, cond in per:
                d = v.filter(cond)
                if not d.height:
                    continue
                agg = d.group_by("predictor", "market", maintain_order=True).agg(
                    pl.len().alias("n"), pl.col("wis").mean(), pl.col("wis_raw").mean(),
                    pl.col("wis_unclipped").mean(),
                    *[pl.col(f"pinball_{q}").mean() for q in QCOLS], pl.col("pinball_mean").mean(),
                    *[pl.col(f"in{i}_{k}").mean().alias(f"cov{i}_{k}")
                      for i in INTERVALS for k in ("closed", "open")],
                    *[pl.col(f"width{i}").mean() for i in INTERVALS],
                    pl.col("brier").mean(), pl.col("log_loss").mean(),
                    pl.col("p_zero").mean().alias("mean_p_zero"),
                    (pl.col("actual") == 0).mean().alias("zero_rate"))
                rows.append(agg.with_columns(pl.lit(cohort).alias("cohort"),
                                             pl.lit(view).alias("view"),
                                             pl.lit(period).alias("period")))
    return pl.concat(rows).select("cohort", "view", "period", "predictor", "market",
                                  pl.exclude("cohort", "view", "period", "predictor", "market")
                                  ).sort("cohort", "view", "period", "predictor", "market")


def quantile_calibration(s: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    _, per = periods(eval_seasons)
    rows = []
    for cohort, c in cohorts(s):
        for period, cond in per:
            d = c.filter(cond)
            agg = d.group_by("predictor", "market", maintain_order=True).agg(
                pl.len().alias("n"), *[pl.col(f"lt_{q}").mean() for q in QCOLS],
                *[pl.col(f"le_{q}").mean() for q in QCOLS])
            for r in agg.iter_rows(named=True):
                for q, tau in zip(QCOLS, QUANTILES, strict=True):
                    lo, hi = r[f"lt_{q}"], r[f"le_{q}"]
                    rows.append({"cohort": cohort, "period": period, "predictor": r["predictor"],
                                 "market": r["market"], "quantile": q, "nominal": tau,
                                 "n": r["n"], "p_below": lo, "p_at_or_below": hi,
                                 "mid": (lo + hi) / 2, "error": bracket_error(tau, lo, hi)})
    return pl.DataFrame(rows).sort("cohort", "period", "predictor", "market", "quantile")


def band(err: float) -> str:
    return ("strong" if err <= CAL_STRONG else "acceptable" if err <= CAL_ACCEPT
            else "material")


def interval_calibration(m: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for r in m.filter(pl.col("view") == "all").iter_rows(named=True):
        for name, (_, _, alpha) in INTERVALS.items():
            target = 1 - alpha
            err = bracket_error(target, r[f"cov{name}_open"], r[f"cov{name}_closed"])
            rows.append({"cohort": r["cohort"], "period": r["period"],
                         "predictor": r["predictor"], "market": r["market"],
                         "interval": f"{name}%", "target": target, "n": r["n"],
                         "coverage_open": r[f"cov{name}_open"],
                         "coverage_closed": r[f"cov{name}_closed"],
                         "coverage_mid": (r[f"cov{name}_open"] + r[f"cov{name}_closed"]) / 2,
                         "error": err, "band": band(err), "mean_width": r[f"width{name}"]})
    return pl.DataFrame(rows).sort("cohort", "period", "predictor", "market", "interval")


def zero_calibration(s: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for cohort, c in cohorts(s):
        agg = c.with_columns(zero_bin()).group_by("predictor", "market", "p_zero_bin",
                                                  maintain_order=True).agg(
            pl.len().alias("n"), pl.col("p_zero").mean().alias("mean_predicted"),
            (pl.col("actual") == 0).mean().alias("observed_zero_rate"))
        rows.append(agg.with_columns(pl.lit(cohort).alias("cohort")))
    return pl.concat(rows).select("cohort", "predictor", "market", "p_zero_bin", "n",
                                  "mean_predicted", "observed_zero_rate").with_columns(
        (pl.col("observed_zero_rate") - pl.col("mean_predicted")).alias("gap")).sort(
        "cohort", "predictor", "market", "p_zero_bin")


def mean_vs_median(s: pl.DataFrame) -> pl.DataFrame:
    d = s.filter(pl.col("predictor") == "quantile_lgbm")
    y = pl.col("actual")
    out = d.group_by("market", maintain_order=True).agg(
        pl.len().alias("n"),
        *[(pl.col(c) - y).mean().alias(f"{n}_bias") for n, c in
          (("q50", "q50"), ("mean_model", "mean_prediction"), ("l1", "l1_median"))],
        *[(pl.col(c) - y).abs().mean().alias(f"{n}_mae") for n, c in
          (("q50", "q50"), ("mean_model", "mean_prediction"), ("l1", "l1_median"))],
        *[(pl.col(c) - y).pow(2).mean().sqrt().alias(f"{n}_rmse") for n, c in
          (("q50", "q50"), ("mean_model", "mean_prediction"), ("l1", "l1_median"))],
        (y <= pl.col("q50")).mean().alias("p_le_q50"), (y < pl.col("q50")).mean().alias("p_lt_q50"),
        (pl.col("q50") - pl.col("l1_median")).abs().mean().alias("q50_vs_l1_mean_abs_diff"),
        (pl.col("q50") - pl.col("l1_median")).abs().max().alias("q50_vs_l1_max_abs_diff"),
        pl.corr("q50", "l1_median").alias("q50_vs_l1_corr"),
        y.mean().alias("actual_mean"), y.median().alias("actual_median"))
    return out.sort("market")


def residual_shape(s: pl.DataFrame) -> pl.DataFrame:
    d = s.filter(pl.col("predictor") == "quantile_lgbm")
    rows = []
    for name, col in (("actual - q50", "q50"), ("actual - mean_prediction", "mean_prediction")):
        r = pl.col("actual") - pl.col(col)
        agg = d.group_by("market", maintain_order=True).agg(
            r.mean().alias("mean"), r.median().alias("median"), r.std().alias("std"),
            r.skew().alias("skewness"),
            *[r.quantile(q, interpolation="linear").alias(f"p{int(q * 100):02d}")
              for q in (0.05, 0.25, 0.75, 0.95)])
        rows.append(agg.with_columns(pl.lit(name).alias("residual")))
    return pl.concat(rows).sort("residual", "market")


def crossing_report(preds: pl.DataFrame, s: pl.DataFrame) -> pl.DataFrame:
    p = preds.filter(pl.col("predictor") == "quantile_lgbm")
    c = p.group_by("market", maintain_order=True).agg(
        pl.len().alias("rows"), pl.col("crossed_raw").sum().alias("crossed_rows"),
        pl.col("crossed_raw").mean().alias("crossing_rate"),
        pl.col("repair_max").filter(pl.col("crossed_raw")).mean().alias("mean_repair"),
        pl.col("repair_max").max().alias("max_repair"),
        (pl.col("clipped_quantiles") > 0).sum().alias("rows_clipped"),
        pl.col("clipped_quantiles").sum().alias("quantiles_clipped"),
        (pl.col("mean_raw") < 0).sum().alias("mean_negative_raw"))
    w = s.filter(pl.col("predictor") == "quantile_lgbm").group_by("market",
                                                                  maintain_order=True).agg(
        pl.col("wis_raw").mean().alias("wis_raw"),
        pl.col("wis_unclipped").mean().alias("wis_repaired_unclipped"),
        pl.col("wis").mean().alias("wis_final"))
    return c.join(w, on="market").sort("market")


def zero_quantile_consistency(s: pl.DataFrame) -> pl.DataFrame:
    """Count markets: does p_zero agree with where the quantiles put zero? For q in QUANTILES,
    Qq == 0 should hold iff P(y = 0) >= q (for y >= 0)."""
    d = s.filter(pl.col("predictor") == "quantile_lgbm", pl.col("market").is_in(
        list(COUNT_MARKETS)))
    return d.group_by("market", maintain_order=True).agg(
        pl.len().alias("n"),
        *[((pl.col(c) <= 0) != (pl.col("p_zero") >= t)).mean().alias(f"disagree_{c}")
          for c, t in zip(QCOLS, QUANTILES, strict=True)]).sort("market")


def compare_and_classify(s: pl.DataFrame, m: pl.DataFrame, ic: pl.DataFrame,
                         eval_seasons: tuple[int, ...], crossing: pl.DataFrame,
                         integrity_ok: bool) -> pl.DataFrame:
    """Frozen promotion rule per market (primary cohort, participant rows, all rows)."""
    all_, _ = periods(eval_seasons)
    head = m.filter(pl.col("cohort") == "primary", pl.col("view") == "all",
                    pl.col("period") == all_)
    rows = []
    for market in PROP_MARKETS:
        g = head.filter(pl.col("market") == market)
        mod = g.filter(pl.col("predictor") == "quantile_lgbm").row(0, named=True)
        emp = g.filter(pl.col("predictor") == "empirical").row(0, named=True)
        d = s.filter(pl.col("market") == market).pivot("predictor", index=KEY, values="wis")
        d = d.with_columns((pl.col("quantile_lgbm") - pl.col("empirical")).alias("dw"))
        point, lo, hi = cluster_bootstrap_mean(d, "dw", "game_id", BOOTSTRAP_REPS, BOOTSTRAP_SEED)
        pin_wins = sum(mod[f"pinball_{q}"] < emp[f"pinball_{q}"] for q in QCOLS)
        cal = ic.filter(pl.col("cohort") == "primary", pl.col("predictor") == "quantile_lgbm",
                        pl.col("market") == market)
        agg_err = cal.filter(pl.col("period") == all_)["error"].max()
        season_err = cal.filter(pl.col("period") != all_)["error"].max()
        cross = crossing.filter(pl.col("market") == market)["crossing_rate"].item()
        checks = {
            "wis_better": mod["wis"] < emp["wis"],
            "wis_ci_below_zero": hi < 0,
            "pinball_competitive": pin_wins >= MIN_PINBALL_WINS,
            "coverage_parity": mod["n"] == emp["n"],
            "intervals_acceptable": agg_err <= CAL_ACCEPT,
            "seasons_stable": season_err <= CAL_SEASON,
            "crossing_ok": cross <= CROSSING_SEVERE,
            "integrity_ok": integrity_ok,
        }
        if not integrity_ok or cross > CROSSING_SEVERE or agg_err > CAL_REJECT:
            status = REJECT
        elif not checks["wis_better"]:
            status = NO_IMPROVEMENT
        elif all(checks.values()):
            status = PROMOTE
        else:
            status = PROMISING
        rows.append({"market": market, "status": status, "wis_model": mod["wis"],
                     "wis_empirical": emp["wis"], "wis_delta": point, "wis_delta_ci_low": lo,
                     "wis_delta_ci_high": hi, "pinball_mean_model": mod["pinball_mean"],
                     "pinball_mean_empirical": emp["pinball_mean"], "pinball_wins": pin_wins,
                     "interval_error_max": agg_err, "season_interval_error_max": season_err,
                     "crossing_rate": cross, **checks})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------- run

@dataclass
class Run4C:
    predictions: pl.DataFrame
    fold_metadata: pl.DataFrame
    scored: pl.DataFrame
    metrics: pl.DataFrame
    quantile_cal: pl.DataFrame
    interval_cal: pl.DataFrame
    zero_cal: pl.DataFrame
    mean_median: pl.DataFrame
    residuals: pl.DataFrame
    crossing: pl.DataFrame
    consistency: pl.DataFrame
    status: pl.DataFrame
    predictions_sha256: str


def evaluate(inputs: Inputs, p4a: p4b.Phase4A, p4bb: Phase4B, eval_seasons: tuple[int, ...],
             workers: int | None = None) -> Run4C:
    p4b.check_same_inputs(inputs, p4a)
    rows = _modelling_rows(inputs.features, inputs.labels, p4a.cohorts)
    if not walk_forward_folds(rows, eval_seasons).equals(p4a.folds):
        raise DistributionError("recomputed folds differ from the frozen Phase 4A folds")
    if p4bb.manifest["phase4a"]["folds_sha256"] != p4a.manifest["files"]["folds"]["sha256"]:
        raise DistributionError("Phase 4B run used different folds")
    table = p4b.modelling_table(inputs, p4a)
    preds, meta = predict_distributions(table, p4a.folds, inputs, eval_seasons, workers)
    frozen = parquet_bytes(preds)
    preds = pl.read_parquet(frozen)  # score exactly the serialised predictions
    s = scored_frame(preds, inputs, p4a, p4bb, eval_seasons)
    m = distribution_metrics(s, eval_seasons)
    ic = interval_calibration(m)
    crossing = crossing_report(preds, s)
    integrity_ok = meta.filter(pl.col("train_last_available_at")
                               > pl.col("decision_time")).height == 0
    return Run4C(preds, meta, s, m, quantile_calibration(s, eval_seasons), ic,
                 zero_calibration(s), mean_vs_median(s), residual_shape(s), crossing,
                 zero_quantile_consistency(s),
                 compare_and_classify(s, m, ic, eval_seasons, crossing, integrity_ok),
                 hashlib.sha256(frozen).hexdigest())


FILES = ("distribution_predictions", "fold_metadata", "distribution_metrics",
         "quantile_calibration", "interval_calibration", "zero_calibration", "season_metrics",
         "mean_vs_median", "residual_shape", "crossing", "zero_quantile_consistency", "status")
RESEARCH_SOURCES = (
    ("LightGBM Parameters documentation: objective 'quantile' with 'alpha'; 'regression'; "
     "'binary'; deterministic/force_row_wise/seed"),
    ("Bracher, Ray, Gneiting & Reich (2021), Evaluating epidemic forecasts in an interval "
     "format, PLOS Comput Biol 17(2): e1008618, eq. 1-4 (WIS, weights w0=1/2, wk=ak/2)"),
    ("Gneiting & Raftery (2007), Strictly proper scoring rules, JASA (interval and quantile "
     "scores), as cited by Bracher et al."),
    ("Czado, Gneiting & Held (2009), Predictive model assessment for count data, Biometrics "
     "65(4): PIT is not uniform for discrete outcomes -> bracketed calibration"),
    ("Chernozhukov, Fernandez-Val & Galichon (2010), Quantile and probability curves without "
     "crossing, Econometrica: monotone rearrangement (row-wise sorting)"),
)


def parameters(eval_seasons, warmup) -> dict:
    return {
        "eval_seasons": list(eval_seasons), "warmup_seasons": list(warmup),
        "features": list(FEATURES), "quantiles": list(QUANTILES),
        "quantile_params": QUANTILE_PARAMS, "mean_params": MEAN_PARAMS,
        "zero_params": ZERO_PARAMS, "seed": SEED,
        "training_population": "Phase 4B: kickoff + 36h <= decision_time, primary cohort, "
                               "played; zero model target 1{actual == 0} on the same rows",
        "empirical_baseline": EMPIRICAL_RULE,
        "crossing_repair": "record raw crossings; row-wise sort (monotone rearrangement); then "
                           "count markets clipped at 0 (quantiles and mean); yards unclipped; "
                           "no rounding",
        "scoring": {"pinball": "(1{y<=q} - tau)(q - y)",
                    "wis": "1/(K+1/2)(1/2|y-m| + sum_k (a_k/2) IS_{a_k}), K=2: 50% [Q25,Q75] "
                           "a=0.5, 80% [Q10,Q90] a=0.2; equals 2/(2K+1) sum pinball",
                    "calibration": "integer outcomes: bracket [P(y<Q), P(y<=Q)] (intervals: "
                                   "open/closed); error = distance from nominal to bracket",
                    "zero": f"Brier, log loss (eps {1e-15}), bins {list(ZERO_BINS)}",
                    "population": "primary cohort, participant rows (DNP never zero)",
                    "uncertainty": f"WIS delta, {BOOTSTRAP_REPS} game-cluster resamples, seed "
                                   f"{BOOTSTRAP_SEED}"},
        "calibration_thresholds": {"strong": CAL_STRONG, "acceptable": CAL_ACCEPT,
                                   "season_investigate": CAL_SEASON, "reject": CAL_REJECT},
        "promotion_rule": {
            "PROMOTE": ["WIS lower than empirical baseline with game-clustered 95% CI < 0",
                        f"pinball lower on >= {MIN_PINBALL_WINS} of 5 quantiles",
                        "identical row coverage", f"50%/80% interval error <= {CAL_ACCEPT}",
                        f"every season's interval error <= {CAL_SEASON}",
                        f"raw crossing rate <= {CROSSING_SEVERE}", "no integrity violation"],
            "PROMISING": "WIS lower but another PROMOTE condition fails",
            "NO IMPROVEMENT": "WIS not lower than the empirical baseline",
            "REJECT": f"integrity failure, crossing rate > {CROSSING_SEVERE}, or aggregate "
                      f"interval error > {CAL_REJECT}"},
        "research_sources": list(RESEARCH_SOURCES),
    }


def write_run(run: Run4C, inputs: Inputs, p4a: p4b.Phase4A, p4bb: Phase4B,
              eval_seasons, warmup, root: Path | None = None) -> Path:
    all_, _ = periods(eval_seasons)
    frames = {"distribution_predictions": run.predictions, "fold_metadata": run.fold_metadata,
              "distribution_metrics": run.metrics, "quantile_calibration": run.quantile_cal,
              "interval_calibration": run.interval_cal, "zero_calibration": run.zero_cal,
              "season_metrics": run.metrics.filter(pl.col("period") != all_),
              "mean_vs_median": run.mean_median, "residual_shape": run.residuals,
              "crossing": run.crossing, "zero_quantile_consistency": run.consistency,
              "status": run.status}
    blobs = {k: parquet_bytes(v) for k, v in frames.items()}
    shas = {k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()}
    if shas["distribution_predictions"] != run.predictions_sha256:
        raise DistributionError("predictions changed after they were frozen")
    content = hashlib.sha256("".join(shas[k] for k in FILES).encode()).hexdigest()
    built_at = datetime.now(UTC)
    rel = Path(RUN_DIR) / f"{built_at:%Y%m%dT%H%M%S%fZ}_{content[:12]}"
    target = (root or evaluation_root()) / rel
    target.mkdir(parents=True, exist_ok=False)
    for k, b in blobs.items():
        with open(target / f"{k}.parquet", "xb") as f:
            f.write(b)
    manifest = {
        "artifact": "phase4c_distribution_evaluation", "path": rel.as_posix(),
        "built_at": built_at.isoformat(), "content_sha256": content,
        "contract_version": CONTRACT_VERSION, "feature_set_version": FEATURE_SET_VERSION,
        "baseline_set_version": BASELINE_SET_VERSION, "model_set_version": MODEL_SET_VERSION,
        "distribution_set_version": DISTRIBUTION_SET_VERSION,
        "files": {k: {"sha256": shas[k], "rows": frames[k].height} for k in FILES},
        "phase4a": {"path": str(p4a.path), "content_sha256": p4a.manifest["content_sha256"],
                    "folds_sha256": p4a.manifest["files"]["folds"]["sha256"]},
        "phase4b": {"path": str(p4bb.path), "content_sha256": p4bb.manifest["content_sha256"],
                    "model_predictions_sha256":
                        p4bb.manifest["files"]["model_predictions"]["sha256"]},
        "data_roots": {k: os.environ.get(v, d) for k, v, d in (
            ("raw", "LINESEEK_DATA_ROOT", "data/raw"), ("labels", "LINESEEK_LABELS_ROOT",
                                                       "data/labels"),
            ("features", "LINESEEK_FEATURES_ROOT", "data/features"),
            ("evaluation", "LINESEEK_EVALUATION_ROOT", "data/evaluation"))},
        "inputs": inputs.provenance,
        "parameters": parameters(eval_seasons, warmup),
        "runtime": runtime_versions() | {p: _v(p) for p in ("scikit-learn", "lightgbm")},
        "warnings": [POINT_IN_TIME_WARNING, UNIVERSE_WARNING, F3_NOTE, RECONSTRUCTED_WARNING],
        "summary": summarize(run),
    }
    with open(target / "manifest.json", "x") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    return target


def _v(pkg: str) -> str:
    from importlib.metadata import version
    return version(pkg)


def summarize(run: Run4C) -> dict:
    p = run.predictions.filter(pl.col("predictor") == "quantile_lgbm")
    all_ = run.metrics.filter(pl.col("period").str.contains("-"))["period"][0]
    ic = run.interval_cal.filter(pl.col("cohort") == "primary", pl.col("period") == all_,
                                 pl.col("predictor") == "quantile_lgbm")
    zr = run.metrics.filter(pl.col("cohort") == "primary", pl.col("view") == "all",
                            pl.col("period") == all_, pl.col("predictor") == "quantile_lgbm")
    return {
        "folds": run.fold_metadata["fold_id"].n_unique(),
        "fits": run.fold_metadata.height,
        "prediction_rows": run.predictions.height,
        "temporal_violations": run.fold_metadata.filter(
            pl.col("train_last_available_at") > pl.col("decision_time")).height,
        "raw_crossing_rows": int(p["crossed_raw"].sum()),
        "repaired_rows": int(p["crossed_raw"].sum()),
        "scored_participant_rows": run.scored.filter(pl.col("predictor") == "quantile_lgbm").height,
        "zero_rate_observed_vs_predicted": {
            r["market"]: (round(r["zero_rate"], 3), round(r["mean_p_zero"], 3))
            for r in zr.iter_rows(named=True)},
        "interval_coverage_closed": {f"{r['market']}/{r['interval']}": round(
            r["coverage_closed"], 3) for r in ic.iter_rows(named=True)},
        "status": dict(run.status.select("market", "status").iter_rows()),
    }


def validate_run(path: Path) -> dict:
    m = json.loads((path / "manifest.json").read_text())
    for k, meta in m["files"].items():
        blob = (path / f"{k}.parquet").read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            raise DistributionError(f"{k}.parquet: checksum does not match the manifest")
    p = validate_frame(pl.read_parquet(path / "distribution_predictions.parquet"),
                       "distribution_predictions")
    if set(p["predictor"]) != set(PREDICTORS):
        raise DistributionError("unexpected predictors")
    if m["summary"]["temporal_violations"]:
        raise DistributionError("temporal violations recorded")
    if POINT_IN_TIME_WARNING not in m["warnings"]:
        raise DistributionError("manifest lacks the point-in-time warning")
    return m
