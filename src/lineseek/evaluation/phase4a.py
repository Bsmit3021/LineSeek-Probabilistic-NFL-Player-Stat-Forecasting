"""Phase 4A run: frozen baseline predictions, walk-forward folds and metrics; docs/adr/0004.

Order is enforced: predictions are generated from feature artifacts (+ raw history for EWMA)
and serialised before any label is read; labels are joined only for scoring.
"""

import hashlib
import io
import json
import os
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import polars as pl

from lineseek.contracts import CONTRACT_VERSION, PROP_MARKETS, validate_frame
from lineseek.contracts.temporal import POSTGAME_STATS_LAG
from lineseek.evaluation.metrics import (
    HISTORY_BUCKETS,
    METRIC_COLUMNS,
    P90_QUANTILE,
    TRACKS,
    cluster_bootstrap_diff,
    error_metrics,
    history_bucket,
    track_actuals,
)
from lineseek.evaluation.universe import ADMISSION_PATHS, classify_universe
from lineseek.evaluation.walkforward import walk_forward_folds
from lineseek.features.player_game import (
    ARTIFACT as FEATURE_ARTIFACT,
)
from lineseek.features.player_game import (
    FEATURE_SET_VERSION,
    SeasonHistory,
    features_root,
    season_history,
    validate_features,
)
from lineseek.grade.labels import LABEL_INPUTS, labels_root
from lineseek.ingest.snapshot import data_root, git_revision, latest_manifest
from lineseek.models.baselines import (
    BASELINE_SET_VERSION,
    BASELINES,
    EWMA_ALPHA,
    EWMA_SPAN,
    FEATURE_BASELINES,
    predict_baselines,
)

EVALUATION_ROOT_ENV = "LINESEEK_EVALUATION_ROOT"
DEFAULT_EVALUATION_ROOT = "data/evaluation"
RUN_DIR = "phase4a"
WARMUP_SEASONS = (2018, 2019, 2020)
EVAL_SEASONS = (2021, 2022, 2023, 2024, 2025)  # frozen out-of-sample period
COHORTS = ("primary", "full_label_universe")
VIEWS = ("native", "common_support")
REFERENCE = "prev_game"
METRIC_GRAIN = ["cohort", "track", "view", "market", "baseline", "period", "history_bucket"]
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261002
KEY = ["game_id", "player_id", "market"]
POINT_IN_TIME_WARNING = (
    "Historical nflverse statistics are reconstructed from later snapshots. True point-in-time "
    "historical revisions are unavailable, so statistical revision leakage cannot be completely "
    "ruled out.")
UNIVERSE_WARNING = (
    "Primary metrics are game-day-roster-conditioned: outcome-admitted rows are excluded, but the "
    "remaining universe depends on game-day roster (ACT/INA) information published after the "
    "Wednesday lock. This is not a live Wednesday-lock backtest.")
F3_NOTE = ("F3 (pre-live): snap coverage is tracked per game, not per team; it affected no "
           "2018-2025 rows.")
RESEARCH_SOURCES = (
    ("Hyndman & Athanasopoulos, Forecasting: Principles and Practice (3rd ed.), 5.10 Time "
     "series cross-validation (rolling forecasting origin)"),
    ("Hyndman & Athanasopoulos, FPP3 5.8 Evaluating point forecast accuracy (MAPE undefined at "
     "zero; scale-dependent errors not comparable across units)"),
    "scikit-learn TimeSeriesSplit documentation (expanding train set, gap, no shuffling)",
)
FILES = ("baseline_predictions", "cohorts", "folds", "baseline_metrics", "baseline_comparison",
         "benchmarks")


class EvaluationIntegrityError(ValueError):
    pass


def evaluation_root() -> Path:
    return Path(os.environ.get(EVALUATION_ROOT_ENV, DEFAULT_EVALUATION_ROOT))


def _rank(col: str, values: tuple[str, ...]) -> pl.Expr:
    return pl.col(col).replace_strict({v: i for i, v in enumerate(values)}, return_dtype=pl.Int32)


BUCKET_ORDER = ("all", *(b[0] for b in HISTORY_BUCKETS))
ORDER_KEYS = (_rank("cohort", COHORTS), _rank("track", TRACKS), _rank("view", VIEWS),
              _rank("market", tuple(PROP_MARKETS)), _rank("baseline", BASELINES))


def _check(msg: str, df: pl.DataFrame) -> None:
    if df.height:
        raise EvaluationIntegrityError(f"{df.height} {msg}: {df.head(5).rows()}")


# ---------------------------------------------------------------------------- inputs

@dataclass
class Inputs:
    features: dict[int, pl.DataFrame]
    labels: dict[int, pl.DataFrame]
    # Raw frames per season as recorded in the artifacts' manifests (EWMA history, universe).
    feature_raw: dict[int, dict[int, dict[str, pl.DataFrame]]]
    label_raw: dict[int, dict[str, pl.DataFrame]]
    provenance: list[dict[str, str]] = field(default_factory=list)


def _read(root: Path, rel: str, sha: str) -> pl.DataFrame:
    blob = (root / rel).read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha:
        raise EvaluationIntegrityError(f"checksum mismatch for {rel}")
    return pl.read_parquet(blob)


def _raw_frames(inputs: list[dict], raw_root: Path) -> dict[int, dict[str, pl.DataFrame]]:
    out: dict[int, dict[str, pl.DataFrame]] = {}
    for i in inputs:
        if i["source"] in LABEL_INPUTS:
            season = int(i["path"].split("season=")[1].split("/")[0])
            out.setdefault(season, {})[i["source"]] = _read(raw_root, i["path"], i["sha256"])
    return out


def load_inputs(seasons: tuple[int, ...], eval_seasons: tuple[int, ...],
                feat_root: Path | None = None, label_root: Path | None = None,
                raw_root: Path | None = None) -> Inputs:
    """Latest feature and label artifacts per season (checksum-verified, built from the same
    labels), plus the exact raw snapshots their manifests list."""
    feat_root, label_root = feat_root or features_root(), label_root or labels_root()
    raw_root = raw_root or data_root()
    inp = Inputs({}, {}, {}, {})
    for s in seasons:
        fm = latest_manifest(FEATURE_ARTIFACT, s, feat_root)
        lm = latest_manifest("prop_labels", s, label_root)
        used = [i["sha256"] for i in fm["inputs"] if i["source"] == "prop_labels"]
        if used != [lm["sha256"]]:
            raise EvaluationIntegrityError(
                f"features {fm['path']} were built from labels {used}, latest is {lm['path']}")
        if fm["parameters"].get("feature_set_version") != FEATURE_SET_VERSION:
            raise EvaluationIntegrityError(f"{fm['path']}: feature set is not {FEATURE_SET_VERSION}")
        inp.features[s] = validate_features(_read(feat_root, fm["path"], fm["sha256"]))
        inp.labels[s] = validate_frame(_read(label_root, lm["path"], lm["sha256"]), "prop_labels")
        inp.label_raw[s] = _raw_frames(lm["inputs"], raw_root)[s]
        if s in eval_seasons:
            inp.feature_raw[s] = _raw_frames(fm["inputs"], raw_root)
        inp.provenance += [
            {"artifact": "features", "season": str(s), "path": fm["path"], "sha256": fm["sha256"]},
            {"artifact": "prop_labels", "season": str(s), "path": lm["path"],
             "sha256": lm["sha256"]},
        ]
    return inp


# ---------------------------------------------------------------------------- prediction

def build_predictions(features: dict[int, pl.DataFrame],
                      histories: dict[int, dict[int, SeasonHistory]]) -> pl.DataFrame:
    """Baseline predictions for the evaluation seasons' feature rows. No labels are passed."""
    return pl.concat([predict_baselines(features[s], histories[s]) for s in sorted(features)])


def histories_for(feature_raw: dict[int, dict[int, dict[str, pl.DataFrame]]]
                  ) -> dict[int, dict[int, SeasonHistory]]:
    return {t: {s: season_history(**frames) for s, frames in by_season.items()}
            for t, by_season in feature_raw.items()}


# ---------------------------------------------------------------------------- scoring

def build_cohorts(labels: dict[int, pl.DataFrame],
                  label_raw: dict[int, dict[str, pl.DataFrame]]) -> pl.DataFrame:
    return pl.concat([classify_universe(labels[s], **label_raw[s]) for s in sorted(labels)])


def _modelling_rows(features: dict[int, pl.DataFrame], labels: dict[int, pl.DataFrame],
                    cohorts: pl.DataFrame) -> pl.DataFrame:
    f = pl.concat([features[s] for s in sorted(features)]).select(
        *KEY, "season", "week", "kickoff_utc", "decision_time")
    lk = pl.concat([labels[s] for s in sorted(labels)]).select(KEY)
    _check("feature rows without a label", f.join(lk, on=KEY, how="anti"))
    _check("label rows without a feature row", lk.join(f, on=KEY, how="anti"))
    rows = f.join(cohorts.select("game_id", "player_id", "outcome_admitted"),
                  on=["game_id", "player_id"], how="left")
    _check("rows without a universe class", rows.filter(pl.col("outcome_admitted").is_null()))
    return rows


def score(predictions: pl.DataFrame, labels: dict[int, pl.DataFrame], cohorts: pl.DataFrame,
          folds: pl.DataFrame) -> pl.DataFrame:
    """Frozen predictions + labels + cohort + fold: one row per prediction."""
    lab = track_actuals(pl.concat([labels[s] for s in sorted(labels)])).select(
        *KEY, "played", "label_status", *(f"actual_{t}" for t in TRACKS))
    support = predictions.group_by(KEY).agg(
        pl.col("prediction").is_not_null().all().alias("common_support"))
    out = (predictions.join(lab, on=KEY, how="left")
           .join(cohorts.select("game_id", "player_id", "admission_path", "outcome_admitted"),
                 on=["game_id", "player_id"], how="left")
           .join(folds.select("decision_time", "fold_id"), on="decision_time", how="left")
           .join(support, on=KEY, how="left")
           .with_columns(history_bucket()))
    _check("predictions without a label", out.filter(pl.col("played").is_null()))
    _check("predictions without a universe class", out.filter(pl.col("admission_path").is_null()))
    _check("predictions outside every fold", out.filter(pl.col("fold_id").is_null()))
    return out


def compute_metrics(scored: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """cohort x track x view x market x baseline over (each season | all seasons) and, for all
    seasons, each history bucket."""
    period_all = f"{min(eval_seasons)}-{max(eval_seasons)}"
    parts = []
    for cohort in COHORTS:
        c = scored if cohort == "full_label_universe" else scored.filter(~pl.col("outcome_admitted"))
        for view in VIEWS:
            v = c if view == "native" else c.with_columns(
                pl.when(pl.col("common_support")).then(pl.col("prediction")).alias("prediction"))
            for track in TRACKS:
                a = f"actual_{track}"
                base = ["market", "baseline"]
                tags = [pl.lit(cohort).alias("cohort"), pl.lit(track).alias("track"),
                        pl.lit(view).alias("view")]
                by_season = error_metrics(v.with_columns(pl.col("season").cast(pl.String)
                                                         .alias("period")), [*base, "period"], a)
                overall = error_metrics(v.with_columns(pl.lit(period_all).alias("period")),
                                        [*base, "period"], a)
                buckets = error_metrics(v.with_columns(pl.lit(period_all).alias("period")),
                                        [*base, "period", "history_bucket"], a)
                for df, bucket in ((by_season, "all"), (overall, "all"), (buckets, None)):
                    if bucket:
                        df = df.with_columns(pl.lit(bucket).alias("history_bucket"))
                    parts.append(df.with_columns(tags).select(*METRIC_GRAIN, *METRIC_COLUMNS))
    out = pl.concat(parts)
    return out.sort(*ORDER_KEYS[:5], "period", _rank("history_bucket", BUCKET_ORDER))


def compare(scored: pl.DataFrame, metrics: pl.DataFrame, eval_seasons: tuple[int, ...]
            ) -> pl.DataFrame:
    """Each baseline vs prev_game, overall period, both views; game-clustered bootstrap
    interval for the common-support MAE difference (primary cohort)."""
    period_all = f"{min(eval_seasons)}-{max(eval_seasons)}"
    m = metrics.filter(pl.col("history_bucket") == "all")
    keys = ["cohort", "track", "view", "market"]
    ref = m.filter(pl.col("baseline") == REFERENCE).select(
        *keys, "period", *[pl.col(c).alias(f"ref_{c}") for c in ("mae", "rmse", "coverage_rate")])
    j = m.join(ref, on=[*keys, "period"])
    seasons = j.filter(pl.col("period") != period_all).group_by(*keys, "baseline").agg(
        (pl.col("mae") < pl.col("ref_mae")).sum().cast(pl.Int32).alias("seasons_mae_better"),
        (pl.col("mae") > pl.col("ref_mae")).sum().cast(pl.Int32).alias("seasons_mae_worse"))
    out = j.filter(pl.col("period") == period_all).join(seasons, on=[*keys, "baseline"]).select(
        *keys, "baseline", "period", "n_predicted", "coverage_rate", "mae", "rmse", "p90_ae",
        (pl.col("mae") - pl.col("ref_mae")).alias("mae_diff_vs_prev"),
        (pl.col("rmse") - pl.col("ref_rmse")).alias("rmse_diff_vs_prev"),
        (pl.col("coverage_rate") - pl.col("ref_coverage_rate")).alias("coverage_diff_vs_prev"),
        "seasons_mae_better", "seasons_mae_worse")
    ci = []
    prim = scored.filter(~pl.col("outcome_admitted"), pl.col("common_support"))
    for track in TRACKS:
        a = f"actual_{track}"
        for market in PROP_MARKETS:
            pm = prim.filter(pl.col("market") == market)
            ref_pred = pm.filter(pl.col("baseline") == REFERENCE).select(
                *KEY, pl.col("prediction").alias("reference"))
            for b in BASELINES:
                if b == REFERENCE:
                    continue
                d = pm.filter(pl.col("baseline") == b).join(ref_pred, on=KEY)
                _, lo, hi = cluster_bootstrap_diff(d, a, "game_id", BOOTSTRAP_REPS, BOOTSTRAP_SEED)
                ci.append({"cohort": "primary", "track": track, "view": "common_support",
                           "market": market, "baseline": b, "mae_diff_ci_low": lo,
                           "mae_diff_ci_high": hi})
    ci_df = pl.DataFrame(ci, schema={"cohort": pl.String, "track": pl.String, "view": pl.String,
                                     "market": pl.String, "baseline": pl.String,
                                     "mae_diff_ci_low": pl.Float64,
                                     "mae_diff_ci_high": pl.Float64})
    return out.join(ci_df, on=[*keys, "baseline"], how="left").sort(*ORDER_KEYS)


def select_benchmarks(comparison: pl.DataFrame) -> pl.DataFrame:
    """Per market and track (primary cohort): the baseline every Phase 4B model must beat."""
    rank = {b: i for i, b in enumerate(BASELINES)}
    rows = []
    prim = comparison.filter(pl.col("cohort") == "primary")
    criteria = (("best_mae_common_support", "common_support", "mae", min),
                ("best_rmse_common_support", "common_support", "rmse", min),
                ("best_mae_native", "native", "mae", min),
                ("highest_coverage_native", "native", "coverage_rate", max))
    for track in TRACKS:
        for market in PROP_MARKETS:
            for name, view, col, pick in criteria:
                c = prim.filter(pl.col("track") == track, pl.col("market") == market,
                                pl.col("view") == view)
                best = pick(c[col].to_list())
                tied = sorted(c.filter(pl.col(col) == best)["baseline"].to_list(), key=rank.get)
                rows.append({"cohort": "primary", "track": track, "market": market,
                             "criterion": name, "baseline": tied[0], "value": float(best),
                             "tied_with": ",".join(tied[1:])})
    return pl.DataFrame(rows)


@dataclass
class Run:
    predictions: pl.DataFrame
    cohorts: pl.DataFrame
    folds: pl.DataFrame
    scored: pl.DataFrame
    metrics: pl.DataFrame
    comparison: pl.DataFrame
    benchmarks: pl.DataFrame
    predictions_sha256: str


def parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd", statistics=True)
    return buf.getvalue()


def evaluate(inputs: Inputs, eval_seasons: tuple[int, ...]) -> Run:
    """Predict (features only), freeze, then classify, fold and score with labels."""
    feats = {s: inputs.features[s] for s in eval_seasons}
    predictions = build_predictions(feats, histories_for(inputs.feature_raw))
    frozen = parquet_bytes(predictions)
    predictions = pl.read_parquet(frozen)  # score exactly the serialised predictions
    cohorts = build_cohorts(inputs.labels, inputs.label_raw)
    rows = _modelling_rows(inputs.features, inputs.labels, cohorts)
    folds = walk_forward_folds(rows, eval_seasons)
    scored = score(predictions, {s: inputs.labels[s] for s in eval_seasons}, cohorts, folds)
    metrics = compute_metrics(scored, eval_seasons)
    comparison = compare(scored, metrics, eval_seasons)
    return Run(predictions, cohorts, folds, scored, metrics, comparison,
               select_benchmarks(comparison), hashlib.sha256(frozen).hexdigest())


# ---------------------------------------------------------------------------- storage

def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(),
            **{p: version(p) for p in ("polars", "pyarrow", "numpy", "nflreadpy", "lineseek")},
            "git_revision": git_revision() or "unknown"}


def parameters(eval_seasons: tuple[int, ...], warmup: tuple[int, ...]) -> dict:
    return {
        "eval_seasons": list(eval_seasons), "warmup_seasons": list(warmup),
        "baselines": {**{b: f"feature column {c}" for b, (c, _) in FEATURE_BASELINES.items()},
                      "ewma_span5": f"EWMA over Phase 3 eligible history, span={EWMA_SPAN}, "
                                    f"alpha={EWMA_ALPHA!r}, adjust=False, oldest-first"},
        "fold_rule": "one fold per unique decision_time of the eval seasons; test = rows with "
                     "that decision_time; train = rows with kickoff_utc + "
                     f"{POSTGAME_STATS_LAG} <= decision_time",
        "timing_rule": "decision_time = Wednesday 12:00 America/New_York strictly before kickoff; "
                       f"outcome available at kickoff_utc + {POSTGAME_STATS_LAG}",
        "universe_policy": {"paths": list(ADMISSION_PATHS), "primary": "exclude outcome_only",
                            "diagnostic": "full_label_universe (all paths)",
                            "warning": UNIVERSE_WARNING},
        "dnp_policy": {"participant": "played rows only; actual as labelled (0 is a real 0); "
                                      "equals the market track (non-participants void)",
                       "forecast_lock": "all rows; resolve_actual(actual, Track.FORECAST): "
                                        "did_not_play scores 0"},
        "metrics": {"error": "prediction - actual (positive mean_error = overprediction)",
                    "mae": "mean |error|", "rmse": "sqrt(mean error^2)",
                    "median_ae": "median |error|",
                    "p90_ae": f"quantile {P90_QUANTILE} of |error|, linear interpolation",
                    "max_ae": "max |error| (diagnostic)",
                    "coverage_rate": "n_predicted / n_eligible (eligible = scored on the track)",
                    "views": "native: every row a baseline predicts; common_support: rows where "
                             "all six baselines predict",
                    "uncertainty": f"MAE difference vs {REFERENCE}, paired, {BOOTSTRAP_REPS} "
                                   f"game-cluster bootstrap resamples, seed {BOOTSTRAP_SEED}, "
                                   "95% percentile interval",
                    "excluded": "MAPE (undefined at zero outcomes); any cross-market raw score"},
        "research_sources": list(RESEARCH_SOURCES),
        "limitations": [POINT_IN_TIME_WARNING, UNIVERSE_WARNING, F3_NOTE],
    }


def write_run(run: Run, inputs: Inputs, eval_seasons: tuple[int, ...],
              warmup: tuple[int, ...], root: Path | None = None) -> Path:
    """Append-only run directory: Parquet files + manifest with checksums and parameters."""
    frames = {"baseline_predictions": run.predictions, "cohorts": run.cohorts,
              "folds": run.folds, "baseline_metrics": run.metrics,
              "baseline_comparison": run.comparison, "benchmarks": run.benchmarks}
    blobs = {k: parquet_bytes(v) for k, v in frames.items()}
    shas = {k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()}
    if shas["baseline_predictions"] != run.predictions_sha256:
        raise EvaluationIntegrityError("predictions changed after they were frozen")
    content = hashlib.sha256("".join(shas[k] for k in FILES).encode()).hexdigest()
    built_at = datetime.now(UTC)
    rel = Path(RUN_DIR) / f"{built_at:%Y%m%dT%H%M%S%fZ}_{content[:12]}"
    target = (root or evaluation_root()) / rel
    target.mkdir(parents=True, exist_ok=False)
    for k, b in blobs.items():
        with open(target / f"{k}.parquet", "xb") as f:
            f.write(b)
    s = run.scored
    manifest = {
        "artifact": "phase4a_baseline_evaluation", "path": rel.as_posix(),
        "built_at": built_at.isoformat(), "content_sha256": content,
        "contract_version": CONTRACT_VERSION, "feature_set_version": FEATURE_SET_VERSION,
        "baseline_set_version": BASELINE_SET_VERSION,
        "files": {k: {"sha256": shas[k], "rows": frames[k].height} for k in FILES},
        "inputs": inputs.provenance,
        "parameters": parameters(eval_seasons, warmup),
        "runtime": runtime_versions(),
        "warnings": [POINT_IN_TIME_WARNING, UNIVERSE_WARNING],
        "summary": summarize(run),
        "scored_rows": s.height,
    }
    with open(target / "manifest.json", "x") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return target


def summarize(run: Run) -> dict:
    s = run.scored.filter(pl.col("baseline") == REFERENCE)
    pc = run.cohorts.group_by("admission_path").len()
    cov = run.predictions.group_by("baseline").agg(
        pl.col("prediction").is_not_null().mean().round(4))
    return {
        "folds": run.folds.height,
        "prediction_rows": run.predictions.height,
        "player_game_markets": s.height,
        "primary_rows": s.filter(~pl.col("outcome_admitted")).height,
        "full_label_universe_rows": s.height,
        "outcome_admitted_rows": s.filter(pl.col("outcome_admitted")).height,
        "participant_rows": s.filter(pl.col("played")).height,
        "did_not_play_rows": s.filter(~pl.col("played")).height,
        "admission_paths_player_games_all_seasons": dict(pc.select("admission_path", "len")
                                                         .sort("admission_path").iter_rows()),
        "coverage_by_baseline": dict(cov.sort("baseline").iter_rows()),
        "temporal_violations": int((run.folds["train_last_available_at"]
                                    > run.folds["decision_time"]).sum()),
    }


def validate_run(path: Path) -> dict:
    """Checksums, contracts and cross-file invariants of a run directory."""
    m = json.loads((path / "manifest.json").read_text())
    frames = {}
    for k, meta in m["files"].items():
        blob = (path / f"{k}.parquet").read_bytes()
        if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
            raise EvaluationIntegrityError(f"{k}.parquet: checksum does not match the manifest")
        frames[k] = pl.read_parquet(blob)
        if frames[k].height != meta["rows"]:
            raise EvaluationIntegrityError(f"{k}.parquet: row count differs from the manifest")
    p, folds, met = (validate_frame(frames["baseline_predictions"], "baseline_predictions"),
                     validate_frame(frames["folds"], "walk_forward_folds"),
                     frames["baseline_metrics"])
    if set(p["baseline"]) != set(BASELINES):
        raise EvaluationIntegrityError(f"baselines {sorted(set(p['baseline']))} != {BASELINES}")
    _check("prediction rows outside the folds' decision times",
           p.join(folds, on="decision_time", how="anti").select(KEY))
    per_key = p.group_by(KEY).len().filter(pl.col("len") != len(BASELINES))
    _check("player-game-markets without exactly one row per baseline", per_key)
    _check("cohort rows duplicated", frames["cohorts"].filter(
        pl.struct("game_id", "player_id").is_duplicated()))
    _check("metric rows with n_predicted > n_eligible or n_eligible > n_rows", met.filter(
        (pl.col("n_predicted") > pl.col("n_eligible")) | (pl.col("n_eligible") > pl.col("n_rows"))))
    if m["warnings"][0] != POINT_IN_TIME_WARNING:
        raise EvaluationIntegrityError("manifest lacks the point-in-time warning")
    return m
