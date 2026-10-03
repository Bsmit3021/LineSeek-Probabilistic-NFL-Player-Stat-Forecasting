import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from lineseek.contracts import ContractViolation, validate_frame
from lineseek.evaluation import models_cli
from lineseek.evaluation import phase4a as p4
from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation.metrics import cluster_bootstrap_diff
from lineseek.features.player_game import build_features, season_history
from lineseek.grade.labels import build_labels
from lineseek.models.learned import (
    FEATURES,
    FORBIDDEN,
    HISTORY_ONLY_FEATURES,
    LGBM_PARAMS,
    MODEL_SET_VERSION,
    MODELS,
    RIDGE_PARAMS,
    ModelInputError,
    clip,
    feature_matrix,
    fit_predict,
    make_ridge,
)
from tests.test_baselines import frames

KEY = ["game_id", "player_id", "market"]


@pytest.fixture(scope="module")
def world():
    fr = {s: frames(s) for s in (2024, 2025)}
    hist = {s: season_history(**fr[s]) for s in fr}
    labels = {s: build_labels(**fr[s]).labels for s in fr}
    feats = {2024: build_features(2024, labels[2024], {2024: hist[2024]},
                                  fr[2024]["schedules"]).features,
             2025: build_features(2025, labels[2025], hist, fr[2025]["schedules"]).features}
    w = p4.Inputs(feats, labels, {2025: fr}, fr)
    for s in (2024, 2025):
        w.provenance += [{"artifact": "features", "season": str(s), "path": f"f{s}",
                          "sha256": f"f{s}"},
                         {"artifact": "prop_labels", "season": str(s), "path": f"l{s}",
                          "sha256": f"l{s}"}]
    return w


@pytest.fixture(scope="module")
def p4a(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    path = p4.write_run(p4.evaluate(world, (2025,)), world, (2025,), (2024,), root=root)
    return p4b.load_phase4a(path)


@pytest.fixture(scope="module")
def run(world, p4a):
    return p4b.evaluate(world, p4a, (2025,), workers=1)


def preds_of(world, p4a, models=MODELS):
    table = p4b.modelling_table(world, p4a)
    return p4b.predict_models(table, p4a.folds, (2025,), models=models, workers=1)[0]


def mutate_labels(world, cond: pl.Expr, value: float = 777.0, scramble: bool = False):
    """Radically different but contract-valid outcomes for rows matching `cond`."""
    lab = world.labels[2025]
    played = pl.when(cond).then(~pl.col("played")).otherwise(pl.col("played")) if scramble \
        else pl.col("played")
    lab = lab.with_columns(played.alias("played")).with_columns(
        pl.when(cond & pl.col("played")).then(pl.lit(value))
        .when(cond).then(pl.lit(None, pl.Float64)).otherwise(pl.col("actual")).alias("actual"),
        pl.when(cond & pl.col("played")).then(pl.lit("recorded"))
        .when(cond).then(pl.lit("did_not_play")).otherwise(pl.col("label_status"))
        .alias("label_status"))
    return replace(world, labels={**world.labels, 2025: validate_frame(lab, "prop_labels")})


def last_fold(p4a) -> dict:
    return p4a.folds.sort("fold_id").row(-1, named=True)


def week_of_fold(world, fold) -> int:
    return world.features[2025].filter(pl.col("decision_time") == fold["decision_time"])[
        "week"].min()


# ---------------------------------------------------------------- Ridge

def test_ridge_tiny_exact_example():
    # x standardised to z = (-sqrt(1.5), 0, sqrt(1.5)); indicator column constant -> 0;
    # coef = sum(z * (y - 2)) / (sum z^2 + alpha) = 2 sqrt(1.5) / 4; at x = 4, z = 2 sqrt(1.5)
    # -> 2 + 2 sqrt(1.5) * 2 sqrt(1.5) / 4 = 3.5.
    x, y = np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0])
    raw, model = fit_predict("ridge", x, y, np.array([[4.0]]))
    assert RIDGE_PARAMS == {"alpha": 1.0}
    assert raw[0] == pytest.approx(3.5, abs=1e-12)
    assert model.named_steps["ridge"].coef_[0] == pytest.approx(2 * np.sqrt(1.5) / 4)


def test_ridge_preprocessing_is_fit_on_training_rows_only():
    train = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 30.0], [5.0, 20.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    test_a = np.array([[1e6, np.nan]])
    test_b = np.array([[-1e6, 5e7]])
    _, ma = fit_predict("ridge", train, y, test_a)
    _, mb = fit_predict("ridge", train, y, test_b)
    imp = ma.named_steps["features"].transformer_list[0][1]
    assert imp.statistics_.tolist() == [3.0, 20.0]  # medians of the training rows
    sc = ma.named_steps["scale"]
    assert sc.mean_[:2].tolist() == pytest.approx([3.0, 20.0])  # imputed training columns
    assert np.array_equal(sc.var_, mb.named_steps["scale"].var_)
    assert np.array_equal(ma.named_steps["ridge"].coef_, mb.named_steps["ridge"].coef_)


def test_ridge_keeps_fixed_width_with_empty_and_missing_columns():
    train = np.array([[1.0, np.nan, 2.0], [2.0, np.nan, 4.0], [3.0, np.nan, 6.0]])
    raw, m = fit_predict("ridge", train, np.array([1.0, 2.0, 3.0]),
                         np.array([[np.nan, 4.0, np.nan]]))
    assert np.isfinite(raw).all()
    assert m.named_steps["ridge"].coef_.shape == (6,)  # 3 imputed + 3 indicators, always


def test_ridge_is_deterministic():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(200, 5)), rng.normal(size=200)
    a, ma = fit_predict("ridge", x, y, x[:10])
    b, mb = fit_predict("ridge", x, y, x[:10])
    assert np.array_equal(a, b)
    assert np.array_equal(ma.named_steps["ridge"].coef_, mb.named_steps["ridge"].coef_)


def test_future_feature_distribution_cannot_alter_earlier_preprocessing(world, p4a, run):
    # Multiply every feature of the last fold's rows by 1000: earlier folds' imputer/scaler
    # statistics and predictions are unchanged.
    fold = last_fold(p4a)
    f = world.features[2025]
    late = pl.col("decision_time") == fold["decision_time"]
    num = [c for c in FEATURES if f.schema[c].is_numeric() and f.schema[c] != pl.Boolean]
    f2 = f.with_columns(pl.when(late).then(pl.col(c) * 1000).otherwise(pl.col(c)).alias(c)
                        for c in num)
    w2 = replace(world, features={**world.features, 2025: f2})
    table, table2 = p4b.modelling_table(world, p4a), p4b.modelling_table(w2, p4a)
    first = p4a.folds.sort("fold_id").row(0, named=True)
    for t in (table, table2):
        assert t.filter(pl.col("decision_time") == first["decision_time"]).height
    stats = []
    for t in (table, table2):
        m = t.filter(pl.col("market") == "receptions")
        avail = (m["outcome_available_at"] <= fold["decision_time"]) & m["y"].is_not_null()
        x = feature_matrix(m.filter(avail))
        model = make_ridge().fit(x, m.filter(avail)["y"].to_numpy())
        stats.append((model.named_steps["features"].transformer_list[0][1].statistics_,
                      model.named_steps["scale"].mean_, model.named_steps["scale"].var_))
    for a, b in zip(stats[0], stats[1], strict=True):
        assert np.array_equal(a, b)
    p2 = preds_of(w2, p4a)
    early = pl.col("decision_time") < fold["decision_time"]
    assert p2.filter(early).equals(run.predictions.filter(pl.col("model").is_in(MODELS), early))
    assert not p2.filter(late).equals(
        run.predictions.filter(pl.col("model").is_in(MODELS), late))


def test_identifiers_never_enter_the_feature_matrix(run):
    assert not set(FEATURES) & FORBIDDEN and not set(HISTORY_ONLY_FEATURES) & FORBIDDEN
    for col in ("player_id", "team", "opponent", "game_id", "market", "decision_time", "actual",
                "played", "label_status", "season", "week", "qb_role", "outcome_admitted"):
        with pytest.raises(ModelInputError):
            feature_matrix(pl.DataFrame({col: [1]}), (col,))
    params = json.loads(run.fold_metadata["params"][0])
    assert "lightgbm" in params or "ridge" in params
    assert set(run.fold_metadata["features"].unique()) == {
        ",".join(FEATURES), ",".join(HISTORY_ONLY_FEATURES)}


def test_count_clipping_and_yards_untouched():
    p, c = clip("receptions", np.array([-1.5, 0.0, 2.25]))
    assert p.tolist() == [0.0, 0.0, 2.25] and c.tolist() == [True, False, False]
    for yards in ("rush_yds", "reception_yds"):
        p, c = clip(yards, np.array([-7.5, 3.0]))
        assert p.tolist() == [-7.5, 3.0] and not c.any()


def test_prediction_contract_enforces_clipping_and_no_outcomes(run):
    p = run.predictions
    assert (p.filter(pl.col("clipped"))["prediction"] == 0).all()
    assert p.filter(pl.col("market").is_in(["rush_yds", "reception_yds"])).select(
        (pl.col("prediction") == pl.col("prediction_raw")).all()).item()
    bad = p.with_columns(pl.col("prediction_raw").alias("prediction"))
    if p.filter(pl.col("clipped")).height:
        with pytest.raises(ContractViolation, match="max\\(0, raw\\)"):
            validate_frame(bad, "model_predictions")
    with pytest.raises(ContractViolation, match="label-only"):
        validate_frame(p.with_columns(pl.lit(1.0).alias("actual")), "model_predictions")
    with pytest.raises(ContractViolation, match="duplicate"):
        validate_frame(pl.concat([p, p.head(1)]), "model_predictions")
    assert "actual" not in p.columns and "played" not in p.columns


# ---------------------------------------------------------------- LightGBM

def test_lightgbm_parameters_are_frozen():
    assert LGBM_PARAMS == {
        "objective": "l1", "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 100, "reg_lambda": 1.0, "subsample": 1.0, "subsample_freq": 0,
        "colsample_bytree": 1.0, "max_bin": 255, "deterministic": True,
        "force_row_wise": True, "n_jobs": 1, "random_state": 20261002, "verbose": -1}
    assert MODELS == ("ridge", "lightgbm") and MODEL_SET_VERSION == "1.0.0"


def test_lightgbm_tiny_deterministic_example_with_missing_values():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1000, 3))
    x[rng.random((1000, 3)) < 0.2] = np.nan
    y = np.where(np.nan_to_num(x[:, 0]) > 0, 10.0, 0.0)
    test = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    a, _ = fit_predict("lightgbm", x, y, test)
    b, _ = fit_predict("lightgbm", x, y, test)
    assert np.array_equal(a, b) and np.isfinite(a).all()
    assert a[0] == pytest.approx(10.0, abs=0.5) and a[1] == pytest.approx(0.0, abs=0.5)


def test_training_population_excludes_dnp_and_outcome_admitted(world, p4a, run):
    t = p4b.modelling_table(world, p4a)
    lab = pl.concat([world.labels[s] for s in (2024, 2025)]).select(*KEY, "played")
    j = t.join(lab, on=KEY)
    assert j.filter(~pl.col("played"))["y"].is_null().all()
    assert j.filter(pl.col("outcome_admitted"))["y"].is_null().all()
    assert j.filter(pl.col("played"), ~pl.col("outcome_admitted"))["y"].is_not_null().all()
    # train_rows per fit equals an independent count of eligible available rows
    for r in run.fold_metadata.filter(pl.col("market") == "rush_yds").iter_rows(named=True):
        n = j.filter(pl.col("market") == "rush_yds", pl.col("played"),
                     ~pl.col("outcome_admitted"),
                     pl.col("outcome_available_at") <= r["decision_time"]).height
        assert r["train_rows"] == n


def test_dnp_rows_still_receive_predictions(run, world):
    dnp = world.labels[2025].filter(~pl.col("played")).select(KEY)
    p = run.predictions.join(dnp, on=KEY)
    assert p.height == dnp.height * (len(MODELS) + 1) and p["prediction"].is_not_null().all()


# ---------------------------------------------------------------- temporal mutations

def test_target_fold_outcomes_cannot_change_target_predictions(world, p4a, run):
    fold = last_fold(p4a)
    wk = week_of_fold(world, fold)
    w2 = mutate_labels(world, pl.col("week") == wk, 999.0, scramble=True)
    p2 = preds_of(w2, p4a, models=(*MODELS, *p4b.DIAGNOSTIC_MODELS))
    assert p4.parquet_bytes(p2) == p4.parquet_bytes(run.predictions)
    # Scoring does respond to the changed truth: the benchmark no longer reproduces the frozen
    # Phase 4A metrics (real runs pin labels by checksum, so this guard must fire here).
    with pytest.raises(p4b.Phase4BIntegrityError, match="differs from the frozen"):
        p4b.evaluate(w2, p4a, (2025,), workers=1)


def test_future_labels_cannot_change_earlier_folds(world, p4a, run):
    folds = p4a.folds.sort("fold_id")
    mid = folds.row(folds.height // 2, named=True)
    wk = week_of_fold(world, mid)
    p2 = preds_of(mutate_labels(world, pl.col("week") >= wk, 55.0), p4a)
    base = run.predictions.filter(pl.col("model").is_in(MODELS))
    upto = pl.col("decision_time") <= mid["decision_time"]
    assert p2.filter(upto).equals(base.filter(upto))
    later = pl.col("decision_time") > mid["decision_time"]
    for model in MODELS:  # legitimately downstream fits change
        assert not p2.filter(later, pl.col("model") == model).equals(
            base.filter(later, pl.col("model") == model))


def test_historical_feature_change_flows_downstream_only(world, p4a, run):
    folds = p4a.folds.sort("fold_id")
    mid = folds.row(folds.height // 2, named=True)
    f = world.features[2025].with_columns(
        pl.when(pl.col("decision_time") == mid["decision_time"])
        .then(pl.col("market_roll5_mean") + 25).otherwise(pl.col("market_roll5_mean"))
        .alias("market_roll5_mean"))
    p2 = preds_of(replace(world, features={**world.features, 2025: f}), p4a)
    base = run.predictions.filter(pl.col("model").is_in(MODELS))
    before = pl.col("decision_time") < mid["decision_time"]
    assert p2.filter(before).equals(base.filter(before))
    assert not p2.filter(~before).equals(base.filter(~before))


def test_scrambled_target_labels_leave_predictions_identical(world, p4a, run):
    fold = last_fold(p4a)
    w2 = mutate_labels(world, pl.col("week") == week_of_fold(world, fold), 400.0,
                       scramble=True)
    p2 = preds_of(w2, p4a, models=(*MODELS, *p4b.DIAGNOSTIC_MODELS))
    assert p4.parquet_bytes(p2) == p4.parquet_bytes(run.predictions)


def test_predictions_are_reproducible_across_processes(world, p4a, run):
    table = p4b.modelling_table(world, p4a)
    two = p4b.predict_models(table, p4a.folds, (2025,), workers=2)[0]
    assert p4.parquet_bytes(two) == p4.parquet_bytes(run.predictions)


# ---------------------------------------------------------------- evaluation protocol

def test_phase4a_folds_and_benchmarks_are_reused(run, p4a):
    assert run.folds_reused
    used = run.predictions.select("fold_id", "decision_time").unique().sort("fold_id")
    assert used.equals(p4a.folds.select("fold_id", "decision_time").sort("fold_id"))
    assert (run.fold_metadata["train_last_available_at"]
            <= run.fold_metadata["decision_time"]).all()
    b = run.benchmarks
    assert set(b["phase4a_benchmarks_sha256"]) == {
        p4a.manifest["files"]["benchmarks"]["sha256"]}


def test_tampered_phase4a_benchmarks_are_rejected(p4a, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(p4a.path, d)
    b = pl.read_parquet(d / "benchmarks.parquet").with_columns(pl.lit("prev_game")
                                                               .alias("baseline"))
    b.write_parquet(d / "benchmarks.parquet")
    with pytest.raises(p4.EvaluationIntegrityError, match="checksum"):
        p4b.load_phase4a(d)


def test_benchmark_metrics_reproduce_the_frozen_phase4a_values(run, p4a):
    bench = p4b.benchmark_baselines(p4a)
    p4b.check_benchmark_reproduced(run.metrics, p4a, bench)
    with pytest.raises(p4b.Phase4BIntegrityError):
        p4b.check_benchmark_reproduced(
            run.metrics.with_columns(pl.col("mae") + 0.01), p4a, bench)


def test_participant_and_forecast_lock_scoring(run):
    s = run.scored.filter(pl.col("model") == "ridge", ~pl.col("outcome_admitted"),
                          pl.col("market") == "receptions")
    m = run.metrics.filter(pl.col("model") == "ridge", pl.col("cohort") == "primary",
                           pl.col("view") == "native", pl.col("market") == "receptions",
                           pl.col("period") == "2025", pl.col("history_bucket") == "all")
    part = m.filter(pl.col("track") == "participant").row(0, named=True)
    lock = m.filter(pl.col("track") == "forecast_lock").row(0, named=True)
    pl_ = s.filter(pl.col("played"))
    assert part["n_eligible"] == pl_.height
    assert part["mae"] == pytest.approx((pl_["prediction"] - pl_["actual_participant"])
                                        .abs().mean())
    assert lock["n_eligible"] == s.height
    dnp = s.filter(~pl.col("played"))
    assert (dnp["actual_forecast_lock"] == 0).all()
    assert lock["mae"] == pytest.approx((s["prediction"] - s["actual_forecast_lock"])
                                        .abs().mean())


def test_common_support_rows_are_identical_across_predictors(run):
    m = run.metrics.filter(pl.col("view") == "common_support", pl.col("history_bucket") == "all")
    n = m.group_by("cohort", "track", "market", "period").agg(
        pl.col("n_predicted").n_unique().alias("k"))
    assert (n["k"] == 1).all()


def test_qb_role_is_diagnostic_only(run, world, p4a):
    assert "qb_role" in FORBIDDEN and "qb_role" not in FEATURES
    qb = run.scored.filter(pl.col("market") == "pass_attempts", pl.col("qb_role"))
    assert qb.height and qb.filter(~pl.col("qb_role")).height == 0
    assert set(run.qb["cohort"]) == {"primary", "qb_role"}
    full = run.qb.filter(pl.col("cohort") == "primary")["n_rows"].sum()
    assert run.qb.filter(pl.col("cohort") == "qb_role")["n_rows"].sum() < full


def test_cluster_bootstrap_resamples_games_not_rows():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(40), 5)
    df = pl.DataFrame({"game_id": g.astype(str), "prediction": rng.normal(size=200),
                       "reference": rng.normal(size=200), "a": rng.normal(size=200)})
    one = cluster_bootstrap_diff(df, "a", "game_id", 500, 1)
    twice = cluster_bootstrap_diff(pl.concat([df, df]), "a", "game_id", 500, 1)
    assert one == pytest.approx(twice)  # duplicating rows inside games changes nothing


def _row(**kw):
    base = {"market": "receptions", "model": "ridge", "delta_mae": -0.05,
            "delta_mae_ci_high": -0.01, "seasons_won": 4, "coverage_diff": 0.0,
            "p90_ratio": 1.0}
    return base | kw


def test_promotion_rule():
    def status(primary, full_delta=-0.05, ok=True):
        comp = pl.DataFrame([
            {"cohort": "primary", "track": "participant", **primary},
            {"cohort": "full_label_universe", "track": "participant",
             **(primary | {"delta_mae": full_delta})}])
        return p4b.classify(comp, ok)["status"][0]
    assert status(_row()) == p4b.PROMOTE
    assert status(_row(delta_mae_ci_high=0.01)) == p4b.PROMISING
    assert status(_row(seasons_won=2)) == p4b.PROMISING
    assert status(_row(coverage_diff=-0.02)) == p4b.PROMISING
    assert status(_row(p90_ratio=1.07)) == p4b.PROMISING
    assert status(_row(), full_delta=0.01) == p4b.PROMISING
    assert status(_row(delta_mae=0.0, delta_mae_ci_high=0.02)) == p4b.NO_IMPROVEMENT
    assert status(_row(p90_ratio=1.11)) == p4b.REJECT
    assert status(_row(), ok=False) == p4b.REJECT


def test_selection_keeps_the_benchmark_unless_a_model_is_promoted(run, p4a):
    bench = p4b.benchmark_baselines(p4a)
    st = pl.DataFrame([{"market": m, "model": "ridge", "status": p4b.PROMISING}
                       for m in bench])
    sel = p4b.select_point_models(st, run.comparison, bench)
    assert sel["selected_point_model"].to_list() == [f"phase4a:{bench[m]}" for m in
                                                     sel["market"]]
    st2 = st.with_columns(pl.when(pl.col("market") == "rush_yds").then(pl.lit(p4b.PROMOTE))
                          .otherwise(pl.col("status")).alias("status"))
    sel2 = dict(p4b.select_point_models(st2, run.comparison, bench).select(
        "market", "selected_point_model").iter_rows())
    assert sel2["rush_yds"] == "ridge"


def test_clipping_report_matches_predictions(run):
    c = run.clipping
    for r in c.iter_rows(named=True):
        p = run.predictions.filter(pl.col("model") == r["model"], pl.col("market") == r["market"])
        assert r["raw_negative"] == r["clipped"] == p["clipped"].sum()


def test_write_validate_cli_and_determinism(run, world, p4a, tmp_path, capsys):
    a = p4b.write_run(run, world, p4a, (2025,), (2024,), root=tmp_path)
    again = p4b.evaluate(world, p4a, (2025,), workers=1)
    b = p4b.write_run(again, world, p4a, (2025,), (2024,), root=tmp_path)
    ma, mb = p4b.validate_run(a), p4b.validate_run(b)
    assert ma["files"] == mb["files"] and ma["content_sha256"] == mb["content_sha256"]
    assert ma["phase4a"]["folds_reused_exactly"]
    assert {"lightgbm", "scikit-learn", "polars", "pyarrow"} <= set(ma["runtime"])
    assert p4.POINT_IN_TIME_WARNING in ma["warnings"]
    assert models_cli.main(["validate", str(a)]) == 0
    assert "OK" in capsys.readouterr().out
    (a / "model_predictions.parquet").write_bytes(b"x")
    with pytest.raises(p4b.Phase4BIntegrityError, match="checksum"):
        p4b.validate_run(a)


def test_inputs_must_match_the_phase4a_run(world, p4a):
    bad = replace(world, provenance=[{**r, "sha256": "zz"} for r in world.provenance])
    with pytest.raises(p4b.Phase4BIntegrityError, match="differ"):
        p4b.evaluate(bad, p4a, (2025,), workers=1)


def test_training_matrices_are_exactly_the_frozen_features(world, p4a):
    table = p4b.modelling_table(world, p4a)
    tasks = p4b.build_tasks(table, p4a.folds, (*MODELS, *p4b.DIAGNOSTIC_MODELS))
    assert len(tasks) == 5 * 3
    for t in tasks:
        feats = p4b.MODEL_FEATURES[t.model]
        expect = feature_matrix(table.filter(pl.col("market") == t.market), feats)
        assert t.x.shape[1] == len(feats)
        assert np.array_equal(t.x, expect, equal_nan=True)


def test_tuesday_game_is_not_trainable_at_the_next_wednesday_lock():
    # Rows: 60 Sunday games (available), one Tuesday game kicked off before the lock but
    # available only on Thursday (huge target), and the Sunday test game after the lock.
    from datetime import UTC, datetime, timedelta
    us = lambda d: int(d.timestamp() * 1_000_000)
    lock = datetime(2021, 12, 22, 17, tzinfo=UTC)  # Wed 12:00 ET
    sundays = [lock - timedelta(days=3, hours=-1) - timedelta(days=7 * i) for i in range(60)]
    tuesday = datetime(2021, 12, 22, 0, tzinfo=UTC)  # Tue 19:00 ET
    kick = [*sundays, tuesday, lock + timedelta(days=4)]
    lag = timedelta(hours=36)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(len(kick), len(p4b.MODEL_FEATURES["ridge"])))
    y = np.r_[rng.normal(size=60), 1e6, np.nan]
    task = p4b.MarketTask(
        "rush_yds", "ridge", x, y, np.array([us(k + lag) for k in kick]),
        np.array([us(k) for k in kick[:-1]] + [us(lock)]), np.array([us(k) for k in kick]),
        np.arange(len(kick)), ((0, us(lock)),))
    out = p4b.run_market(task)
    assert out["meta"][0]["train_rows"] == 60  # the Tuesday game is excluded
    assert abs(out["pred"][-1]) < 100  # its huge target never reached the fit
