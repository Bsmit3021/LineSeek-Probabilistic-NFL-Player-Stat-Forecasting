import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from lineseek.contracts import ContractViolation, validate_frame
from lineseek.evaluation import distributions_cli
from lineseek.evaluation import phase4a as p4
from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c as p4c
from lineseek.evaluation.scoring import (
    QCOLS,
    bracket_error,
    cluster_bootstrap_mean,
    score_columns,
    zero_scores,
)
from lineseek.models.distribution import (
    DISTRIBUTION_SET_VERSION,
    MEAN_PARAMS,
    QUANTILE_PARAMS,
    ZERO_PARAMS,
    empirical_history,
    fallback_distribution,
    fit_predict_component,
    repair_and_clip,
)
from lineseek.models.learned import FEATURES, LGBM_PARAMS
from tests.test_baselines import HURTS, SUN, appearances, hist_of
from tests.test_models import mutate_labels, p4a, world  # noqa: F401 (fixtures)

KEY = ["game_id", "player_id", "market"]


@pytest.fixture(scope="module")
def p4bb(world, p4a, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("p4b")
    path = p4b.write_run(p4b.evaluate(world, p4a, (2025,), workers=1), world, p4a, (2025,),
                         (2024,), root=root)
    return p4c.load_phase4b(path, p4a)


@pytest.fixture(scope="module")
def run(world, p4a, p4bb):  # noqa: F811
    return p4c.evaluate(world, p4a, p4bb, (2025,), workers=1)


def preds_of(w, p4a_):
    table = p4b.modelling_table(w, p4a_)
    return p4c.predict_distributions(table, p4a_.folds, w, (2025,), workers=1)[0]


# ---------------------------------------------------------------- scoring (hand-computed)

def scores(y, q):
    df = pl.DataFrame({"actual": [float(y)], **{c: [float(v)] for c, v in zip(QCOLS, q,
                                                                              strict=True)}})
    return df.with_columns(score_columns()).row(0, named=True)


def test_pinball_loss_exact():
    r = scores(3, [5, 5, 5, 5, 5])  # y below q: (1 - tau)(q - y)
    assert r["pinball_q10"] == pytest.approx(0.9 * 2) and r["pinball_q90"] == pytest.approx(0.1 * 2)
    r = scores(7, [5, 5, 5, 5, 5])  # y above q: tau (y - q)
    assert r["pinball_q10"] == pytest.approx(0.1 * 2) and r["pinball_q90"] == pytest.approx(0.9 * 2)
    assert r["pinball_q50"] == pytest.approx(1.0)


def test_wis_exact_and_equals_scaled_pinball_sum():
    # q = (0, 2, 4, 6, 10), y = 12. Median 1/2*8 = 4; IS_.5 = 4 + 4*6 = 28 (x .25 = 7);
    # IS_.2 = 10 + 10*2 = 30 (x .1 = 3); WIS = (4 + 7 + 3) / 2.5 = 5.6.
    r = scores(12, [0, 2, 4, 6, 10])
    assert r["wis"] == pytest.approx(5.6)
    pin = sum(r[f"pinball_{c}"] for c in QCOLS)  # 1.2 + 2.5 + 4 + 4.5 + 1.8 = 14
    assert pin == pytest.approx(14.0) and r["wis"] == pytest.approx(2 / 5 * pin)
    assert r["pinball_mean"] == pytest.approx(14.0 / 5)
    inside = scores(4, [0, 2, 4, 6, 10])  # widths only: (0.25*4 + 0.1*10) / 2.5 = 0.8
    assert inside["wis"] == pytest.approx(0.8)


def test_interval_and_quantile_coverage_flags_with_ties():
    r = scores(2, [0, 2, 4, 6, 10])
    assert r["in50_closed"] and not r["in50_open"]  # on the boundary
    assert r["in80_closed"] and r["in80_open"]
    assert r["le_q25"] and not r["lt_q25"] and r["lt_q50"]
    assert bracket_error(0.25, 0.20, 0.30) == 0
    assert bracket_error(0.50, 0.55, 0.70) == pytest.approx(0.05)
    assert bracket_error(0.90, 0.70, 0.86) == pytest.approx(0.04)


def test_zero_scores_and_bounds():
    df = pl.DataFrame({"p": [0.8, 0.1, 1.0], "z": [True, False, False]}).with_columns(
        zero_scores(pl.col("p"), pl.col("z")))
    assert df["brier"].to_list() == pytest.approx([0.04, 0.01, 1.0])
    assert df["log_loss"][0] == pytest.approx(-math.log(0.8))
    assert math.isfinite(df["log_loss"][2])  # clipped, never infinite


def test_cluster_bootstrap_groups_by_game():
    rng = np.random.default_rng(1)
    df = pl.DataFrame({"g": np.repeat(np.arange(30), 4).astype(str), "v": rng.normal(size=120)})
    assert cluster_bootstrap_mean(df, "v", "g", 300, 2) == pytest.approx(
        cluster_bootstrap_mean(pl.concat([df, df]), "v", "g", 300, 2))


# ---------------------------------------------------------------- quantiles: repair, domain

def test_crossing_repair_is_sorting_and_recorded():
    r = repair_and_clip("rush_yds", np.array([[3.0, 1.0, 2.0, 5.0, 4.0], [1, 2, 3, 4, 5.0]]))
    assert r["final"][0].tolist() == [1, 2, 3, 4, 5] and r["crossed"].tolist() == [True, False]
    assert r["repair_max"].tolist() == [2.0, 0.0]


def test_count_clipping_and_negative_yardage():
    c = repair_and_clip("receptions", np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]]))
    assert c["final"][0].tolist() == [0, 0, 0, 1, 2] and c["clipped"][0] == 2
    y = repair_and_clip("rush_yds", np.array([[-6.0, -1.0, 3.0, 9.0, 20.0]]))
    assert y["final"][0].tolist() == [-6, -1, 3, 9, 20] and y["clipped"][0] == 0


def test_parameters_differ_from_phase4b_only_in_objective():
    for prm in QUANTILE_PARAMS.values():
        diff = {k for k in {*prm, *LGBM_PARAMS} if prm.get(k) != LGBM_PARAMS.get(k)}
        assert diff == {"objective", "alpha"} and prm["objective"] == "quantile"
    for prm, obj in ((MEAN_PARAMS, "regression"), (ZERO_PARAMS, "binary")):
        assert {k for k in {*prm, *LGBM_PARAMS} if prm.get(k) != LGBM_PARAMS.get(k)} == {
            "objective"} and prm["objective"] == obj
    assert [p["alpha"] for p in QUANTILE_PARAMS.values()] == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert DISTRIBUTION_SET_VERSION == "1.0.0"


def test_quantile_models_are_calibrated_on_a_continuous_fixture():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6000, 2))
    y = 10 + 3 * x[:, 0] + rng.gamma(2.0, 2.0, size=6000)  # skewed noise
    xt = rng.normal(size=(4000, 2))
    yt = 10 + 3 * xt[:, 0] + rng.gamma(2.0, 2.0, size=4000)
    for c, tau in zip(QCOLS, (0.1, 0.25, 0.5, 0.75, 0.9), strict=True):
        q = fit_predict_component(c, x, y, xt)
        assert abs((yt <= q).mean() - tau) < 0.03, c
    mean = fit_predict_component("mean", x, y, xt)
    assert abs((mean - yt).mean()) < 0.15  # mean model ~unbiased; median would be below
    q50 = fit_predict_component("q50", x, y, xt)
    assert (q50 - yt).mean() < (mean - yt).mean() - 0.3


def test_zero_classifier_probabilities():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3000, 1))
    y = np.where(rng.random(3000) < 1 / (1 + np.exp(-2 * x[:, 0])), 0.0, 3.0)
    p = fit_predict_component("zero", x, y, np.array([[-3.0], [0.0], [3.0]]))
    assert ((0 <= p) & (p <= 1)).all() and p[0] < 0.2 and p[2] > 0.8
    assert fit_predict_component("zero", x, np.zeros(3000), x[:2]).tolist() == [1.0, 1.0]


# ---------------------------------------------------------------- empirical baseline

def test_empirical_history_quantiles_exact():
    app = appearances([("2025_01_DAL_PHI", 2025, SUN, 10), ("2025_02_PHI_KC", 2025,
                                                           SUN + timedelta(days=7), 4),
                       ("2025_03_LA_PHI", 2025, SUN + timedelta(days=14), 7),
                       ("2025_09_DAL_PHI", 2025, SUN + timedelta(days=56), 999)])  # target
    kick = SUN + timedelta(days=56)
    from lineseek.contracts import forecast_time_utc
    t = pl.DataFrame({"game_id": ["2025_09_DAL_PHI"], "player_id": [HURTS],
                      "season": pl.Series([2025], dtype=pl.Int32),
                      "decision_time": pl.Series([forecast_time_utc(kick)],
                                                 dtype=pl.Datetime("us", "UTC"))})
    r = empirical_history(t, hist_of(app)).filter(pl.col("market") == "rush_yds").row(
        0, named=True)
    # sorted history 4, 7, 10 (target 999 excluded): linear type-7 quantiles
    assert r["history_n"] == 3
    assert [r[c] for c in QCOLS] == pytest.approx([4.6, 5.5, 7.0, 8.5, 9.4])
    assert r["mean_prediction"] == pytest.approx(7.0) and r["p_zero"] == 0.0


def test_cold_start_fallback_uses_training_targets():
    fb = fallback_distribution(np.array([0.0, 0.0, 1.0, 3.0]))
    assert fb["q50"] == pytest.approx(0.5) and fb["p_zero"] == 0.5 and fb["q90"] == pytest.approx(
        2.4)


def test_cold_start_rows_use_their_fold_fallback(run, world, p4a):  # noqa: F811
    table = p4b.modelling_table(world, p4a)
    e = run.predictions.filter(pl.col("predictor") == "empirical", pl.col("history_n") == 0)
    assert e.height
    r = e.row(0, named=True)
    dt = r["decision_time"]
    y = table.filter(pl.col("market") == r["market"], pl.col("y").is_not_null(),
                     pl.col("outcome_available_at") <= dt)["y"].to_numpy()
    fb = fallback_distribution(y)
    assert [r[c] for c in QCOLS] == pytest.approx([fb[c] for c in QCOLS])


def test_empirical_baseline_matches_feature_history(run, world):  # noqa: F811
    e = run.predictions.filter(pl.col("predictor") == "empirical").join(
        world.features[2025].select(*KEY, "prior_games_n", "market_prev"), on=KEY)
    assert (e["history_n"] == e["prior_games_n"]).all()
    one = e.filter(pl.col("history_n") == 1)
    assert one.height and (one["q10"] == one["market_prev"]).all() and (
        one["q90"] == one["market_prev"]).all()


# ---------------------------------------------------------------- protocol and leakage

def test_frozen_folds_and_training_population(run, p4a, world):  # noqa: F811
    assert set(run.predictions["fold_id"]) <= set(p4a.folds["fold_id"])
    used = run.predictions.select("fold_id", "decision_time").unique().join(
        p4a.folds.select("fold_id", "decision_time"), on=["fold_id", "decision_time"],
        how="anti")
    assert used.height == 0
    table = p4b.modelling_table(world, p4a)
    lab = pl.concat([world.labels[s] for s in (2024, 2025)]).select(*KEY, "played")
    j = table.join(lab, on=KEY)
    for r in run.fold_metadata.filter(pl.col("component") == "zero", pl.col("market") ==
                                      "receptions").iter_rows(named=True):
        tr = j.filter(pl.col("market") == "receptions", pl.col("played"),
                      ~pl.col("outcome_admitted"),
                      pl.col("outcome_available_at") <= r["decision_time"])
        assert r["train_rows"] == tr.height  # DNP and outcome-only rows excluded
        assert r["train_zero_rate"] == pytest.approx((tr["y"] == 0).mean())
    assert (run.fold_metadata["train_last_available_at"]
            <= run.fold_metadata["decision_time"]).all()


def test_target_fold_outcomes_cannot_change_distributions(run, world, p4a):  # noqa: F811
    last = p4a.folds.sort("fold_id").row(-1, named=True)
    wk = world.features[2025].filter(pl.col("decision_time") == last["decision_time"])[
        "week"].min()
    w2 = mutate_labels(world, pl.col("week") == wk, 333.0, scramble=True)
    assert p4.parquet_bytes(preds_of(w2, p4a)) == p4.parquet_bytes(run.predictions)


def test_future_outcomes_and_features_cannot_change_earlier_distributions(run, world, p4a):  # noqa: F811
    folds = p4a.folds.sort("fold_id")
    mid = folds.row(folds.height // 2, named=True)
    wk = world.features[2025].filter(pl.col("decision_time") == mid["decision_time"])[
        "week"].min()
    w2 = mutate_labels(world, pl.col("week") >= wk, 44.0)
    f = world.features[2025].with_columns(
        pl.when(pl.col("decision_time") > mid["decision_time"])
        .then(pl.col("target_share_roll5") * 0.1).otherwise(pl.col("target_share_roll5"))
        .alias("target_share_roll5"))
    w2 = replace(w2, features={**w2.features, 2025: f})
    p2 = preds_of(w2, p4a)
    upto = pl.col("decision_time") <= mid["decision_time"]
    assert p2.filter(upto).equals(run.predictions.filter(upto))
    assert not p2.filter(~upto).equals(run.predictions.filter(~upto))


def test_models_see_only_the_frozen_features(world, p4a):  # noqa: F811
    table = p4b.modelling_table(world, p4a)
    for t in p4b.build_tasks(table, p4a.folds, ("lightgbm",)):
        assert t.x.shape[1] == len(FEATURES) == 42


def test_prediction_artifact_contract(run):
    p = run.predictions
    assert not {"actual", "played", "label_status"} & set(p.columns)
    with pytest.raises(ContractViolation, match="label-only"):
        validate_frame(p.with_columns(pl.lit(1.0).alias("actual")), "distribution_predictions")
    with pytest.raises(ContractViolation, match="not ordered"):
        validate_frame(p.with_columns((pl.col("q90") - 1e6).alias("q90")),
                       "distribution_predictions")
    with pytest.raises(ContractViolation, match="p_zero"):
        validate_frame(p.with_columns(pl.lit(1.2).alias("p_zero")), "distribution_predictions")
    with pytest.raises(ContractViolation, match="duplicate"):
        validate_frame(pl.concat([p, p.head(1)]), "distribution_predictions")
    count = p.filter(pl.col("market").is_in(["pass_attempts", "rush_attempts", "receptions"]))
    assert (count["q10"] >= 0).all() and (count["mean_prediction"] >= 0).all()


def test_dnp_rows_are_predicted_but_not_scored(run, world):  # noqa: F811
    dnp = world.labels[2025].filter(~pl.col("played")).select(KEY)
    assert run.predictions.join(dnp, on=KEY).height == 2 * dnp.height
    assert run.scored.join(dnp, on=KEY).height == 0


def test_mean_vs_median_and_status_outputs(run):
    mm = run.mean_median
    assert set(mm["market"]) == {"pass_attempts", "rush_attempts", "rush_yds", "receptions",
                                 "reception_yds"}
    assert set(run.status["status"]) <= {p4c.PROMOTE, p4c.PROMISING, p4c.NO_IMPROVEMENT,
                                         p4c.REJECT}
    assert p4c.band(0.02) == "strong" and p4c.band(0.04) == "acceptable" and p4c.band(
        0.06) == "material"


def test_promotion_rule_rejects_integrity_failure(run):
    s = p4c.compare_and_classify(run.scored, run.metrics, run.interval_cal, (2025,),
                                 run.crossing, integrity_ok=False)
    assert set(s["status"]) == {p4c.REJECT}


def test_write_validate_cli_and_determinism(run, world, p4a, p4bb, tmp_path, capsys):  # noqa: F811
    a = p4c.write_run(run, world, p4a, p4bb, (2025,), (2024,), root=tmp_path)
    b = p4c.write_run(p4c.evaluate(world, p4a, p4bb, (2025,), workers=1), world, p4a, p4bb,
                      (2025,), (2024,), root=tmp_path)
    ma, mb = p4c.validate_run(a), p4c.validate_run(b)
    assert ma["files"] == mb["files"] and ma["content_sha256"] == mb["content_sha256"]
    assert p4.POINT_IN_TIME_WARNING in ma["warnings"] and p4c.RECONSTRUCTED_WARNING in ma[
        "warnings"]
    assert ma["distribution_set_version"] == "1.0.0" and "lightgbm" in ma["runtime"]
    assert distributions_cli.main(["validate", str(a)]) == 0
    assert "OK" in capsys.readouterr().out
    (a / "status.parquet").write_bytes(b"x")
    with pytest.raises(p4c.DistributionError, match="checksum"):
        p4c.validate_run(a)


def test_reproducible_across_processes(run, world, p4a):  # noqa: F811
    table = p4b.modelling_table(world, p4a)
    two = p4c.predict_distributions(table, p4a.folds, world, (2025,), workers=2)[0]
    assert p4.parquet_bytes(two) == p4.parquet_bytes(run.predictions)

