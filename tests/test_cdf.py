from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from lineseek.contracts import ContractViolation, forecast_time_utc, validate_frame
from lineseek.evaluation import cdf_cli
from lineseek.evaluation import phase4a as p4
from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c as p4c
from lineseek.evaluation import phase4c1 as p4c1
from lineseek.models.cdf import (
    CDF_PARAMS,
    CDF_SET_VERSION,
    CDFError,
    empirical_cdf,
    fit_predict_le,
    half_point,
    monotone_repair,
    pmf,
    quantile_approx_cdf,
    threshold_grid,
)
from lineseek.models.distribution import ZERO_PARAMS
from lineseek.models.learned import FEATURES
from tests.test_baselines import HURTS, SUN, appearances, hist_of
from tests.test_distributions import p4bb  # noqa: F401 (fixture)
from tests.test_models import mutate_labels, p4a, world  # noqa: F401 (fixtures)

KEY = ["game_id", "player_id", "market"]


@pytest.fixture(scope="module")
def p4cc(world, p4a, p4bb, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("p4c")
    path = p4c.write_run(p4c.evaluate(world, p4a, p4bb, (2025,), workers=1), world, p4a, p4bb,
                         (2025,), (2024,), root=root)
    return p4c1.load_phase4c(path, p4a)


@pytest.fixture(scope="module")
def run(world, p4a, p4cc):  # noqa: F811
    return p4c1.evaluate(world, p4a, p4cc, (2025,), workers=1)


def preds_of(w, p4a_, grids):
    table = p4b.modelling_table(w, p4a_)
    return p4c1.predict_cdf(table, p4a_.folds, w, grids, (2025,), workers=1)[0]


# ---------------------------------------------------------------- pure functions

def test_threshold_grid_rule_exact():
    y = np.array([0, 0, 0, 1, 2, 2, 3, 4, 5, 6, 8, 9, 12], dtype=float)
    g = threshold_grid("receptions", y)
    assert g[0] == 0 and g == sorted(set(g))
    assert set(range(2, 9)) <= set(g)  # dense central integers [floor Q25, floor Q90] of nonzero
    yards = np.array([-4, -2, 0, 0, 3, 7, 12, 18, 25, 31, 44, 60, 90], dtype=float)
    gy = threshold_grid("rush_yds", yards)
    assert -1 in gy and 0 in gy and min(gy) < 0  # negatives kept; -1 separates them from 0
    assert {10, 15, 20, 25, 30} <= set(gy)  # multiples of 5 in the central range


def test_monotone_repair_and_pmf():
    raw = np.array([[0.2, 0.1, 0.5, 0.45, 0.9], [0.1, 0.2, 0.3, 0.6, 0.8]])
    r = monotone_repair(raw)
    assert r["repaired"][0].tolist() == [0.1, 0.2, 0.45, 0.5, 0.9]
    assert r["violations"].tolist() == [2, 0] and r["repair_max"][1] == 0
    assert r["changed"][1].sum() == 0
    anchored = monotone_repair(np.array([[0.3, 0.2, 0.25, 0.6]]), anchor_first=True)
    assert anchored["repaired"][0].tolist() == [0.3, 0.3, 0.3, 0.6]  # CDF(0) never moves
    plain = monotone_repair(np.array([[0.3, 0.2, 0.25, 0.6]]))
    assert plain["repaired"][0].tolist() == [0.2, 0.25, 0.3, 0.6]
    m = pmf(r["repaired"])
    assert (m["interior"] >= 0).all()
    assert np.allclose(m["lower"] + m["interior"].sum(1) + m["upper"], 1.0)
    assert m["upper"][0] == pytest.approx(0.1) and m["lower"][1] == pytest.approx(0.1)


def test_half_point_helper():
    cdf = {0: 0.3, 1: 0.55, 2: 0.8, 4: 0.95}
    assert half_point(cdf, 1.5) == pytest.approx((0.55, 0.45))  # under 1.5 = P(Y <= 1)
    assert half_point(cdf, 0.5) == pytest.approx((0.3, 0.7))
    for bad in (3.5, 2.0, 7.5):  # off-grid or not a half point
        with pytest.raises(CDFError):
            half_point(cdf, bad)


def test_quantile_approx_cdf_exact():
    q = np.array([[0, 0, 0, 2, 6.0], [1, 2, 3, 4, 5.0], [1.03, 1.03, 1.03, 2, 4]])
    assert quantile_approx_cdf(q, 0).tolist() == pytest.approx([0.5, 0.05, 0.05])
    assert quantile_approx_cdf(q, 1).tolist() == pytest.approx([0.625, 0.1, 0.05])
    assert quantile_approx_cdf(q, 5).tolist() == pytest.approx([0.8625, 0.9, 0.95])
    assert quantile_approx_cdf(q, 7).tolist() == pytest.approx([0.95, 0.95, 0.95])


def test_threshold_classifiers_are_calibrated_on_a_synthetic_count():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8000, 2))
    y = rng.poisson(np.exp(0.5 + 0.6 * x[:, 0])).astype(float)
    xt = rng.normal(size=(5000, 2))
    yt = rng.poisson(np.exp(0.5 + 0.6 * xt[:, 0]))
    p = fit_predict_le([0, 1, 2, 4], x, y, xt)
    for j, k in enumerate((0, 1, 2, 4)):
        assert abs(p[:, j].mean() - (yt <= k).mean()) < 0.02
    assert ((0 <= p) & (p <= 1)).all()
    assert CDF_PARAMS == ZERO_PARAMS and CDF_SET_VERSION == "1.0.0"


def test_empirical_cdf_uses_only_eligible_history():
    app = appearances([("2025_01_DAL_PHI", 2025, SUN, 0), ("2025_02_PHI_KC", 2025,
                                                          SUN + timedelta(days=7), 3),
                       ("2025_03_LA_PHI", 2025, SUN + timedelta(days=14), 5),
                       ("2025_09_DAL_PHI", 2025, SUN + timedelta(days=56), 99)])  # target
    t = pl.DataFrame({"game_id": ["2025_09_DAL_PHI"], "player_id": [HURTS],
                      "season": pl.Series([2025], dtype=pl.Int32),
                      "decision_time": pl.Series([forecast_time_utc(SUN + timedelta(days=56))],
                                                 dtype=pl.Datetime("us", "UTC"))})
    grids = {m: [0, 3, 4, 50] for m in ("pass_attempts", "rush_attempts", "rush_yds",
                                        "receptions", "reception_yds")}
    e = empirical_cdf(t, hist_of(app), grids).filter(pl.col("market") == "receptions").sort(
        "threshold")
    assert e["p_le"].to_list() == pytest.approx([1 / 3, 2 / 3, 2 / 3, 1.0])  # 99 excluded
    assert set(e["history_n"]) == {3}


# ---------------------------------------------------------------- protocol on fixtures

def test_grid_depends_only_on_warmup(world, p4a, run):  # noqa: F811
    w2 = mutate_labels(world, pl.col("week") >= 1, 40.0)  # every 2025 outcome changed
    t2 = p4b.modelling_table(w2, p4a)
    assert p4c1.grids_from_warmup(t2, p4a.folds, (2025,)) == run.grids
    lab = world.labels[2024].with_columns(
        pl.when(pl.col("played")).then(pl.col("actual") + 7).otherwise(pl.col("actual"))
        .alias("actual"))
    t3 = p4b.modelling_table(replace(world, labels={**world.labels, 2024: lab}), p4a)
    assert p4c1.grids_from_warmup(t3, p4a.folds, (2025,)) != run.grids


def test_training_excludes_dnp_and_outcome_only(run, world, p4a):  # noqa: F811
    table = p4b.modelling_table(world, p4a)
    lab = pl.concat([world.labels[s] for s in (2024, 2025)]).select(*KEY, "played")
    j = table.join(lab, on=KEY)
    for r in run.fold_metadata.filter(pl.col("market") == "rush_yds").iter_rows(named=True):
        tr = j.filter(pl.col("market") == "rush_yds", pl.col("played"),
                      ~pl.col("outcome_admitted"),
                      pl.col("outcome_available_at") <= r["decision_time"])
        assert r["train_rows"] == tr.height
        assert r["train_rate"] == pytest.approx((tr["y"] <= r["threshold"]).mean())


def test_target_and_future_mutations(run, world, p4a):  # noqa: F811
    folds = p4a.folds.sort("fold_id")
    last, mid = folds.row(-1, named=True), folds.row(folds.height // 2, named=True)
    wk_last = world.features[2025].filter(pl.col("decision_time") == last["decision_time"])[
        "week"].min()
    p1 = preds_of(mutate_labels(world, pl.col("week") == wk_last, 222.0, scramble=True), p4a,
                  run.grids)
    assert p4.parquet_bytes(p1) == p4.parquet_bytes(run.predictions)
    wk_mid = world.features[2025].filter(pl.col("decision_time") == mid["decision_time"])[
        "week"].min()
    w2 = mutate_labels(world, pl.col("week") >= wk_mid, 31.0)
    f = world.features[2025].with_columns(
        pl.when(pl.col("decision_time") > mid["decision_time"]).then(pl.col("snap_pct_prev") / 9)
        .otherwise(pl.col("snap_pct_prev")).alias("snap_pct_prev"))
    p2 = preds_of(replace(w2, features={**w2.features, 2025: f}), p4a, run.grids)
    upto = pl.col("decision_time") <= mid["decision_time"]
    assert p2.filter(upto).equals(run.predictions.filter(upto))
    assert not p2.filter(~upto).equals(run.predictions.filter(~upto))


def test_models_use_only_frozen_features(world, p4a):  # noqa: F811
    for t in p4b.build_tasks(p4b.modelling_table(world, p4a), p4a.folds, ("lightgbm",)):
        assert t.x.shape[1] == len(FEATURES)


def test_count_cdf0_equals_phase4c_zero_model(run):
    z = run.zero.filter(pl.col("market").is_in(["pass_attempts", "rush_attempts", "receptions"]))
    assert (z["mean_abs_diff"] == 0).all() and (z["max_abs_diff"] == 0).all()


def test_artifact_contract_and_validity(run):
    p = run.predictions
    assert not {"actual", "played", "label_status"} & set(p.columns)
    with pytest.raises(ContractViolation, match="label-only"):
        validate_frame(p.with_columns(pl.lit(1.0).alias("actual")), "cdf_predictions")
    bad = p.with_columns(pl.when(pl.col("threshold") == pl.col("threshold").max().over(KEY))
                         .then(0.0).otherwise(pl.col("p_le")).alias("p_le"))
    with pytest.raises(ContractViolation, match="decreasing"):
        validate_frame(bad, "cdf_predictions")
    assert (run.pmf["negative_masses"] == 0).all()
    assert (run.pmf["max_total_mass_error"] < 1e-12).all()
    grid_n = {m: len(g) for m, g in run.grids.items()}
    per = p.filter(pl.col("predictor") == "cdf_lgbm").group_by(KEY).len()
    assert all(r["len"] == grid_n[r["market"]] for r in per.iter_rows(named=True))


def test_readiness_rule(run):
    s = p4c1.classify(run.scored, run.market_metrics, run.monotonic, run.zero, (2025,),
                      integrity_ok=False)
    assert set(s["status"]) == {p4c1.NOT_READY}
    assert p4c1.phase5_decision(pl.DataFrame({"status": [p4c1.READY] * 5})) == "GO"
    assert p4c1.phase5_decision(pl.DataFrame({"status": [p4c1.READY, p4c1.NOT_READY]})) == \
        "CONDITIONAL GO"
    assert p4c1.phase5_decision(pl.DataFrame({"status": [p4c1.CONDITIONAL]})) == "NO-GO"


def test_write_validate_cli_and_determinism(run, world, p4a, p4cc, tmp_path, capsys):  # noqa: F811
    a = p4c1.write_run(run, world, p4a, p4cc, (2025,), root=tmp_path)
    b = p4c1.write_run(p4c1.evaluate(world, p4a, p4cc, (2025,), workers=1), world, p4a, p4cc,
                       (2025,), root=tmp_path)
    ma, mb = p4c1.validate_run(a), p4c1.validate_run(b)
    assert ma["files"] == mb["files"] and ma["content_sha256"] == mb["content_sha256"]
    assert ma["cdf_set_version"] == "1.0.0" and p4.POINT_IN_TIME_WARNING in ma["warnings"]
    assert cdf_cli.main(["validate", str(a)]) == 0
    assert "OK" in capsys.readouterr().out
    (a / "status.parquet").write_bytes(b"x")
    with pytest.raises(CDFError, match="checksum"):
        p4c1.validate_run(a)


def test_reproducible_across_processes(run, world, p4a):  # noqa: F811
    table = p4b.modelling_table(world, p4a)
    two = p4c1.predict_cdf(table, p4a.folds, world, run.grids, (2025,), workers=2)[0]
    assert p4.parquet_bytes(two) == p4.parquet_bytes(run.predictions)


def test_quantile_consistency_aligns_rows_by_key(run, p4cc):
    # Independent recomputation by key (no positional alignment) must match the report.
    grid = run.grids["receptions"]
    p = run.predictions.filter(pl.col("predictor") == "cdf_lgbm", pl.col("market") ==
                               "receptions")
    implied = p.filter(pl.col("p_le") >= 0.5).group_by(KEY).agg(
        pl.col("threshold").min().alias("implied"))
    j = implied.join(p4cc.dist.select(*KEY, "q50"), on=KEY)
    want = (j["implied"] - j["q50"]).abs().mean()
    got = run.quantiles.filter(pl.col("market") == "receptions", pl.col("quantile") == "q50")
    assert got["mean_abs_diff"].item() == pytest.approx(want)
    assert got["rows"].item() - got["above_grid"].item() == j.height and grid
