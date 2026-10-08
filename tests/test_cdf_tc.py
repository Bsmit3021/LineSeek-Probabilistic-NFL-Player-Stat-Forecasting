import hashlib
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from lineseek.evaluation import cdf_tc_cli
from lineseek.evaluation import phase4a as p4
from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c1 as p4c1
from lineseek.evaluation import phase4c1b as p4c1b
from lineseek.models.cdf import CDF_PARAMS, CDFError, pmf
from lineseek.models.cdf_tc import (
    FP_TOL,
    TC_CDF_SET_VERSION,
    TC_FEATURES,
    TC_PARAMS,
    check_monotone,
    fit_predict_tc,
    stack,
)
from lineseek.models.learned import FEATURES, FORBIDDEN
from tests.test_cdf import p4cc  # noqa: F401 (fixture)
from tests.test_distributions import p4bb  # noqa: F401 (fixture)
from tests.test_models import mutate_labels, p4a, world  # noqa: F401 (fixtures)

KEY = ["game_id", "player_id", "market"]
SEASONS = (2025,)


@pytest.fixture(scope="module")
def p41(world, p4a, p4cc, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("p4c1")
    path = p4c1.write_run(p4c1.evaluate(world, p4a, p4cc, SEASONS, workers=1), world, p4a,
                          p4cc, SEASONS, root=root)
    return p4c1b.load_phase4c1(path, p4a, p4cc)


@pytest.fixture(scope="module")
def run(world, p4a, p4cc, p41):  # noqa: F811
    return p4c1b.evaluate(world, p4a, p4cc, p41, SEASONS, workers=1)


def preds_of(w, p4a_, grids):
    table = p4b.modelling_table(w, p4a_)
    return p4c1b.predict_tc(table, p4a_.folds, grids, SEASONS, workers=1)[0]


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    return x, rng.poisson(np.exp(0.8 + 0.7 * x[:, 0] - 0.3 * x[:, 1])).astype(float)


# ---------------------------------------------------------------- model

def test_one_new_input_and_constraint_only_on_threshold():
    assert TC_FEATURES == (*FEATURES, "threshold") and len(TC_FEATURES) == len(FEATURES) + 1
    assert TC_PARAMS["monotone_constraints"] == [0] * len(FEATURES) + [1]
    assert {k: v for k, v in TC_PARAMS.items() if not k.startswith("monotone")} == CDF_PARAMS
    assert TC_PARAMS["objective"] == "binary" and TC_PARAMS["n_jobs"] == 1
    assert TC_CDF_SET_VERSION == "2.0.0"
    x = np.arange(6, dtype=float).reshape(3, 2)
    s = stack(x, [0, 2, 5])
    assert s.shape == (9, 3) and s[:3, :2].tolist() == [[0, 1]] * 3
    assert s[:, -1].tolist() == [0, 2, 5] * 3


def test_threshold_conditioned_cdf_is_monotone_and_calibrated():
    x, y = synthetic(6000, 1)
    xt, yt = synthetic(4000, 2)
    grid = [0, 1, 2, 3, 5, 8]
    p = fit_predict_tc(grid, x, y, xt)
    assert p.shape == (4000, len(grid)) and ((0 <= p) & (p <= 1)).all()
    assert (np.diff(p, axis=1) >= 0).all()  # exact: no repair applied
    for j, k in enumerate(grid):
        assert abs(p[:, j].mean() - (yt <= k).mean()) < 0.03


def test_check_monotone_rejects_real_violations_and_cleans_fp_noise():
    ok = check_monotone(np.array([[0.1, 0.2, 0.2 - FP_TOL / 2, 0.9]]))
    assert ok["max_decrease"] <= FP_TOL and ok["violations"].tolist() == [1]
    assert (np.diff(ok["final"], axis=1) >= 0).all() and ok["changed"].sum() == 1
    with pytest.raises(CDFError, match="monotone constraint violated"):
        check_monotone(np.array([[0.1, 0.3, 0.29, 0.9]]))


def test_fit_refuses_unfrozen_inputs():
    x, y = synthetic(500, 3)
    with pytest.raises(CDFError):
        fit_predict_tc([0, 2, 1], x, y, x)  # grid must be ascending and unique
    with pytest.raises(CDFError):
        fit_predict_tc([0, 1], np.column_stack([x, y]), y, x)  # no extra (label) column


# ---------------------------------------------------------------- protocol on fixtures

def test_threshold_grid_is_frozen(run, p41, world, p4a):  # noqa: F811
    assert run.grids == p41.grids
    per = run.predictions.group_by("market").agg(pl.col("threshold").unique().sort())
    assert {r["market"]: r["threshold"] for r in per.iter_rows(named=True)} == run.grids
    table = p4b.modelling_table(world, p4a)
    bad = replace(p41, grids={**p41.grids, "receptions": [*p41.grids["receptions"], 999]})
    with pytest.raises(CDFError, match="frozen Phase 4C.1 grid"):
        p4c1b.frozen_grids(table, p4a.folds, SEASONS, bad)


def test_monotone_and_pmf_nonnegative(run):
    assert (run.monotonic["max_raw_decrease"] <= FP_TOL).all()
    p = run.predictions.sort(*KEY, "threshold")
    d = p.with_columns(pl.col("p_le").diff().over(KEY).alias("d"))["d"].drop_nulls()
    assert (d >= 0).all()
    assert (run.pmf["negative_masses"] == 0).all()
    assert (run.pmf["max_total_mass_error"] <= p4c1b.PMF_TOL).all()
    for market, g in run.grids.items():
        mat = p.filter(pl.col("market") == market)["p_le"].to_numpy().reshape(-1, len(g))
        assert (pmf(mat)["interior"] >= 0).all()


def test_training_excludes_dnp_and_outcome_only(run, world, p4a):  # noqa: F811
    table = p4b.modelling_table(world, p4a)
    lab = pl.concat([world.labels[s] for s in (2024, 2025)]).select(*KEY, "played")
    j = table.join(lab, on=KEY)
    for r in run.fold_metadata.filter(pl.col("market") == "rush_yds").iter_rows(named=True):
        tr = j.filter(pl.col("market") == "rush_yds", pl.col("played"),
                      ~pl.col("outcome_admitted"),
                      pl.col("outcome_available_at") <= r["decision_time"])
        assert r["train_rows"] == tr.height
        assert r["train_stacked_rows"] == tr.height * len(run.grids["rush_yds"])
        assert r["train_rate_k0"] == pytest.approx((tr["y"] <= 0).mean())
    assert (run.fold_metadata["train_last_available_at"]
            <= run.fold_metadata["decision_time"]).all()


def test_no_leakage_from_targets_or_the_future(run, world, p4a):  # noqa: F811
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


def test_qb_diagnostic_is_isolated(run, world, p4a, p4cc, p41):  # noqa: F811
    assert "qb_role" in FORBIDDEN and "qb_role" not in TC_FEATURES
    ros = world.label_raw[2025]["rosters_weekly"]
    flipped = ros.with_columns(pl.when(pl.col("position") == "QB").then(pl.lit("WR"))
                               .otherwise(pl.lit("QB")).alias("position"))
    w = replace(world, label_raw={**world.label_raw,
                                  2025: {**world.label_raw[2025], "rosters_weekly": flipped}})
    assert p4.parquet_bytes(preds_of(w, p4a, run.grids)) == p4.parquet_bytes(run.predictions)
    rows, rows2 = (p4c1b.scored_rows(x, p4a, SEASONS) for x in (world, w))
    assert rows.select(KEY, "actual").equals(rows2.select(KEY, "actual"))
    assert not rows["qb_role"].equals(rows2["qb_role"])
    markets = p4c1b.build_markets(run.predictions, p4c1b.phase4c1_predictions(p41), rows,
                                  run.grids, p4cc)
    for market, mk in markets.items():  # the QB cohort exists only for pass attempts
        names = [c for c, _ in p4c1b.cohorts(mk, market)]
        assert names == (["primary", "qb_role"] if market == "pass_attempts" else ["primary"])
    qb = run.threshold_metrics.filter(pl.col("cohort") == "qb_role")
    assert set(qb["market"]) <= {"pass_attempts"}
    n_qb = int(rows.filter(pl.col("market") == "pass_attempts")["qb_role"].sum())
    assert (qb["n"].filter(qb["period"] == "2025") == n_qb).all()


def test_comparators_on_identical_rows(run, p41, world, p4a, p4cc):  # noqa: F811
    rows = p4c1b.scored_rows(world, p4a, SEASONS)
    other = p4c1b.phase4c1_predictions(p41)
    assert set(other["predictor"]) == {p4c1b.INDEP, p4c1b.EMP}
    markets = p4c1b.build_markets(run.predictions, other, rows, run.grids, p4cc)
    for mk in markets.values():
        shapes = {v.shape for v in mk.P.values()}
        assert shapes == {mk.y.shape}
    tm = run.threshold_metrics.filter(pl.col("period") == "2025", pl.col("cohort") == "primary")
    n = tm.group_by("market", "threshold").agg(pl.col("n").n_unique().alias("u"))
    assert (n["u"] == 1).all()  # every predictor scored on the same rows
    with pytest.raises(CDFError, match="without a prediction"):
        p4c1b.build_markets(run.predictions, other.filter(pl.col("game_id") != rows[
            "game_id"][0]), rows, run.grids, p4cc)


def test_informative_rule_and_readiness(run):
    y = np.zeros((1000, 3))
    y[:300, 0] = 1  # 300 / 700, 30 %: informative
    y[:150, 1] = 1  # 150 positives: not informative
    y[:960, 2] = 1  # 96 %: not informative
    assert p4c1b.informative(y).tolist() == [True, False, False]
    s = p4c1b.classify(run.market_metrics, run.comparisons, run.persistent_bias, run.monotonic,
                       run.pmf, SEASONS, integrity_ok=False)
    assert set(s["status"]) == {p4c1b.NOT_READY}
    assert {"qb_mean_ece_ok", "qb_worst_threshold_ok", "qb_no_persistent_bias"} <= set(s.columns)
    assert p4c1b.phase5_decision(pl.DataFrame({"status": [p4c1b.READY] * 5})) == "GO"
    assert p4c1b.phase5_decision(pl.DataFrame({"status": [p4c1b.CONDITIONAL]})) == "NO-GO"


def test_deterministic_write_validate_and_workers(run, world, p4a, p4cc, p41, tmp_path,  # noqa: F811
                                                  capsys):
    a = p4c1b.write_run(run, world, p4a, p4cc, p41, SEASONS, root=tmp_path)
    b = p4c1b.write_run(p4c1b.evaluate(world, p4a, p4cc, p41, SEASONS, workers=2), world, p4a,
                        p4cc, p41, SEASONS, root=tmp_path)
    ma, mb = p4c1b.validate_run(a), p4c1b.validate_run(b)
    assert ma["files"] == mb["files"] and ma["content_sha256"] == mb["content_sha256"]
    assert ma["phase4c1"]["official_status"] == p41.manifest["summary"]["status"]
    assert cdf_tc_cli.main(["validate", str(a)]) == 0
    assert "OK" in capsys.readouterr().out
    (a / "status.parquet").write_bytes(b"x")
    with pytest.raises(CDFError, match="checksum"):
        p4c1b.validate_run(a)


def test_phase4c1_run_is_read_not_rewritten(p41, run):
    files = p41.manifest["files"]
    for k, meta in files.items():  # checked after the 4C.1b evaluation has read the run
        assert hashlib.sha256((p41.path / f"{k}.parquet").read_bytes()).hexdigest() == \
            meta["sha256"]
    assert sorted(f.name for f in p41.path.iterdir()) == sorted(
        [*(f"{k}.parquet" for k in files), "manifest.json"])
    assert run.predictions["predictor"].unique().to_list() == [p4c1b.TC]
