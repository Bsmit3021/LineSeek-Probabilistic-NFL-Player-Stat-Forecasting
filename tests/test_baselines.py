import hashlib
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import polars as pl
import pytest

from lineseek.contracts import (
    CONTRACT_VERSION,
    ContractViolation,
    forecast_time_utc,
    kickoff_utc,
    postgame_stats_available_at,
    validate_frame,
)
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.evaluation import cli
from lineseek.evaluation import phase4a as p4
from lineseek.evaluation.metrics import error_metrics, track_actuals
from lineseek.evaluation.universe import classify_universe
from lineseek.evaluation.walkforward import (
    FoldIntegrityError,
    split,
    walk_forward_folds,
)
from lineseek.features.player_game import (
    FEATURE_SET_VERSION,
    SeasonHistory,
    build_features,
    season_history,
)
from lineseek.grade.labels import LABEL_INPUTS, build_labels
from lineseek.ingest.sources import SOURCES
from lineseek.models.baselines import (
    BASELINE_SET_VERSION,
    BASELINES,
    EWMA_ALPHA,
    FEATURE_INPUTS,
    BaselineIntegrityError,
    ewma_history,
    predict_baselines,
    predict_feature_baselines,
)

FIX = Path(__file__).parent / "fixtures" / "features"
T0 = datetime(2026, 9, 28, 12, tzinfo=UTC)
HURTS, BARKLEY = "00-0036389", "00-0034844"
LAG = timedelta(hours=36)
STATS = ("attempts", "carries", "rushing_yards", "receptions", "receiving_yards", "targets")
MARKET_STAT = {"pass_attempts": "attempts", "rush_attempts": "carries",
               "rush_yds": "rushing_yards", "receptions": "receptions",
               "reception_yds": "receiving_yards"}


@cache
def raw(name: str, season: int) -> pl.DataFrame:
    return pl.read_parquet(FIX / f"{name}_{season}.parquet")


def frames(season: int, **raw_overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {n: SOURCES[n].build(raw_overrides.get(n, raw(n, season)), season, T0)
            for n in LABEL_INPUTS}


@pytest.fixture(scope="module")
def world():
    """In-memory Phase 4A inputs from the Phase 3 fixtures: warm-up 2024, evaluation 2025."""
    fr = {s: frames(s) for s in (2024, 2025)}
    hist = {s: season_history(**fr[s]) for s in fr}
    labels = {s: build_labels(**fr[s]).labels for s in fr}
    feats = {2024: build_features(2024, labels[2024], {2024: hist[2024]},
                                  fr[2024]["schedules"]).features,
             2025: build_features(2025, labels[2025], hist, fr[2025]["schedules"]).features}
    return p4.Inputs(feats, labels, {2025: fr}, fr)


@pytest.fixture(scope="module")
def run(world):
    return p4.evaluate(world, (2025,))


# ---------------------------------------------------------------- synthetic helpers

def feature_rows(**cols) -> pl.DataFrame:
    """Minimal feature rows (FEATURE_INPUTS) for one player; cols override per-row values."""
    n = len(next(iter(cols.values())))
    kick = [datetime(2025, 9, 7, 17, tzinfo=UTC) + timedelta(days=7 * i) for i in range(n)]
    base = {
        "game_id": [f"2025_{i + 1:02d}_DAL_PHI" for i in range(n)],
        "player_id": [HURTS] * n, "season": [2025] * n, "week": list(range(1, n + 1)),
        "team": ["PHI"] * n, "opponent": ["DAL"] * n, "market": ["rush_yds"] * n,
        "kickoff_utc": kick, "decision_time": [forecast_time_utc(k) for k in kick],
        "prior_games_n": [0] * n, "feature_set_version": [FEATURE_SET_VERSION] * n,
        "market_prev": [None] * n, "market_roll3_mean": [None] * n,
        "market_roll5_mean": [None] * n, "market_roll8_mean": [None] * n,
        "market_season_mean": [None] * n, "market_roll3_n": [0] * n, "market_roll5_n": [0] * n,
        "market_roll8_n": [0] * n, "market_season_games": [0] * n,
    }
    base.update(cols)
    schema = {**ARTIFACT_SCHEMAS["features"], **{c: pl.Float64 for c in FEATURE_INPUTS
                                                 if c.startswith("market_") and "_n" not in c
                                                 and not c.endswith("games")}}
    schema |= {c: pl.Int32 for c in ("prior_games_n", "market_roll3_n", "market_roll5_n",
                                     "market_roll8_n", "market_season_games", "season", "week")}
    return pl.DataFrame(base).with_columns(pl.col(c).cast(t) for c, t in schema.items()
                                           if c in base).select(FEATURE_INPUTS)


def appearances(rows: list[tuple[str, int, datetime, float]]) -> pl.DataFrame:
    """History rows (game_id, season, kickoff, value of every stat) for HURTS."""
    return pl.DataFrame({
        "game_id": [r[0] for r in rows], "player_id": [HURTS] * len(rows),
        "season": pl.Series([r[1] for r in rows], dtype=pl.Int32),
        "week": pl.Series([int(r[0][5:7]) for r in rows], dtype=pl.Int32),
        "kickoff_utc": pl.Series([r[2] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "available_at": pl.Series([r[2] + LAG for r in rows], dtype=pl.Datetime("us", "UTC")),
        **{c: pl.Series([r[3] for r in rows], dtype=pl.Int32) for c in STATS},
    })


def hist_of(app: pl.DataFrame) -> dict[int, SeasonHistory]:
    return {s: SeasonHistory(s, app.filter(pl.col("season") == s), pl.DataFrame(), (), ())
            for s in set(app["season"])}


def ewma_of(target_kick: datetime, app: pl.DataFrame, season: int = 2025) -> dict:
    t = pl.DataFrame({"game_id": [f"{season}_09_DAL_PHI"], "player_id": [HURTS],
                      "season": pl.Series([season], dtype=pl.Int32),
                      "decision_time": pl.Series([forecast_time_utc(target_kick)],
                                                 dtype=pl.Datetime("us", "UTC"))})
    h = hist_of(app)
    h.setdefault(season, SeasonHistory(season, app.clear(), pl.DataFrame(), (), ()))
    out = ewma_history(t, h).filter(pl.col("market") == "rush_yds")
    return out.row(0, named=True) if out.height else {"ewma": None, "_n": 0}


SUN = datetime(2025, 9, 7, 17, tzinfo=UTC)  # 2025 week 1 Sunday 13:00 ET


# ---------------------------------------------------------------- baselines: exact values

def test_feature_baselines_are_exact_feature_columns():
    f = feature_rows(prior_games_n=[0, 1, 4, 9],
                     market_prev=[None, 12.0, -3.0, 40.0],
                     market_roll3_mean=[None, 12.0, 5.0, 31.0],
                     market_roll5_mean=[None, 12.0, 4.5, 28.4],
                     market_roll8_mean=[None, 12.0, 4.5, 27.125],
                     market_season_mean=[None, 12.0, 4.5, 30.0],
                     market_roll3_n=[0, 1, 3, 3], market_roll5_n=[0, 1, 4, 5],
                     market_roll8_n=[0, 1, 4, 8], market_season_games=[0, 1, 4, 2])
    p = predict_feature_baselines(f)
    got = {(r["week"], r["baseline"]): (r["prediction"], r["history_n"])
           for r in p.iter_rows(named=True)}
    assert got[(2, "prev_game")] == (12.0, 1) and got[(3, "prev_game")] == (-3.0, 1)
    assert got[(4, "prev_game")] == (40.0, 1)
    assert got[(4, "roll3_mean")] == (31.0, 3) and got[(3, "roll3_mean")] == (5.0, 3)
    assert got[(4, "roll5_mean")] == (28.4, 5) and got[(3, "roll5_mean")] == (4.5, 4)
    assert got[(4, "roll8_mean")] == (27.125, 8)
    assert got[(4, "season_mean")] == (30.0, 2)
    assert all(got[(1, b)] == (None, 0) for b in BASELINES if b != "ewma_span5")


def test_ewma_hand_calculation():
    # alpha = 1/3; values oldest-first 10, 4, 7: 10 -> 8 -> 23/3.
    app = appearances([("2025_01_DAL_PHI", 2025, SUN, 10),
                       ("2025_02_PHI_KC", 2025, SUN + timedelta(days=7), 4),
                       ("2025_03_LA_PHI", 2025, SUN + timedelta(days=14), 7)])
    r = ewma_of(SUN + timedelta(days=21), app)
    assert EWMA_ALPHA == 1 / 3
    assert r["ewma"] == pytest.approx(23 / 3, abs=1e-12) and r["_n"] == 3 and r["last"] == 7


def test_ewma_one_game_history_is_that_game():
    r = ewma_of(SUN + timedelta(days=7), appearances([("2025_01_DAL_PHI", 2025, SUN, -6)]))
    assert r["ewma"] == -6 and r["_n"] == 1


def test_ewma_crosses_one_season_only():
    old = SUN - timedelta(days=730)
    prev = SUN - timedelta(days=210)
    app = appearances([("2023_10_DAL_PHI", 2023, old, 500),
                       ("2024_20_LA_PHI", 2024, prev, 30),
                       ("2025_01_DAL_PHI", 2025, SUN, 0)])
    r = ewma_of(SUN + timedelta(days=7), app)
    assert r["_n"] == 2 and r["ewma"] == pytest.approx(30 * 2 / 3)  # 30 -> 1/3*0 + 2/3*30


def test_ewma_excludes_target_and_future_games():
    target = SUN + timedelta(days=14)
    hist = [("2025_01_DAL_PHI", 2025, SUN, 10), ("2025_02_PHI_KC", 2025, SUN + timedelta(days=7), 4)]
    base = ewma_of(target, appearances(hist))
    huge = ewma_of(target, appearances([*hist, ("2025_03_LA_PHI", 2025, target, 999),
                                        ("2025_04_PHI_TB", 2025, target + timedelta(days=7), 999)]))
    assert base == huge and base["ewma"] == pytest.approx(8.0)


def test_ewma_excludes_games_not_available_by_the_lock():
    # Tuesday game (36 h lag ends Thursday) before a Wednesday lock is excluded.
    tue = datetime(2025, 9, 16, 23, tzinfo=UTC)
    target = datetime(2025, 9, 21, 17, tzinfo=UTC)
    r = ewma_of(target, appearances([("2025_01_DAL_PHI", 2025, SUN, 10),
                                     ("2025_02_PHI_KC", 2025, tue, 100)]))
    assert r["_n"] == 1 and r["ewma"] == 10


def test_no_history_gives_null_prediction(world):
    p = predict_baselines(world.features[2025], p4.histories_for(world.feature_raw)[2025])
    cold = p.filter(pl.col("prior_games_n") == 0)
    assert cold.height and cold["prediction"].is_null().all()
    assert (cold["history_n"] == 0).all()


def test_ewma_matches_independent_plain_python(run):
    k = {g: kickoff_utc(d, t) for s in (2024, 2025) for g, d, t in
         raw("schedules", s).select("game_id", "gameday", "gametime").iter_rows()}
    ps = pl.concat([raw("player_stats", s) for s in (2024, 2025)], how="diagonal_relaxed")
    p = run.predictions.filter(pl.col("baseline") == "ewma_span5",
                               pl.col("player_id").is_in([HURTS, BARKLEY]))
    for r in p.iter_rows(named=True):
        mine = ps.filter(pl.col("player_id") == r["player_id"])
        lock = forecast_time_utc(k[r["game_id"]])
        vals = [v for g, v in sorted(mine.select("game_id", MARKET_STAT[r["market"]]).iter_rows(),
                                     key=lambda x: k[x[0]])
                if k[g] + LAG <= lock and int(g[:4]) >= 2024]
        e = None
        for v in vals:
            e = v if e is None else EWMA_ALPHA * v + (1 - EWMA_ALPHA) * e
        assert r["prediction"] == pytest.approx(e, abs=1e-9), r


def test_predictions_equal_feature_columns_on_fixture(run, world):
    f = world.features[2025]
    p = run.predictions
    for b, col in (("prev_game", "market_prev"), ("roll5_mean", "market_roll5_mean"),
                   ("season_mean", "market_season_mean")):
        j = p.filter(pl.col("baseline") == b).join(f, on=["game_id", "player_id", "market"])
        assert j["prediction"].equals(j[col], check_names=False)


def test_ewma_rejects_histories_that_disagree_with_features(world):
    h = p4.histories_for(world.feature_raw)[2025]
    f = world.features[2025].with_columns(pl.col("prior_games_n") + 1)
    with pytest.raises(BaselineIntegrityError, match="disagrees"):
        predict_baselines(f, h)


def test_predictions_are_deterministic_unique_and_frozen(run, world):
    again = p4.evaluate(world, (2025,))
    assert p4.parquet_bytes(again.predictions) == p4.parquet_bytes(run.predictions)
    assert run.predictions.height == run.predictions.select(
        "game_id", "player_id", "market", "decision_time", "baseline").unique().height
    with pytest.raises(ContractViolation, match="duplicate"):
        validate_frame(pl.concat([run.predictions, run.predictions.head(1)]),
                       "baseline_predictions")
    assert BASELINES == ("prev_game", "roll3_mean", "roll5_mean", "roll8_mean", "season_mean",
                         "ewma_span5")
    assert BASELINE_SET_VERSION == "1.0.0" and CONTRACT_VERSION == "0.11.0"
    assert set(run.predictions["baseline_set_version"]) == {BASELINE_SET_VERSION}


def test_prediction_ignores_label_columns(world):
    h = p4.histories_for(world.feature_raw)[2025]
    f = world.features[2025]
    labelled = f.join(world.labels[2025].select("game_id", "player_id", "market", "actual",
                                                "played", "label_status"),
                      on=["game_id", "player_id", "market"])
    scrambled = labelled.with_columns(pl.lit(1e6).alias("actual"), ~pl.col("played"),
                                      pl.lit("recorded").alias("label_status"))
    a, b = predict_baselines(f, h), predict_baselines(scrambled, h)
    assert p4.parquet_bytes(a) == p4.parquet_bytes(b)


# ---------------------------------------------------------------- walk-forward folds

def et(y, mo, d, h, mi=0) -> datetime:
    from zoneinfo import ZoneInfo
    return datetime(y, mo, d, h, mi, tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)


def fold_rows(games: list[tuple[str, datetime]], admitted: set[str] = frozenset()
              ) -> pl.DataFrame:
    rows = [{"game_id": g, "player_id": HURTS, "market": "rush_yds", "season": int(g[:4]),
             "week": int(g[5:7]), "kickoff_utc": k, "decision_time": forecast_time_utc(k),
             "outcome_admitted": g in admitted} for g, k in games]
    return pl.DataFrame(rows).with_columns(pl.col("season", "week").cast(pl.Int32),
                                           pl.col("kickoff_utc", "decision_time")
                                           .cast(pl.Datetime("us", "UTC")))


GAMES = [
    ("2020_17_DAL_PHI", et(2021, 1, 3, 13)),
    ("2021_01_DAL_PHI", et(2021, 9, 12, 13)),
    ("2021_02_PHI_KC", et(2021, 9, 16, 20, 20)),   # Thursday
    ("2021_03_LA_PHI", et(2021, 9, 26, 13)),
    ("2021_04_PHI_TB", et(2021, 10, 5, 19)),       # Tuesday reschedule
    ("2021_05_NYG_PHI", et(2021, 10, 10, 13)),     # lock Wed 10-06 12:00 < Tue + 36 h
    ("2021_06_PHI_NYG", et(2021, 10, 17, 13)),
]


def test_folds_are_chronological_and_train_only_on_available_outcomes():
    rows = fold_rows(GAMES, admitted={"2021_03_LA_PHI"})
    folds = walk_forward_folds(rows, (2021,))
    assert folds["decision_time"].is_sorted() and folds["fold_id"].to_list() == list(range(6))
    for f in folds.iter_rows(named=True):
        s = split(rows, f)
        assert (s.train["kickoff_utc"] + LAG <= f["decision_time"]).all()
        assert (s.test["decision_time"] == f["decision_time"]).all()
        assert not set(s.train["game_id"]) & set(s.test["game_id"])
        assert (s.train["kickoff_utc"] < s.test["kickoff_utc"].min()).all()
        assert s.train.height == f["train_rows"]
    assert folds.filter(pl.col("fold_id") == 2)["test_rows_outcome_admitted"].item() == 1
    assert folds.filter(pl.col("fold_id") == 2)["test_rows_primary"].item() == 0


def test_tuesday_game_is_not_trainable_at_the_next_wednesday_lock():
    rows = fold_rows(GAMES)
    folds = walk_forward_folds(rows, (2021,))
    nyg = folds.filter(pl.col("decision_time") == forecast_time_utc(et(2021, 10, 10, 13))).row(
        0, named=True)
    s = split(rows, nyg)
    assert "2021_04_PHI_TB" not in set(s.train["game_id"])
    assert nyg["started_unavailable_rows"] == 1
    later = split(rows, folds.filter(pl.col("fold_id") == nyg["fold_id"] + 1).row(0, named=True))
    assert "2021_04_PHI_TB" in set(later.train["game_id"])


def test_availability_boundary_is_inclusive():
    lock = forecast_time_utc(et(2021, 9, 26, 13))
    at = [("2021_01_DAL_PHI", lock - LAG), ("2021_03_LA_PHI", et(2021, 9, 26, 13))]
    late = [("2021_01_DAL_PHI", lock - LAG + timedelta(seconds=1)),
            ("2021_03_LA_PHI", et(2021, 9, 26, 13))]
    assert postgame_stats_available_at(lock - LAG) == lock
    f_at = walk_forward_folds(fold_rows(at), (2021,)).filter(pl.col("decision_time") == lock)
    f_late = walk_forward_folds(fold_rows(late), (2021,)).filter(pl.col("decision_time") == lock)
    assert f_at["train_rows"].item() == 1 and f_late["train_rows"].item() == 0


def test_folds_do_not_depend_on_row_order_or_future_seasons():
    rows = fold_rows(GAMES)
    base = walk_forward_folds(rows, (2021,))
    assert walk_forward_folds(rows.sample(fraction=1, shuffle=True, seed=3), (2021,)).equals(base)
    future = fold_rows([("2022_01_DAL_PHI", et(2022, 9, 11, 13)),
                        ("2022_02_PHI_KC", et(2022, 9, 18, 13))])
    more = walk_forward_folds(pl.concat([rows, future]), (2021, 2022))
    assert more.head(base.height).equals(base) and more.height == base.height + 2


def test_split_rejects_a_contaminated_fold():
    # A test row whose decision_time is not before its kickoff must never be scored.
    rows = fold_rows(GAMES)
    late = rows["kickoff_utc"].max()
    bad = rows.with_columns(pl.lit(late).alias("decision_time"))
    with pytest.raises(FoldIntegrityError, match="test kickoff"):
        split(bad, {"fold_id": 0, "decision_time": late})


def test_fold_contract_rejects_training_after_decision_time(run):
    bad = run.folds.with_columns((pl.col("decision_time") + timedelta(seconds=1))
                                 .alias("train_last_available_at"))
    with pytest.raises(ContractViolation, match="training outcome available after"):
        validate_frame(bad, "walk_forward_folds")


# ---------------------------------------------------------------- universe (F1)

def outcome_only_players() -> set[str]:
    ps = raw("player_stats", 2025)
    return set(ps.filter(pl.col("player_name").is_in(["J.Karty", "B.Mann", "G.Stuard"]))
               ["player_id"])


def test_outcome_admitted_rows_are_identified(run):
    c = run.cohorts.filter(pl.col("season") == 2025)
    oa = c.filter(pl.col("outcome_admitted"))
    assert set(oa["player_id"]) == outcome_only_players() and oa.height == 3
    assert set(c["admission_path"]) <= {"roster", "participation", "outcome_only"}


def test_primary_metrics_exclude_outcome_admitted_rows_diagnostic_keeps_them(run):
    m = run.metrics.filter(pl.col("track") == "forecast_lock", pl.col("view") == "native",
                           pl.col("period") == "2025", pl.col("history_bucket") == "all",
                           pl.col("baseline") == "prev_game")
    rows = dict(m.group_by("cohort").agg(pl.col("n_rows").sum()).iter_rows())
    assert rows["full_label_universe"] - rows["primary"] == 3 * 5
    s = run.scored.filter(pl.col("baseline") == "prev_game")
    assert rows["primary"] == s.filter(~pl.col("outcome_admitted")).height
    assert set(run.metrics["cohort"]) == {"primary", "full_label_universe"}
    roster = s.filter(pl.col("admission_path") == "roster").height
    assert roster and rows["primary"] >= roster


def test_universe_classification_cannot_change_predictions(run, world):
    c2 = run.cohorts.with_columns(~pl.col("outcome_admitted"))
    s2 = p4.score(run.predictions, {2025: world.labels[2025]}, c2, run.folds)
    m2 = p4.compute_metrics(s2, (2025,))
    assert s2["prediction"].equals(run.scored["prediction"])
    assert not m2.equals(run.metrics)


def test_classifier_marks_injected_outcome_only_activity():
    # An offensive lineman with snaps and a fake catch is admitted only by that catch.
    ps = raw("player_stats", 2025)
    ol = ps.filter(pl.col("position_group") == "OL").row(0, named=True)
    ps2 = ps.with_columns(pl.when((pl.col("player_id") == ol["player_id"])
                                  & (pl.col("game_id") == ol["game_id"]))
                          .then(1).otherwise(pl.col("receptions")).alias("receptions"))
    fr = frames(2025, player_stats=ps2)
    keys = build_labels(**fr).labels
    c = classify_universe(keys, **fr)
    hit = c.filter(pl.col("player_id") == ol["player_id"], pl.col("game_id") == ol["game_id"])
    assert hit["admission_path"].to_list() == ["outcome_only"]


# ---------------------------------------------------------------- participation tracks

def labels_tiny() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": ["g"] * 4, "player_id": ["a", "b", "c", "d"], "market": ["receptions"] * 4,
        "actual": [0.0, 5.0, 0.0, None], "played": [True, True, True, False],
        "label_status": ["recorded", "recorded", "participant_no_stats_row", "did_not_play"],
    })


def test_track_actuals_follow_the_grading_contract():
    t = track_actuals(labels_tiny()).sort("player_id")
    assert t["actual_participant"].to_list() == [0.0, 5.0, 0.0, None]
    assert t["actual_forecast_lock"].to_list() == [0.0, 5.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="without an actual"):
        track_actuals(labels_tiny().with_columns(pl.lit(True).alias("played")))


def test_participant_track_excludes_dnp_and_keeps_real_zeros():
    t = track_actuals(labels_tiny()).with_columns(pl.lit(2.0).alias("prediction"),
                                                  pl.lit("x").alias("k"))
    part = error_metrics(t, ["k"], "actual_participant").row(0, named=True)
    lock = error_metrics(t, ["k"], "actual_forecast_lock").row(0, named=True)
    assert (part["n_rows"], part["n_eligible"], part["n_predicted"]) == (4, 3, 3)
    assert part["mae"] == pytest.approx((2 + 3 + 2) / 3)
    assert (lock["n_eligible"], lock["n_predicted"]) == (4, 4)
    assert lock["mae"] == pytest.approx((2 + 3 + 2 + 2) / 4)


# ---------------------------------------------------------------- metrics

def test_metrics_hand_computed():
    df = pl.DataFrame({"k": ["x"] * 4, "prediction": [2.0, 4.0, None, 10.0],
                       "actual": [3.0, 4.0, 5.0, 6.0]})
    m = error_metrics(df, ["k"], "actual").row(0, named=True)
    # errors: -1, 0, (unpredicted), +4; |e| sorted 0, 1, 4
    assert (m["n_eligible"], m["n_predicted"], m["coverage_rate"]) == (4, 3, 0.75)
    assert m["mae"] == pytest.approx(5 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(17 / 3))
    assert m["median_ae"] == 1.0
    assert m["mean_error"] == pytest.approx(1.0)  # positive = overprediction
    assert m["p90_ae"] == pytest.approx(1 + 0.8 * (4 - 1))  # linear: position 0.9 * 2 = 1.8
    assert m["max_ae"] == 4.0


def test_native_and_common_support_views(run):
    m = run.metrics.filter(pl.col("cohort") == "primary", pl.col("track") == "participant",
                           pl.col("period") == "2025", pl.col("history_bucket") == "all",
                           pl.col("market") == "receptions")
    nat = dict(m.filter(pl.col("view") == "native").select("baseline", "n_predicted").iter_rows())
    com = dict(m.filter(pl.col("view") == "common_support").select("baseline", "n_predicted")
               .iter_rows())
    assert len(set(com.values())) == 1  # same rows for every baseline
    assert nat["season_mean"] < nat["prev_game"] and com["prev_game"] == nat["season_mean"]
    s = run.scored.filter(~pl.col("outcome_admitted"), pl.col("played"),
                          pl.col("market") == "receptions", pl.col("common_support"),
                          pl.col("baseline") == "roll3_mean")
    mae = (s["prediction"] - s["actual_participant"]).abs().mean()
    got = m.filter(pl.col("view") == "common_support", pl.col("baseline") == "roll3_mean")
    assert got["mae"].item() == pytest.approx(mae)


def test_cold_start_buckets_cover_every_row(run):
    m = run.metrics.filter(pl.col("cohort") == "primary", pl.col("track") == "forecast_lock",
                           pl.col("view") == "native", pl.col("baseline") == "prev_game",
                           pl.col("market") == "rush_yds")
    total = m.filter(pl.col("history_bucket") == "all", pl.col("period") == "2025")["n_rows"].item()
    buckets = m.filter(pl.col("history_bucket") != "all")
    assert buckets["n_rows"].sum() == total
    zero = buckets.filter(pl.col("history_bucket") == "0").row(0, named=True)
    assert zero["n_predicted"] == 0 and zero["coverage_rate"] == 0


def test_comparison_and_benchmarks(run):
    c = run.comparison.filter(pl.col("baseline") == "prev_game")
    assert (c["mae_diff_vs_prev"] == 0).all()
    ci = run.comparison.filter(pl.col("view") == "common_support", pl.col("cohort") == "primary",
                               pl.col("baseline") != "prev_game")
    assert ci["mae_diff_ci_low"].is_not_null().all()
    assert (ci["mae_diff_ci_low"] <= ci["mae_diff_vs_prev"]).all()
    assert (ci["mae_diff_vs_prev"] <= ci["mae_diff_ci_high"]).all()
    b = run.benchmarks
    assert b.height == 2 * 5 * 4 and set(b["baseline"]) <= set(BASELINES)


# ---------------------------------------------------------------- mutations

def _mutate_labels(world, cond, value: float):
    lab = world.labels[2025].with_columns(
        pl.when(cond & pl.col("actual").is_not_null()).then(pl.lit(value))
        .otherwise(pl.col("actual")).alias("actual"))
    return replace(world, labels={**world.labels, 2025: lab})


def test_target_outcome_mutation_changes_metrics_not_predictions(run, world):
    w8 = pl.col("week") == 8
    m = p4.evaluate(_mutate_labels(world, w8, 900.0), (2025,))
    assert p4.parquet_bytes(m.predictions) == p4.parquet_bytes(run.predictions)
    assert m.folds.equals(run.folds)
    assert not m.metrics.equals(run.metrics)


def test_future_outcome_mutation_cannot_change_folds_or_earlier_predictions(run, world):
    m = p4.evaluate(_mutate_labels(world, pl.col("week") >= 12, -50.0), (2025,))
    assert m.folds.equals(run.folds)
    assert p4.parquet_bytes(m.predictions) == p4.parquet_bytes(run.predictions)
    assert not m.metrics.equals(run.metrics)  # truth changed, so the scores may


def test_future_feature_mutation_cannot_change_earlier_predictions(run, world):
    f = world.features[2025].with_columns(
        pl.when(pl.col("week") >= 12).then(pl.col("market_roll5_mean") + 100)
        .otherwise(pl.col("market_roll5_mean")).alias("market_roll5_mean"))
    m = p4.evaluate(replace(world, features={**world.features, 2025: f}), (2025,))
    early = pl.col("week") < 12
    assert m.predictions.filter(early).equals(run.predictions.filter(early))
    assert not m.predictions.equals(run.predictions)


def test_scrambled_labels_leave_predictions_byte_identical(run, world):
    lab = world.labels[2025].with_columns(
        (pl.col("actual").fill_null(3) * -9 + 77).alias("actual"), ~pl.col("played"),
        pl.lit("did_not_play").alias("label_status"))
    m = p4.evaluate(replace(world, labels={**world.labels, 2025: lab}), (2025,))
    assert m.predictions_sha256 == run.predictions_sha256
    assert hashlib.sha256(p4.parquet_bytes(m.predictions)).hexdigest() == run.predictions_sha256


# ---------------------------------------------------------------- storage, CLI, determinism

def test_write_validate_and_determinism(run, world, tmp_path, capsys):
    a = p4.write_run(run, world, (2025,), (2024,), root=tmp_path)
    b = p4.write_run(p4.evaluate(world, (2025,)), world, (2025,), (2024,), root=tmp_path)
    ma, mb = p4.validate_run(a), p4.validate_run(b)
    assert ma["files"] == mb["files"] and a != b
    assert ma["warnings"][0] == p4.POINT_IN_TIME_WARNING
    assert ma["baseline_set_version"] == BASELINE_SET_VERSION
    assert ma["feature_set_version"] == FEATURE_SET_VERSION
    assert {"polars", "pyarrow", "numpy", "python"} <= set(ma["runtime"])
    assert cli.main(["validate", str(a)]) == 0
    assert "OK" in capsys.readouterr().out
    (a / "folds.parquet").write_bytes((a / "folds.parquet").read_bytes() + b"x")
    with pytest.raises(p4.EvaluationIntegrityError, match="checksum"):
        p4.validate_run(a)


def test_summary_counts_reconcile(run, world):
    s = p4.summarize(run)
    lab = world.labels[2025]
    assert s["prediction_rows"] == 6 * lab.height
    assert s["did_not_play_rows"] == lab.filter(pl.col("label_status") == "did_not_play").height
    assert s["primary_rows"] + s["outcome_admitted_rows"] == lab.height
    assert s["temporal_violations"] == 0
