import json
import statistics
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import polars as pl
import pytest

from lineseek.contracts import (
    CONTRACT_VERSION,
    PROP_MARKETS,
    ContractViolation,
    forecast_time_utc,
    kickoff_utc,
    postgame_stats_available_at,
    validate_frame,
)
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.features import cli
from lineseek.features.player_game import (
    FEATURE_COLUMNS,
    FEATURE_SET_VERSION,
    FeatureIntegrityError,
    build_features,
    history_seasons,
    season_history,
    validate_features,
    write_features,
)
from lineseek.grade.labels import LABEL_INPUTS, build_labels
from lineseek.ingest.sources import SOURCES

FIX = Path(__file__).parent / "fixtures" / "features"
T0 = datetime(2026, 9, 28, 12, tzinfo=UTC)
HURTS, BARKLEY = "00-0036389", "00-0034844"
LAG = timedelta(hours=36)


@cache
def raw(name: str, season: int) -> pl.DataFrame:
    return pl.read_parquet(FIX / f"{name}_{season}.parquet")


def frames(season: int, **raw_overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Snapshot-shaped inputs (conformed + provenance), optionally from modified raw frames."""
    return {n: SOURCES[n].build(raw_overrides.get(n, raw(n, season)), season, T0)
            for n in LABEL_INPUTS}


@pytest.fixture(scope="module")
def histories():
    return {s: season_history(**frames(s)) for s in (2024, 2025)}


@pytest.fixture(scope="module")
def keys():
    return build_labels(**frames(2025)).labels


@pytest.fixture(scope="module")
def build(histories, keys):
    return build_features(2025, keys, histories, frames(2025)["schedules"])


@pytest.fixture(scope="module")
def feats(build) -> pl.DataFrame:
    return build.features


def rebuild(keys, **raw_2025: pl.DataFrame) -> pl.DataFrame:
    hist = {2024: season_history(**frames(2024)), 2025: season_history(**frames(2025, **raw_2025))}
    return build_features(2025, keys, hist, frames(2025)["schedules"]).features


def row(df: pl.DataFrame, player_id: str, week: int, market: str) -> dict:
    return df.filter(pl.col("player_id") == player_id, pl.col("week") == week,
                     pl.col("market") == market).row(0, named=True)


# Independent reference computed from raw fixture rows with plain Python.

@cache
def kickoffs() -> dict[str, datetime]:
    sched = pl.concat([raw("schedules", s) for s in (2024, 2025)])
    return {g: kickoff_utc(d, t) for g, d, t in sched.select("game_id", "gameday", "gametime")
            .iter_rows()}


def game_of(team: str, week: int, season: int = 2025) -> str:
    return raw("schedules", season).filter(
        pl.col("week") == week, (pl.col("home_team") == team) | (pl.col("away_team") == team)
    )["game_id"].item()


def eligible(games: list[tuple[str, int]], target: str, season: int) -> list[str]:
    """Game ids available at the target's decision time, most recent first."""
    lock = forecast_time_utc(kickoffs()[target])
    ok = [g for g, s in games if kickoffs()[g] + LAG <= lock and season - 1 <= s <= season]
    return sorted(ok, key=lambda g: kickoffs()[g], reverse=True)


def stats_rows() -> pl.DataFrame:
    return pl.concat([raw("player_stats", s) for s in (2024, 2025)], how="diagonal_relaxed")


def ref_player(player_id: str, target: str, column: str) -> dict:
    ps = stats_rows().filter(pl.col("player_id") == player_id)
    value = dict(ps.select("game_id", column).iter_rows())
    season_of = dict(ps.select("game_id", "season").iter_rows())
    games = eligible(list(season_of.items()), target, 2025)
    vals = [value[g] for g in games]
    this = [value[g] for g in games if season_of[g] == 2025]
    out = {"prior_games_n": len(vals), "market_prev": vals[0] if vals else None,
           "market_season_games": len(this),
           "market_season_mean": statistics.mean(this) if this else None,
           "market_season_median": statistics.median(this) if this else None,
           "market_season_std": statistics.stdev(this) if len(this) > 1 else None}
    for k in (3, 5, 8):
        out[f"market_roll{k}_mean"] = statistics.mean(vals[:k]) if vals else None
        out[f"market_roll{k}_n"] = min(k, len(vals))
    return out


def team_totals() -> pl.DataFrame:
    return stats_rows().group_by("game_id", "team").agg(
        pl.col(c).sum() for c in ("attempts", "carries", "rushing_yards", "receptions",
                                  "receiving_yards", "targets"))


def assert_close(got: dict, want: dict) -> None:
    for k, v in want.items():
        assert got[k] == pytest.approx(v), k


# Exact values.

@pytest.mark.parametrize("week", [1, 2, 4, 8, 10, 12])
@pytest.mark.parametrize("market", list(PROP_MARKETS))
def test_player_windows_match_reference(feats, week, market):
    target = game_of("PHI", week)
    for pid in (HURTS, BARKLEY):
        assert_close(row(feats, pid, week, market),
                     ref_player(pid, target, PROP_MARKETS[market].stat_column))


def test_rolling_windows_exact_values(feats):
    # Hurts before 2025 wk8: 2025 wk7..wk1, then 2024 SB (cross-season fill of roll8).
    r = row(feats, HURTS, 8, "pass_attempts")
    ps = stats_rows().filter(pl.col("player_id") == HURTS)
    att = dict(ps.select("game_id", "attempts").iter_rows())
    recent = [att[game_of("PHI", w)] for w in (7, 6, 5, 4, 3, 2, 1)] + [att["2024_22_KC_PHI"]]
    assert r["market_prev"] == recent[0]
    assert r["market_roll3_mean"] == pytest.approx(sum(recent[:3]) / 3)
    assert r["market_roll5_mean"] == pytest.approx(sum(recent[:5]) / 5)
    assert r["market_roll8_mean"] == pytest.approx(sum(recent) / 8)
    assert (r["market_roll3_n"], r["market_roll5_n"], r["market_roll8_n"]) == (3, 5, 8)
    assert r["market_season_games"] == 7
    assert r["market_season_mean"] == pytest.approx(sum(recent[:7]) / 7)


def test_season_to_date_resets_but_rolling_crosses_seasons(feats):
    r = row(feats, HURTS, 1, "rush_attempts")
    assert r["market_season_games"] == 0
    assert r["market_season_mean"] is r["market_season_median"] is r["market_season_std"] is None
    assert r["market_roll3_n"] == 3  # 2024 SB, CON, DIV
    car = dict(stats_rows().filter(pl.col("player_id") == HURTS)
               .select("game_id", "carries").iter_rows())
    want = [car["2024_22_KC_PHI"], car["2024_21_WAS_PHI"], car["2024_20_LA_PHI"]]
    assert r["market_roll3_mean"] == pytest.approx(sum(want) / 3)
    assert r["market_prev"] == want[0]


def test_player_history_equals_stats_rows_for_regular_players(histories):
    app = pl.concat([h.appearances for h in histories.values()])
    for pid in (HURTS, BARKLEY):
        got = set(app.filter(pl.col("player_id") == pid)["game_id"])
        assert got == set(stats_rows().filter(pl.col("player_id") == pid)["game_id"])


def test_cold_start_no_prior_and_one_prior_game(feats, histories):
    app = pl.concat([h.appearances for h in histories.values()])
    first = app.group_by("player_id").agg(pl.col("season").min(), pl.col("week").min(),
                                          pl.col("week").sort().slice(0, 2).alias("w"))
    rookie = first.filter(pl.col("season") == 2025, pl.col("w").list.get(0) == 1,
                          pl.col("w").list.get(1, null_on_oob=True) == 2)["player_id"]
    rookie = min(set(rookie) & set(feats.filter(pl.col("team") == "PHI")["player_id"]))
    none = row(feats, rookie, 1, "receptions")
    assert none["prior_games_n"] == 0 and none["market_prev"] is None
    assert all(none[f"market_roll{k}_n"] == 0 and none[f"market_roll{k}_mean"] is None
               for k in (3, 5, 8))
    assert none["snap_pct_prev"] is None and none["days_since_prev_game"] is None
    assert none["team_games_n"] > 0  # team history exists even when the player has none
    one = row(feats, rookie, 2, "receptions")
    wk1 = stats_rows().filter(pl.col("player_id") == rookie, pl.col("game_id") == game_of("PHI", 1))
    value = wk1["receptions"].item() if wk1.height else 0
    assert one["prior_games_n"] == 1 and one["market_roll3_n"] == 1
    assert one["market_prev"] == one["market_roll3_mean"] == one["market_roll8_mean"] == value
    assert one["market_season_std"] is None and one["market_season_median"] == value


def test_week1_schedule_context_and_home_away(feats):
    r = row(feats, HURTS, 1, "pass_attempts")
    assert r["team_rest_days"] is None  # first game of the season
    assert (r["team"], r["opponent"], r["is_home"], r["is_postseason"]) == ("PHI", "DAL", True,
                                                                           False)
    away = row(feats, HURTS, 2, "pass_attempts")
    assert (away["opponent"], away["is_home"]) == ("KC", False)


def test_rest_days_reflect_the_bye_week(feats):
    k = kickoffs()
    r = row(feats, HURTS, 10, "pass_attempts")
    want = (k[game_of("PHI", 10)] - k[game_of("PHI", 8)]).total_seconds() / 86400
    assert r["team_rest_days"] == pytest.approx(want) and want > 13
    prev = (k[game_of("PHI", 10)] - k[game_of("PHI", 8)]).total_seconds() / 86400
    assert r["days_since_prev_game"] == pytest.approx(prev)


def test_team_opportunity_aggregation(feats):
    tot = team_totals().filter(pl.col("team") == "PHI")
    games = eligible([(g, int(g[:4])) for g in tot["game_id"]], game_of("PHI", 8), 2025)[:5]
    t = tot.filter(pl.col("game_id").is_in(games))
    att, car = t["attempts"].sum(), t["carries"].sum()
    r = row(feats, BARKLEY, 8, "rush_yds")
    assert r["team_games_n"] == 5
    assert r["team_pass_att_roll5_mean"] == pytest.approx(att / 5)
    assert r["team_carries_roll5_mean"] == pytest.approx(car / 5)
    assert r["team_plays_roll5_mean"] == pytest.approx((att + car) / 5)
    assert r["team_pass_rate_roll5"] == pytest.approx(att / (att + car))
    assert r["team_rush_rate_roll5"] == pytest.approx(car / (att + car))


def test_player_opportunity_share(feats):
    target = game_of("PHI", 8)
    ps = stats_rows().filter(pl.col("player_id") == BARKLEY)
    games = eligible(list(ps.select("game_id", "season").iter_rows()), target, 2025)[:5]
    mine = ps.filter(pl.col("game_id").is_in(games))
    team = team_totals().filter(pl.col("team") == "PHI", pl.col("game_id").is_in(games))
    r = row(feats, BARKLEY, 8, "rush_attempts")
    assert r["carries_share_roll5"] == pytest.approx(mine["carries"].sum() / team["carries"].sum())
    assert r["target_share_roll5"] == pytest.approx(mine["targets"].sum() / team["targets"].sum())
    assert r["carries_roll5_mean"] == pytest.approx(mine["carries"].mean())
    assert r["carries_prev"] == ps.filter(pl.col("game_id") == games[0])["carries"].item()


def test_opponent_production_allowed(feats):
    tot = team_totals()
    for week in (1, 6, 12):
        target = game_of("PHI", week)
        opp = row(feats, HURTS, week, "rush_yds")["opponent"]
        opp_games = [(g, int(g[:4])) for g in tot.filter(pl.col("team") == opp)["game_id"]]
        games = eligible(opp_games, target, 2025)
        allowed = [tot.filter(pl.col("game_id") == g, pl.col("team") != opp)["rushing_yards"].item()
                   for g in games]
        this = [a for g, a in zip(games, allowed, strict=True) if g.startswith("2025")]
        r = row(feats, HURTS, week, "rush_yds")
        assert r["opp_allowed_roll5_n"] == min(5, len(games))
        assert r["opp_market_allowed_roll5_mean"] == pytest.approx(
            statistics.mean(allowed[:5]) if allowed else None)
        assert r["opp_allowed_season_games"] == len(this)
        assert r["opp_market_allowed_season_mean"] == pytest.approx(
            statistics.mean(this) if this else None)


def test_snap_features_match_raw_offense_pct(feats):
    sn = pl.concat([raw("snap_counts", s) for s in (2024, 2025)])
    pct = dict(sn.filter(pl.col("pfr_player_id") == "HurtJa00").select("game_id", "offense_pct")
               .iter_rows())
    games = eligible([(g, int(g[:4])) for g in pct], game_of("PHI", 6), 2025)
    r = row(feats, HURTS, 6, "pass_attempts")
    assert r["snap_pct_prev"] == pytest.approx(pct[games[0]])
    assert r["snap_pct_roll3_mean"] == pytest.approx(statistics.mean(pct[g] for g in games[:3]))
    assert r["snap_pct_roll5_n"] == 5


# Leakage.

def test_target_game_stats_cannot_affect_its_own_row(feats, keys):
    g8 = game_of("PHI", 8)
    skill = pl.col("position_group").is_in(["QB", "RB", "WR", "TE"])  # players with offense snaps
    ps = raw("player_stats", 2025).with_columns(
        pl.when((pl.col("game_id") == g8) & skill).then(pl.col(c) + 50).otherwise(pl.col(c))
        .alias(c)
        for c in ("attempts", "carries", "rushing_yards", "receptions", "receiving_yards",
                  "targets"))
    changed = rebuild(keys, player_stats=ps)
    assert changed.filter(pl.col("week") <= 8).equals(feats.filter(pl.col("week") <= 8))
    assert not changed.filter(pl.col("week") == 10).equals(feats.filter(pl.col("week") == 10))


def test_future_games_cannot_change_past_rows(feats, keys):
    g10 = game_of("PHI", 10)
    skill = pl.col("position_group").is_in(["QB", "RB", "WR", "TE"])
    ps = raw("player_stats", 2025).with_columns(
        pl.when((pl.col("game_id") == g10) & skill).then(pl.col("carries") * 3 + 9)
        .otherwise(pl.col("carries")).alias("carries"))
    sn = raw("snap_counts", 2025).with_columns(
        pl.when(pl.col("game_id") == g10).then(0.01).otherwise(pl.col("offense_pct"))
        .alias("offense_pct"))
    changed = rebuild(keys, player_stats=ps, snap_counts=sn)
    for week in (1, 5, 8):
        assert changed.filter(pl.col("week") == week).equals(feats.filter(pl.col("week") == week))
    assert not changed.filter(pl.col("week") == 11).equals(feats.filter(pl.col("week") == 11))


def test_later_seasons_cannot_change_historical_rows(histories):
    keys24 = build_labels(**frames(2024)).labels
    sched = frames(2024)["schedules"]
    alone = build_features(2024, keys24, {2024: histories[2024]}, sched).features
    with_future = build_features(2024, keys24, histories, sched).features
    assert alone.equals(with_future)
    assert alone.filter(pl.col("week") >= 19)["is_postseason"].all()
    assert history_seasons(2018) == [2018] and history_seasons(2025) == [2024, 2025]


def test_rows_available_after_decision_time_are_excluded(feats, histories, keys):
    target = game_of("PHI", 8)
    lock = forecast_time_utc(kickoffs()[target])
    g7 = game_of("PHI", 7)
    h25 = histories[2025]
    late = h25.appearances.with_columns(
        pl.when((pl.col("game_id") == g7) & (pl.col("player_id") == HURTS))
        .then(pl.lit(lock + timedelta(seconds=1))).otherwise(pl.col("available_at"))
        .alias("available_at"))
    at_lock = h25.appearances.with_columns(
        pl.when((pl.col("game_id") == g7) & (pl.col("player_id") == HURTS))
        .then(pl.lit(lock)).otherwise(pl.col("available_at")).alias("available_at"))
    sched = frames(2025)["schedules"]

    def build_with(app):
        hist = {2024: histories[2024], 2025: replace(h25, appearances=app)}
        return row(build_features(2025, keys, hist, sched).features, HURTS, 8, "pass_attempts")

    base, excluded, included = row(feats, HURTS, 8, "pass_attempts"), build_with(late), \
        build_with(at_lock)
    att = dict(stats_rows().filter(pl.col("player_id") == HURTS)
               .select("game_id", "attempts").iter_rows())
    assert excluded["prior_games_n"] == base["prior_games_n"] - 1
    assert excluded["market_prev"] == att[game_of("PHI", 6)] != att[g7]
    # Available exactly at decision_time is allowed: same values, provenance records the lock.
    assert {k: v for k, v in included.items() if k != "max_available_at"} == \
        {k: v for k, v in base.items() if k != "max_available_at"}
    assert included["max_available_at"] == lock
    assert postgame_stats_available_at(kickoffs()[g7]) == kickoffs()[g7] + LAG


def test_every_row_respects_decision_time(feats):
    assert feats.filter(pl.col("max_available_at") > pl.col("decision_time")).height == 0
    assert (feats["decision_time"] < feats["kickoff_utc"]).all()
    locks = {g: forecast_time_utc(k) for g, k in kickoffs().items()}
    assert all(locks[g] == d for g, d in feats.select("game_id", "decision_time").iter_rows())


def test_label_values_cannot_enter_features(feats, histories, keys):
    scrambled = keys.with_columns(pl.lit(999.0).alias("actual"), ~pl.col("played"),
                                  pl.lit("recorded").alias("label_status"))
    again = build_features(2025, scrambled, histories, frames(2025)["schedules"]).features
    assert again.equals(feats)
    assert not {"actual", "played", "label_status"} & set(feats.columns)
    with pytest.raises(ContractViolation, match="label-only column"):
        validate_frame(feats.with_columns(pl.lit(1.0).alias("actual")), "features")


# Missing data, identifiers and validation.

def test_missing_snap_counts_stay_null(keys, feats):
    g5 = game_of("PHI", 5)
    sn = raw("snap_counts", 2025).filter(pl.col("game_id") != g5)
    hist = season_history(**frames(2025, snap_counts=sn))
    assert hist.games_without_snaps == (g5,)
    changed = rebuild(keys, snap_counts=sn)
    r, base = row(changed, HURTS, 6, "pass_attempts"), row(feats, HURTS, 6, "pass_attempts")
    assert r["snap_pct_prev"] is None
    assert r["snap_pct_roll3_n"] == 2 and r["snap_pct_roll5_n"] == 4
    assert r["market_prev"] == base["market_prev"]  # stats of that game still count
    assert r["prior_games_n"] == base["prior_games_n"]


def test_negative_historical_yards_are_valid(keys):
    g7 = game_of("PHI", 7)
    ps = raw("player_stats", 2025).with_columns(
        pl.when((pl.col("game_id") == g7) & (pl.col("player_id") == HURTS)).then(-4)
        .otherwise(pl.col("rushing_yards")).alias("rushing_yards"))
    r = row(rebuild(keys, player_stats=ps), HURTS, 8, "rush_yds")
    assert r["market_prev"] == -4


def _mutate(df: pl.DataFrame, col: str, value, market: str) -> pl.DataFrame:
    hit = (pl.col("player_id") == HURTS) & (pl.col("week") == 8) & (pl.col("market") == market)
    return df.with_columns(pl.when(hit).then(pl.lit(value, dtype=df.schema[col]))
                           .otherwise(pl.col(col)).alias(col))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: pl.concat([df, df.head(1)]), "duplicate"),
        (lambda df: _mutate(df, "player_id", "36389", "rush_yds"), "invalid player_id"),
        (lambda df: _mutate(df, "player_id", None, "rush_yds"), "null required column"),
        (lambda df: _mutate(df, "market", "passing_yards", "rush_yds"), "unsupported market"),
        (lambda df: _mutate(df, "market_prev", -1.0, "rush_attempts"), "count market"),
        (lambda df: _mutate(df, "market_prev", -1.0, "pass_attempts"), "count market"),
        (lambda df: _mutate(df, "market_prev", -1.0, "receptions"), "count market"),
        (lambda df: _mutate(df, "pass_att_prev", -1.0, "rush_yds"), "negative count feature"),
        (lambda df: _mutate(df, "carries_share_roll5", 1.2, "rush_yds"), r"outside \[0, 1\]"),
        (lambda df: _mutate(df, "market_roll3_n", 2, "rush_yds"), "rolling n"),
        (lambda df: _mutate(df, "max_available_at", datetime(2030, 1, 1, tzinfo=UTC),
                            "rush_yds"), "after decision_time"),
        (lambda df: _mutate(df, "decision_time", datetime(2030, 1, 1, tzinfo=UTC), "rush_yds"),
         "not before kickoff"),
        (lambda df: df.with_columns(pl.col("market_roll3_n").cast(pl.Int64)), "wrong_dtype"),
        (lambda df: df.drop("team_rest_days"), "missing"),
    ],
)
def test_validation_rejects_bad_feature_rows(feats, mutate, match):
    with pytest.raises(ContractViolation, match=match):
        validate_features(mutate(feats))


def test_negative_yard_features_pass_validation(feats):
    validate_features(_mutate(feats, "market_prev", -6.0, "rush_yds"))
    validate_features(_mutate(feats, "market_roll3_mean", -1.5, "reception_yds"))


def test_builder_rejects_bad_keys_and_non_gsis_history(histories, keys):
    sched = frames(2025)["schedules"]
    with pytest.raises(FeatureIntegrityError, match="duplicate target keys"):
        build_features(2025, pl.concat([keys, keys.head(1)]), histories, sched)
    bad = keys.with_columns(pl.lit("36389").alias("player_id"))
    with pytest.raises(ContractViolation, match="invalid player_id"):
        build_features(2025, bad.head(5), histories, sched)
    ps = raw("player_stats", 2025).with_columns(
        pl.when(pl.col("player_id") == HURTS).then(pl.lit("HURTS")).otherwise(pl.col("player_id"))
        .alias("player_id"))
    with pytest.raises(FeatureIntegrityError, match="non-GSIS"):
        season_history(**frames(2025, player_stats=ps))


def test_relocated_franchise_keeps_team_and_opponent_history():
    from lineseek.features.player_game import _opponent_features, _team_features
    ts = datetime(2019, 12, 29, 21, tzinfo=UTC)
    hist = pl.DataFrame({
        "game_id": ["2019_17_OAK_DEN", "2019_17_OAK_DEN"], "team": ["OAK", "DEN"],
        "season": [2019, 2019], "week": [17, 17], "kickoff_utc": [ts, ts],
        "available_at": [ts + LAG, ts + LAG], "team_attempts": [30, 25],
        "team_carries": [20, 30],
        **{f"allowed_{c}": [10, 12] for c in ("attempts", "carries", "rushing_yards",
                                              "receptions", "receiving_yards")},
    })
    tgt = pl.DataFrame({"game_id": ["2020_01_LV_CAR"], "team": ["LV"], "opponent": ["OAK"],
                        "season": [2020], "decision_time": [datetime(2020, 9, 9, 16, tzinfo=UTC)]})
    team = _team_features(tgt, hist).row(0, named=True)
    assert team["team_games_n"] == 1 and team["team_pass_att_roll5_mean"] == 30
    opp = _opponent_features(tgt.with_columns(pl.lit("LV").alias("opponent")), hist)
    assert opp.row(0, named=True)["opp_allowed_roll5_n"] == 1


def test_snap_team_swap_correction_is_reused():
    from lineseek.grade.labels import correct_snap_teams
    sn = pl.DataFrame({"game_id": ["2018_21_NE_LA", "2018_20_NE_KC"], "team": ["LA", "NE"],
                       "opponent": ["NE", "KC"]})
    out = correct_snap_teams(sn)
    assert out.rows() == [("2018_21_NE_LA", "NE", "LA"), ("2018_20_NE_KC", "NE", "KC")]


# Schema, provenance, determinism, CLI.

def test_schema_is_exactly_the_contract_plus_feature_set(feats):
    assert dict(feats.schema) == {**ARTIFACT_SCHEMAS["features"], **FEATURE_COLUMNS}
    assert feats.height == feats.select("game_id", "player_id", "market").unique().height
    assert set(feats["feature_set_version"]) == {FEATURE_SET_VERSION}
    keys = feats.select("season", "week", "game_id", "team", "player_id", "market")
    assert keys.equals(keys.sort(keys.columns))


def test_provenance_and_manifest(build, tmp_path):
    df = build.features
    assert set(df["worst_temporal_quality"]) == {"inferred"}
    assert (df["max_available_at"] <= df["decision_time"]).all()
    ins = [{"source": "prop_labels", "path": "p.parquet", "sha256": "abc"}]
    m = write_features(build, 2025, ins, root=tmp_path)
    on_disk = json.loads((tmp_path / m.path).with_suffix(".manifest.json").read_text())
    assert on_disk["inputs"] == ins and on_disk["contract_version"] == CONTRACT_VERSION
    assert on_disk["parameters"]["feature_set_version"] == FEATURE_SET_VERSION
    assert "Wednesday 12:00" in on_disk["parameters"]["decision_time"]
    assert on_disk["schema_variant"] == f"player_game_features/v{FEATURE_SET_VERSION}"


def test_generation_is_deterministic(build, histories, keys, tmp_path):
    fresh = {s: season_history(**frames(s)) for s in (2024, 2025)}
    again = build_features(2025, keys.sample(fraction=1.0, shuffle=True, seed=7), fresh,
                           frames(2025)["schedules"])
    assert again.features.equals(build.features)
    a = write_features(build, 2025, [], root=tmp_path)
    b = write_features(again, 2025, [], root=tmp_path)
    assert a.sha256 == b.sha256 and a.path != b.path


def test_cli_validate(build, tmp_path, capsys):
    path = tmp_path / write_features(build, 2025, [], root=tmp_path).path
    assert cli.main(["validate", str(path)]) == 0
    assert "OK" in capsys.readouterr().out
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        cli.main(["validate", str(path)])
