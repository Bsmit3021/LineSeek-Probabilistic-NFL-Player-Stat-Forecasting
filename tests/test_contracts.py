from datetime import UTC, datetime

import polars as pl
import pytest

from lineseek.contracts import (
    PROP_MARKETS,
    ContractViolation,
    Provenance,
    TemporalQuality,
    Track,
    assert_as_of,
    grade_outcome,
    injury_report_available_at,
    kickoff_utc,
    parse_game_id,
    resolve_actual,
    validate_frame,
    validate_game_id,
    validate_player_id,
)
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS


def test_game_id_parsing_and_validation():
    assert parse_game_id("2026_03_LAC_BUF") == (2026, 3, "LAC", "BUF")
    validate_game_id("2026_03_LAC_BUF")
    for bad in ("2026_3_LAC_BUF", "2026_03_LAR_BUF", "garbage"):
        with pytest.raises(ValueError):
            validate_game_id(bad)


def test_player_id():
    assert validate_player_id("00-0034857") == "00-0034857"
    with pytest.raises(ValueError):
        validate_player_id("34857")


def test_kickoff_is_eastern_to_utc_across_dst():
    # Verified against The Odds API commence_time for 2026_03_LAC_BUF (17:00Z).
    assert kickoff_utc("2026-09-27", "13:00") == datetime(2026, 9, 27, 17, 0, tzinfo=UTC)
    assert kickoff_utc("2026-12-06", "13:00") == datetime(2026, 12, 6, 18, 0, tzinfo=UTC)


def test_injury_rule_is_conservative_day_before_4pm_et():
    assert injury_report_available_at("2026-09-27") == datetime(2026, 9, 26, 20, 0, tzinfo=UTC)
    assert injury_report_available_at("2026-09-27") < kickoff_utc("2026-09-27", "13:00")


def test_provenance_rules():
    now = datetime.now(UTC)
    Provenance("injuries", "nflreadpy==0.1.5", TemporalQuality.INFERRED, now, now, rule="x")
    with pytest.raises(ValueError, match="rule"):
        Provenance("injuries", "v", TemporalQuality.INFERRED, now, now)
    with pytest.raises(ValueError, match="available_at"):
        Provenance("odds_api", "v", TemporalQuality.EXACT, now)
    with pytest.raises(ValueError, match="UTC"):
        Provenance("x", "v", TemporalQuality.OBSERVED, datetime(2026, 1, 1))  # noqa: DTZ001


def test_assert_as_of_blocks_leakage():
    t = datetime(2026, 9, 27, 16, 0, tzinfo=UTC)
    ok = pl.DataFrame({"available_at": [datetime(2026, 9, 26, tzinfo=UTC)]})
    assert_as_of(ok, t)
    late = pl.DataFrame({"available_at": [datetime(2026, 9, 27, 18, tzinfo=UTC), None]})
    with pytest.raises(ValueError, match="2 row"):
        assert_as_of(late, t)


def test_targets_are_exactly_the_prd_five():
    assert {m.stat_column for m in PROP_MARKETS.values()} == {
        "attempts", "carries", "rushing_yards", "receptions", "receiving_yards",
    }
    assert all(m.odds_api_key.startswith("player_") for m in PROP_MARKETS.values())
    stats = pl.DataFrame({"rushing_yards": [40], "receiving_yards": [None]})
    row = stats.select(
        PROP_MARKETS["rush_yds"].actual(), PROP_MARKETS["reception_yds"].actual()
    ).row(0)
    assert row == (40, 0)


def test_grading_played_games_same_on_both_tracks():
    for track in Track:
        assert grade_outcome(40, 39.5, track, played=True) == "over"
        assert grade_outcome(40, 40, track, played=True) == "push"
        assert grade_outcome(40, 40.5, track, played=True) == "under"


def test_forecast_track_scores_missed_games_as_zero():
    assert resolve_actual(None, Track.FORECAST) == 0.0
    assert resolve_actual(None, Track.FORECAST, played=False) == 0.0
    assert grade_outcome(None, 45.5, Track.FORECAST, played=False) == "under"


def test_market_track_voids_non_participants_only():
    assert grade_outcome(None, 45.5, Track.MARKET, played=False) == "void"
    assert grade_outcome(12, 45.5, Track.MARKET, played=False) == "void"
    # Participated but no stats row (zero touches): graded, not void.
    assert grade_outcome(None, 0.5, Track.MARKET, played=True) == "under"
    with pytest.raises(ValueError, match="participation"):
        grade_outcome(None, 45.5, Track.MARKET)


def _empty(artifact: str) -> pl.DataFrame:
    return pl.DataFrame(schema=ARTIFACT_SCHEMAS[artifact])


def test_validate_frame():
    for name in ARTIFACT_SCHEMAS:
        validate_frame(_empty(name), name)
    with pytest.raises(ContractViolation, match="missing"):
        validate_frame(_empty("games").drop("kickoff_utc"), "games")
    naive = _empty("games").with_columns(pl.col("kickoff_utc").dt.replace_time_zone(None))
    with pytest.raises(ContractViolation, match="wrong_dtype"):
        validate_frame(naive, "games")
    t = datetime(2026, 9, 27, 16, tzinfo=UTC)
    leaky = pl.DataFrame(
        {"game_id": ["g"], "player_id": ["p"], "decision_time": [t],
         "max_available_at": [datetime(2026, 9, 27, 17, tzinfo=UTC)],
         "worst_temporal_quality": ["exact"], "feature_set_version": ["v"]},
        schema=ARTIFACT_SCHEMAS["features"],
    )
    with pytest.raises(ContractViolation, match="after decision_time"):
        validate_frame(leaky, "features")
