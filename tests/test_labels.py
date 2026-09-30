import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from lineseek.contracts import (
    PROP_MARKETS,
    ContractViolation,
    LabelStatus,
    Track,
    assert_as_of,
    grade_outcome,
    validate_frame,
)
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.grade import cli
from lineseek.grade.labels import (
    LABEL_INPUTS,
    LabelIntegrityError,
    build_labels,
    build_season,
    normalize_stats_teams,
    write_labels,
)
from lineseek.grade.participation import UnmatchedPlayerError
from lineseek.ingest.snapshot import write_snapshot
from lineseek.ingest.sources import SOURCES

FIX = Path(__file__).parent / "fixtures" / "labels"
T0 = datetime(2026, 9, 28, 12, tzinfo=UTC)
G = "2025_01_DAL_PHI"
HURTS, METCHIE, SANDERS, MCKEE = "00-0036389", "00-0037614", "00-0035243", "00-0038400"


def raw(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIX / f"{name}.parquet")


def snapshot_frames(observed_at: datetime = T0) -> dict[str, pl.DataFrame]:
    """Label inputs as the ingest pipeline would store them (conformed + provenance)."""
    return {n: SOURCES[n].build(raw(n), 2025, observed_at) for n in LABEL_INPUTS}


@pytest.fixture(scope="module")
def inputs() -> dict[str, pl.DataFrame]:
    return snapshot_frames()


@pytest.fixture(scope="module")
def build(inputs):
    return build_labels(**inputs)


@pytest.fixture(scope="module")
def labels(build) -> pl.DataFrame:
    return build.labels


def gsis(pfr_id: str) -> str:
    return raw("players").filter(pl.col("pfr_id") == pfr_id)["gsis_id"].item()


def label(labels: pl.DataFrame, player_id: str, market: str) -> dict:
    return labels.filter(pl.col("player_id") == player_id, pl.col("market") == market).row(
        0, named=True)


def test_known_player_values_for_all_five_markets(labels):
    got = {m: label(labels, HURTS, m)["actual"] for m in PROP_MARKETS}
    assert got == {"pass_attempts": 23, "rush_attempts": 14, "rush_yds": 62, "receptions": 0,
                   "reception_yds": 0}
    row = label(labels, HURTS, "pass_attempts")
    assert (row["season"], row["week"], row["team"], row["opponent"]) == (2025, 1, "PHI", "DAL")
    assert row["played"] and row["label_status"] == LabelStatus.RECORDED


def test_every_recorded_label_equals_its_official_stat_column(labels, inputs):
    stats = inputs["player_stats"]
    for key, m in PROP_MARKETS.items():
        joined = labels.filter(pl.col("market") == key, pl.col("label_status") == "recorded").join(
            stats.select("game_id", "player_id", m.stat_column), on=["game_id", "player_id"])
        assert joined.height == 20  # skill players with a stats row in the fixture game
        assert (joined["actual"] == joined[m.stat_column].cast(pl.Float64)).all(), key


def test_grain_is_unique_and_covers_exactly_the_five_markets(labels):
    assert set(labels["market"]) == set(PROP_MARKETS)
    assert labels.height == labels.select("game_id", "player_id", "market").unique().height
    per_player = labels.group_by("player_id").len()["len"]
    assert set(per_player) == {len(PROP_MARKETS)}


def test_played_zero_differs_from_did_not_play(labels):
    zero = label(labels, METCHIE, "receptions")  # 1 offensive snap, stats row, no catches
    no_row = label(labels, gsis("CalcGr00"), "receptions")  # 37 snaps, no stats row
    st_only = label(labels, gsis("FlouRy00"), "rush_attempts")  # special teams snaps only
    dnp = label(labels, MCKEE, "receptions")  # declared inactive
    assert (zero["actual"], zero["played"], zero["label_status"]) == (0, True, "recorded")
    assert (no_row["actual"], no_row["played"]) == (0, True)
    assert no_row["label_status"] == st_only["label_status"] == "participant_no_stats_row"
    assert st_only["played"] is True
    assert (dnp["actual"], dnp["played"], dnp["label_status"]) == (None, False, "did_not_play")
    # Phase 0 grading: participant zero is graded, non-participant is void on the market track.
    assert grade_outcome(zero["actual"], 0.5, Track.MARKET, zero["played"]) == "under"
    assert grade_outcome(dnp["actual"], 0.5, Track.MARKET, dnp["played"]) == "void"
    assert grade_outcome(dnp["actual"], 0.5, Track.FORECAST, dnp["played"]) == "under"


def test_legitimate_negative_yards_are_kept(labels):
    assert label(labels, SANDERS, "reception_yds")["actual"] == -3
    df = labels.with_columns(
        pl.when((pl.col("player_id") == HURTS) & (pl.col("market") == "rush_yds"))
        .then(-5.0).otherwise(pl.col("actual")).alias("actual"))
    validate_frame(df, "prop_labels")


def _set(labels: pl.DataFrame, col: str, value, market: str = "rush_attempts") -> pl.DataFrame:
    hit = (pl.col("player_id") == HURTS) & (pl.col("market") == market)
    return labels.with_columns(
        pl.when(hit).then(pl.lit(value, dtype=labels.schema[col])).otherwise(pl.col(col))
        .alias(col))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: pl.concat([df, df.head(1)]), "duplicate"),
        (lambda df: _set(df, "market", "passing_yards"), "unsupported market"),
        (lambda df: _set(df, "player_id", None), "null required column"),
        (lambda df: _set(df, "player_id", "36389"), "invalid player_id"),
        (lambda df: _set(df, "actual", -1.0, "pass_attempts"), "negative count market"),
        (lambda df: _set(df, "actual", -1.0, "rush_attempts"), "negative count market"),
        (lambda df: _set(df, "actual", -1.0, "receptions"), "negative count market"),
        (lambda df: _set(df, "actual", 2.5), "non-integer"),
        (lambda df: _set(df, "week", 2), "disagree with game_id"),
        (lambda df: _set(df, "opponent", "NYG"), "team/opponent"),
        (lambda df: _set(df, "temporal_quality", "exact"), "not postgame"),
        (lambda df: _set(df, "played", False), "recorded row not played"),
    ],
)
def test_validation_rejects_bad_labels(labels, mutate, match):
    with pytest.raises(ContractViolation, match=match):
        validate_frame(mutate(labels), "prop_labels")


def test_did_not_play_with_a_value_is_rejected(labels):
    df = labels.with_columns(pl.when(pl.col("player_id") == MCKEE).then(0.0)
                             .otherwise(pl.col("actual")).alias("actual"))
    with pytest.raises(ContractViolation, match="did_not_play row played or has actual"):
        validate_frame(df, "prop_labels")


def test_missing_player_id_rows_are_quarantined_not_labelled(build, inputs):
    assert build.quarantined.height == 1
    assert build.quarantined["player_id"].null_count() == 1
    assert build.labels["player_id"].null_count() == 0
    active = inputs | {"player_stats": inputs["player_stats"].with_columns(
        pl.when(pl.col("player_id").is_null()).then(3).otherwise(pl.col("carries"))
        .alias("carries"))}
    with pytest.raises(LabelIntegrityError, match="without player_id but with market activity"):
        build_labels(**active)


def test_null_market_stat_fails_instead_of_becoming_zero(inputs):
    ps = inputs["player_stats"].with_columns(
        pl.when(pl.col("player_id") == HURTS).then(None).otherwise(pl.col("carries"))
        .alias("carries"))
    with pytest.raises(LabelIntegrityError, match="null market stat"):
        build_labels(**inputs | {"player_stats": ps})


def test_duplicate_source_stats_row_fails(inputs):
    ps = inputs["player_stats"]
    with pytest.raises(LabelIntegrityError, match="duplicate player_stats"):
        build_labels(**inputs | {"player_stats": pl.concat([ps, ps.head(1)])})


def test_stats_without_snaps_is_a_conflict(inputs):
    sn = inputs["snap_counts"].filter(pl.col("pfr_player_id") != "BarkSa00")
    with pytest.raises(LabelIntegrityError, match="snap/stat conflict"):
        build_labels(**inputs | {"snap_counts": sn})


def test_unmatched_snap_player_id_fails(inputs):
    sn = inputs["snap_counts"]
    extra = sn.head(1).with_columns(pl.lit("NobodY99").alias("pfr_player_id"),
                                    pl.lit("Nobody").alias("player"))
    with pytest.raises(UnmatchedPlayerError, match="NobodY99"):
        build_labels(**inputs | {"snap_counts": pl.concat([sn, extra])})


def test_incomplete_game_fails_or_is_explicitly_skipped(inputs):
    partial = inputs | {"snap_counts": inputs["snap_counts"].filter(pl.col("team") != "PHI")}
    with pytest.raises(LabelIntegrityError, match="lack stats or snap rows"):
        build_labels(**partial)
    skipped = build_labels(**partial, skip_incomplete=True)
    assert skipped.incomplete_games == [G] and skipped.labels.height == 0


def test_team_comes_from_stats_not_snap_counts(inputs):
    sn = inputs["snap_counts"].with_columns(
        pl.when(pl.col("pfr_player_id") == "HurtJa00").then(pl.lit("DAL"))
        .otherwise(pl.col("team")).alias("team"))
    b = build_labels(**inputs | {"snap_counts": sn})
    assert label(b.labels, HURTS, "rush_yds")["team"] == "PHI"
    assert b.snap_team_mismatches == 1


def test_stats_team_disagreeing_with_roster_fails(inputs):
    rw = inputs["rosters_weekly"].with_columns(
        pl.when(pl.col("gsis_id") == HURTS).then(pl.lit("DAL")).otherwise(pl.col("team"))
        .alias("team"))
    with pytest.raises(LabelIntegrityError, match="disagrees with the game-day roster"):
        build_labels(**inputs | {"rosters_weekly": rw})


def test_relocated_team_code_is_normalized_only_for_that_game():
    games = pl.DataFrame({"game_id": ["2018_01_LA_OAK", "2020_01_LV_CAR"], "season": [2018, 2020],
                          "week": [1, 1], "home_team": ["OAK", "CAR"],
                          "away_team": ["LA", "LV"]})
    ps = pl.DataFrame({"game_id": ["2018_01_LA_OAK", "2018_01_LA_OAK", "2020_01_LV_CAR"],
                       "team": ["LV", "LA", "LV"]})
    assert normalize_stats_teams(ps, games)["team"].to_list() == ["OAK", "LA", "LV"]


def test_provenance_survives_and_blocks_model_use(inputs):
    later = T0 + timedelta(hours=3)
    b = build_labels(**inputs | {"snap_counts": snapshot_frames(later)["snap_counts"]})
    df = b.labels
    assert set(df["source"]) == {"player_stats+snap_counts"}
    assert set(df["source_version"]) == {"nflreadpy==0.1.5"}
    assert set(df["temporal_quality"]) == {"postgame"}
    assert set(df["observed_at"]) == set(df["available_at"]) == {later}
    with pytest.raises(ValueError, match="postgame"):
        assert_as_of(df, datetime(2030, 1, 1, tzinfo=UTC))


def test_output_schema_matches_contract_exactly(labels):
    assert dict(labels.schema) == ARTIFACT_SCHEMAS["prop_labels"]
    keys = labels.select("season", "week", "game_id", "team", "player_id", "market")
    assert keys.equals(keys.sort(keys.columns))


def test_build_is_deterministic_and_written_append_only(inputs, tmp_path):
    first, second = build_labels(**inputs), build_labels(**snapshot_frames())
    assert first.labels.equals(second.labels)
    ins = [{"source": "player_stats", "path": "x.parquet", "sha256": "abc"}]
    m1 = write_labels(first, 2025, ins, root=tmp_path)
    m2 = write_labels(second, 2025, ins, root=tmp_path)
    assert m1.sha256 == m2.sha256 and m1.path != m2.path
    on_disk = json.loads((tmp_path / m1.path).with_suffix(".manifest.json").read_text())
    assert on_disk["inputs"] == ins and on_disk["source"] == "prop_labels"
    assert on_disk["temporal_quality"] == {"outcomes": "postgame"}


def test_build_season_reads_verified_snapshots(tmp_path):
    frames = snapshot_frames()
    for name, df in frames.items():
        write_snapshot(df, name, name, [2025], None, T0, root=tmp_path)
    b, ins = build_season(2025, tmp_path)
    assert b.labels.equals(build_labels(**frames).labels)
    assert [i["source"] for i in ins] == list(LABEL_INPUTS)
    target = tmp_path / ins[1]["path"]
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(LabelIntegrityError, match="checksum mismatch"):
        build_season(2025, tmp_path)


def test_cli_validate_checks_contract_and_checksum(build, tmp_path, capsys):
    m = write_labels(build, 2025, [], root=tmp_path)
    path = tmp_path / m.path
    assert cli.main(["validate", str(path)]) == 0
    assert "OK" in capsys.readouterr().out
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        cli.main(["validate", str(path)])


def test_players_source_is_an_unknown_quality_reference_table(inputs):
    assert SOURCES["players"].temporal_quality(2025) == {"ids": "unknown"}
    assert set(inputs["players"]["temporal_quality"]) == {"unknown"}
