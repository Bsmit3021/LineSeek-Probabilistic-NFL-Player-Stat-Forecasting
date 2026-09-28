from datetime import UTC, datetime

import polars as pl
import pytest

from lineseek.contracts import assert_as_of
from lineseek.grade.participation import (
    UnmatchedPlayerError,
    build_pfr_gsis_map,
    derive_played,
    load_overrides,
)

OBSERVED = datetime(2026, 9, 28, 12, tzinfo=UTC)
G = "2025_01_ARI_NO"


def snaps(rows: list[tuple[str, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(G, pid, name, "NO", 2025, off, st) for pid, name, off, st in rows],
        schema=["game_id", "pfr_player_id", "player", "team", "season", "offense_snaps", "st_snaps"],
        orient="row",
    )


PLAYERS = pl.DataFrame({"pfr_id": ["AaaaA00", "BbbbB00"], "gsis_id": ["00-0000001", "00-0000002"]})
ROSTERS = pl.DataFrame(
    {"full_name": ["Cee Roster"], "team": ["NO"], "season": [2025], "gsis_id": ["00-0000003"]}
)
NO_OVERRIDES = pl.DataFrame(schema={"pfr_player_id": pl.String, "gsis_id": pl.String})


def test_unmatched_id_raises_with_ids():
    s = snaps([("AaaaA00", "A", 10, 0), ("ZzzzZ00", "Nobody", 5, 0), ("YyyyY00", "Ghost", 0, 1)])
    with pytest.raises(UnmatchedPlayerError, match="YyyyY00, ZzzzZ00") as e:
        build_pfr_gsis_map(s, PLAYERS, ROSTERS, NO_OVERRIDES)
    assert e.value.pfr_ids == ["YyyyY00", "ZzzzZ00"]


def test_roster_name_fallback_and_explicit_override():
    s = snaps([("CeeeC00", "Cee Roster", 3, 0), ("XxxxX00", "Excluded", 1, 0)])
    ov = pl.DataFrame({"pfr_player_id": ["XxxxX00"], "gsis_id": [None]}, schema=NO_OVERRIDES.schema)
    m = build_pfr_gsis_map(s, PLAYERS, ROSTERS, ov)
    assert dict(m.iter_rows()) == {"CeeeC00": "00-0000003", "XxxxX00": None}


def test_special_teams_only_played_and_missing_snap_row_false():
    s = snaps([("AaaaA00", "A", 0, 7), ("BbbbB00", "B", 0, 0)])
    m = build_pfr_gsis_map(s, PLAYERS, ROSTERS, NO_OVERRIDES)
    projected = pl.DataFrame({"game_id": [G] * 3, "player_id": ["00-0000001", "00-0000002", "00-0000009"]})
    out = derive_played(projected, s, m, OBSERVED, "nflreadpy==0.1.5")
    assert dict(out.select("player_id", "played").iter_rows()) == {
        "00-0000001": True,   # special-teams only
        "00-0000002": False,  # snap row with zero snaps
        "00-0000009": False,  # projected, no snap row
    }
    assert out.height == projected.height
    assert set(out["temporal_quality"]) == {"postgame"}


def test_played_is_blocked_from_model_inputs():
    s = snaps([("AaaaA00", "A", 10, 0)])
    m = build_pfr_gsis_map(s, PLAYERS, ROSTERS, NO_OVERRIDES)
    out = derive_played(pl.DataFrame({"game_id": [G], "player_id": ["00-0000001"]}), s, m,
                        OBSERVED, "v")
    with pytest.raises(ValueError, match="postgame"):
        assert_as_of(out, datetime(2030, 1, 1, tzinfo=UTC))


def test_checked_in_overrides_are_well_formed():
    ov = load_overrides()
    assert ov["pfr_player_id"].is_unique().all()
    assert ov["gsis_id"].drop_nulls().str.contains(r"^00-\d{7}$").all()
