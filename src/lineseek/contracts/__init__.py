"""Frozen Phase 0 contracts. Changing anything here requires bumping CONTRACT_VERSION."""

from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS, ContractViolation, validate_frame
from lineseek.contracts.identifiers import (
    GAME_ID_RE,
    PLAYER_ID_RE,
    TEAMS,
    SeasonType,
    parse_game_id,
    validate_game_id,
    validate_player_id,
)
from lineseek.contracts.targets import (
    PROP_MARKETS,
    LabelStatus,
    PropMarket,
    Track,
    grade_outcome,
    resolve_actual,
)
from lineseek.contracts.temporal import (
    SOURCE_TEMPORAL_QUALITY,
    Provenance,
    TemporalQuality,
    assert_as_of,
    closing_snapshot_time_utc,
    forecast_time_utc,
    injury_report_available_at,
    kickoff_utc,
    postgame_stats_available_at,
)

CONTRACT_VERSION = "0.11.0"

__all__ = [
    "ARTIFACT_SCHEMAS",
    "CONTRACT_VERSION",
    "GAME_ID_RE",
    "PLAYER_ID_RE",
    "PROP_MARKETS",
    "SOURCE_TEMPORAL_QUALITY",
    "TEAMS",
    "ContractViolation",
    "LabelStatus",
    "PropMarket",
    "Provenance",
    "SeasonType",
    "TemporalQuality",
    "Track",
    "assert_as_of",
    "closing_snapshot_time_utc",
    "forecast_time_utc",
    "grade_outcome",
    "injury_report_available_at",
    "kickoff_utc",
    "parse_game_id",
    "postgame_stats_available_at",
    "resolve_actual",
    "validate_frame",
    "validate_game_id",
    "validate_player_id",
]
