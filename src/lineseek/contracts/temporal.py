"""Timestamp semantics and temporal-quality provenance.

All datetimes are timezone-aware UTC. Four distinct times exist for any record:

- event_time:    when the real-world thing happened (a play, a snapshot, a kickoff).
- available_at:  earliest time the information could have been known publicly. Must never be
                 earlier than the truth; when unknown it is estimated conservatively (late).
- observed_at:   when our system fetched it.
- decision_time: the as-of time of a prediction. A record may feed a prediction only if
                 available_at <= decision_time.
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

import polars as pl

NFL_TZ = ZoneInfo("America/New_York")
INJURY_REPORT_LOCAL_TIME = time(16, 0)
FORECAST_RUN_WEEKDAY = 2  # Wednesday
FORECAST_RUN_LOCAL_TIME = time(12, 0)  # ET
CLOSING_OFFSET = timedelta(seconds=1)  # closing = last snapshot strictly before kickoff


class TemporalQuality(StrEnum):
    """How trustworthy `available_at` is. Ordered best to worst."""

    EXACT = "exact"        # Source publishes a publication/snapshot timestamp.
    OBSERVED = "observed"  # Only our fetch time is known; available_at = observed_at (upper bound).
    INFERRED = "inferred"  # Derived from a documented, conservative rule (see Provenance.rule).
    POSTGAME = "postgame"  # Known only after the game ends; labels/outcomes only, never features.
    UNKNOWN = "unknown"    # No defensible availability time; not usable as a pre-game feature.


@dataclass(frozen=True)
class Provenance:
    source: str
    source_version: str
    temporal_quality: TemporalQuality
    observed_at: datetime
    available_at: datetime | None = None
    rule: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("observed_at", "available_at"):
            v = getattr(self, name)
            if v is not None and v.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be tz-aware UTC, got {v!r}")
        if self.temporal_quality is TemporalQuality.INFERRED and not self.rule:
            raise ValueError("INFERRED provenance requires a rule")
        if self.temporal_quality in (TemporalQuality.EXACT, TemporalQuality.INFERRED) and (
            self.available_at is None
        ):
            raise ValueError(f"{self.temporal_quality} provenance requires available_at")


# Phase 1 audit results (see docs/audit/phase1_sources.md). Key: (source, field_group).
SOURCE_TEMPORAL_QUALITY: dict[tuple[str, str], TemporalQuality] = {
    ("schedules", "kickoff"): TemporalQuality.EXACT,
    ("schedules", "scores"): TemporalQuality.POSTGAME,
    ("schedules", "lines"): TemporalQuality.UNKNOWN,
    ("pbp", "plays"): TemporalQuality.POSTGAME,
    ("player_stats", "stats"): TemporalQuality.POSTGAME,
    ("snap_counts", "snaps"): TemporalQuality.POSTGAME,
    ("injuries", "2024_and_earlier"): TemporalQuality.EXACT,
    ("injuries", "2025_and_later"): TemporalQuality.INFERRED,
    ("depth_charts", "2024_and_earlier"): TemporalQuality.UNKNOWN,
    ("depth_charts", "2025_and_later"): TemporalQuality.EXACT,
    ("rosters_weekly", "status"): TemporalQuality.UNKNOWN,
    ("players", "ids"): TemporalQuality.UNKNOWN,
    ("prop_labels", "outcomes"): TemporalQuality.POSTGAME,
    ("odds_api", "prop_lines"): TemporalQuality.EXACT,
}


def kickoff_utc(gameday: str | date, gametime: str) -> datetime:
    """nflverse schedules store gameday/gametime in US/Eastern local time."""
    d = date.fromisoformat(gameday) if isinstance(gameday, str) else gameday
    local = datetime.combine(d, time.fromisoformat(gametime), tzinfo=NFL_TZ)
    return local.astimezone(UTC)


def injury_report_available_at(gameday: str | date) -> datetime:
    """Conservative availability for final game-status injury reports lacking timestamps.

    NFL final reports (with game statuses) are out by 16:00 ET the day before kickoff at the
    latest (Wed for Thu games, Fri for Sun, Sat for Mon), so this is never earlier than truth.
    """
    d = date.fromisoformat(gameday) if isinstance(gameday, str) else gameday
    local = datetime.combine(d - timedelta(days=1), INJURY_REPORT_LOCAL_TIME, tzinfo=NFL_TZ)
    return local.astimezone(UTC)


def forecast_time_utc(kickoff: datetime) -> datetime:
    """Latest Wednesday 12:00 ET strictly before kickoff; the weekly forecast lock, in UTC."""
    local_kick = kickoff.astimezone(NFL_TZ)
    d = local_kick.date() - timedelta(days=(local_kick.weekday() - FORECAST_RUN_WEEKDAY) % 7)
    run = datetime.combine(d, FORECAST_RUN_LOCAL_TIME, tzinfo=NFL_TZ)
    if run >= local_kick:
        run = datetime.combine(d - timedelta(days=7), FORECAST_RUN_LOCAL_TIME, tzinfo=NFL_TZ)
    return run.astimezone(UTC)


def closing_snapshot_time_utc(kickoff: datetime) -> datetime:
    """Request time for the closing line: provider returns the last snapshot at or before it."""
    return kickoff.astimezone(UTC) - CLOSING_OFFSET


NOT_FEATURE_QUALITY = frozenset({TemporalQuality.POSTGAME.value, TemporalQuality.UNKNOWN.value})


def assert_as_of(df: pl.DataFrame, decision_time: datetime, col: str = "available_at") -> None:
    """Leakage guard: fail if any row was not available at decision_time."""
    if decision_time.utcoffset() != timedelta(0):
        raise ValueError("decision_time must be tz-aware UTC")
    if "temporal_quality" in df.columns:
        n = df.filter(pl.col("temporal_quality").is_in(list(NOT_FEATURE_QUALITY))).height
        if n:
            raise ValueError(f"{n} row(s) have postgame/unknown temporal quality")
    bad = df.filter(pl.col(col).is_null() | (pl.col(col) > decision_time))
    if bad.height:
        raise ValueError(f"{bad.height} row(s) not available as of {decision_time.isoformat()}")
