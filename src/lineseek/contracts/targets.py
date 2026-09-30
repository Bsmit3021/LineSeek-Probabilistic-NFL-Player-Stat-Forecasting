"""Frozen v1 prediction targets: five NFL player-prop stats, full game, regular + postseason.

Label source of record is nflreadpy `load_player_stats(summary_level="week")`, keyed by
(player_id, game_id). PBP aggregates are NOT used for labels (they diverge from official
player stats for a small share of players, e.g. laterals and stat corrections).

Two grading tracks (survivorship rule: players who miss games are still scored):
- FORECAST: every locked projection is scored. A player without a stats row scores 0.
- MARKET:   sportsbook rules. A player who did not participate is void; a participant with
            no stats row scores 0. Participation is an explicit input, never inferred from
            a missing stats row (zero-touch participants may have no row).
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import polars as pl


class Track(StrEnum):
    FORECAST = "forecast"
    MARKET = "market"


class LabelStatus(StrEnum):
    """Why a prop_labels row has the `actual` it has (see docs/adr/0002)."""

    RECORDED = "recorded"  # official stats row; played; actual = stat value
    PARTICIPANT_NO_STATS_ROW = "participant_no_stats_row"  # played, no stats row; actual = 0
    DID_NOT_PLAY = "did_not_play"  # played is False; actual is null


@dataclass(frozen=True)
class PropMarket:
    key: str
    odds_api_key: str
    stat_column: str
    # Counts can't be negative; rushing/receiving yards legitimately can.
    nonnegative: bool

    def actual(self) -> pl.Expr:
        """Realized stat for this market, evaluated against a player_stats frame."""
        return pl.col(self.stat_column).fill_null(0).alias(f"actual_{self.key}")


PROP_MARKETS: dict[str, PropMarket] = {
    m.key: m
    for m in (
        PropMarket("pass_attempts", "player_pass_attempts", "attempts", True),
        PropMarket("rush_attempts", "player_rush_attempts", "carries", True),
        PropMarket("rush_yds", "player_rush_yds", "rushing_yards", False),
        PropMarket("receptions", "player_receptions", "receptions", True),
        PropMarket("reception_yds", "player_reception_yds", "receiving_yards", False),
    )
}

Outcome = Literal["over", "under", "push", "void"]


def resolve_actual(actual: float | None, track: Track, played: bool | None = None) -> float | None:
    """Actual used for scoring. `actual is None` means no player_stats row for the game."""
    if track is Track.MARKET:
        if played is None:
            raise ValueError("market track requires an explicit participation flag")
        if not played:
            return None
    return 0.0 if actual is None else actual


def grade_outcome(
    actual: float | None, line: float, track: Track, played: bool | None = None
) -> Outcome:
    value = resolve_actual(actual, track, played)
    if value is None:
        return "void"
    if value > line:
        return "over"
    return "under" if value < line else "push"
