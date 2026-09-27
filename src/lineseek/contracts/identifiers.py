"""Canonical identifiers. nflverse IDs are the system of record; external IDs map onto them."""

import re
from enum import StrEnum
from typing import NamedTuple

# nflverse game_id: {season}_{week:02d}_{away}_{home}, e.g. 2026_03_LAC_BUF
GAME_ID_RE = re.compile(r"^(\d{4})_(\d{2})_([A-Z]{2,3})_([A-Z]{2,3})$")
# NFL GSIS player id, e.g. 00-0034857. Used as player_id / gsis_id across nflverse tables.
PLAYER_ID_RE = re.compile(r"^00-\d{7}$")

# nflverse current-era abbreviations (schedules 2024-2026). Rams are "LA", not "LAR".
TEAMS: frozenset[str] = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE",
    "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
})


class SeasonType(StrEnum):
    REG = "REG"
    POST = "POST"


class GameKey(NamedTuple):
    season: int
    week: int
    away_team: str
    home_team: str


def parse_game_id(game_id: str) -> GameKey:
    m = GAME_ID_RE.match(game_id)
    if not m:
        raise ValueError(f"invalid game_id: {game_id!r}")
    season, week, away, home = m.groups()
    return GameKey(int(season), int(week), away, home)


def validate_game_id(game_id: str) -> str:
    key = parse_game_id(game_id)
    unknown = {key.away_team, key.home_team} - TEAMS
    if unknown:
        raise ValueError(f"unknown team(s) {unknown} in game_id {game_id!r}")
    return game_id


def validate_player_id(player_id: str) -> str:
    if not PLAYER_ID_RE.match(player_id or ""):
        raise ValueError(f"invalid gsis player_id: {player_id!r}")
    return player_id
