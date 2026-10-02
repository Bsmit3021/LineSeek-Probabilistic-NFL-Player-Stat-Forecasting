"""Evaluation-only player-universe classification (Phase 3 audit finding F1; docs/adr/0004).

Each labelled (game_id, player_id) is assigned the first path that admits it under the
Phase 2 label-universe rules (labels._universe):

- roster:        game-day roster entry (ACT/INA) at a skill position
- participation: no such entry, but a stats row with a skill position group or a skill-position
                 snap; admitted independently of the five target-market stats
- outcome_only:  admitted only because the player recorded activity in a target market

`outcome_only` rows are selected on the outcome and are excluded from primary evaluation. The
classification uses postgame information and must never be a model input.
"""

import polars as pl

from lineseek.grade.labels import (
    SKILL_GROUPS,
    SKILL_POSITIONS,
    STAT_COLUMNS,
    correct_snap_teams,
    normalize_stats_teams,
    roster_game_teams,
)
from lineseek.grade.participation import build_pfr_gsis_map, load_overrides

ADMISSION_PATHS = ("roster", "participation", "outcome_only")
OUTCOME_ADMITTED = "outcome_only"


class UniverseError(ValueError):
    pass


def classify_universe(
    keys: pl.DataFrame,
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    snap_counts: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    players: pl.DataFrame,
    overrides: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """(game_id, player_id, season, admission_path, outcome_admitted) for every key player-game.

    Inputs are the season's snapshot frames the labels were built from.
    """
    pg = ["game_id", "player_id"]
    games = schedules.filter(pl.col("result").is_not_null()).select(
        "game_id", "season", "week", "home_team", "away_team")
    ps = normalize_stats_teams(player_stats.filter(pl.col("player_id").is_not_null()), games)
    sn = correct_snap_teams(snap_counts).filter(pl.col("game_id").is_in(games["game_id"].implode()))
    id_map = build_pfr_gsis_map(sn, players, rosters_weekly,
                                load_overrides() if overrides is None else overrides)
    snap_total = pl.col("offense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)
    skill_snaps = (sn.join(id_map.drop_nulls("gsis_id"), on="pfr_player_id")
                   .rename({"gsis_id": "player_id"})
                   .filter(pl.col("position").is_in(SKILL_POSITIONS), snap_total > 0))
    roster = roster_game_teams(rosters_weekly, games).filter(
        pl.col("position").is_in(SKILL_POSITIONS))
    skill_stats = ps.filter(pl.col("position_group").is_in(SKILL_GROUPS))
    activity = ps.filter(pl.any_horizontal(pl.col(c) != 0 for c in STAT_COLUMNS))

    def flag(df: pl.DataFrame, name: str) -> pl.DataFrame:
        return df.select(pg).unique().with_columns(pl.lit(True).alias(name))

    out = keys.select(*pg, "season").unique()
    for df, name in ((roster, "_roster"), (skill_stats, "_stats"), (skill_snaps, "_snaps"),
                     (activity, "_activity")):
        out = out.join(flag(df, name), on=pg, how="left").with_columns(pl.col(name).fill_null(False))
    path = (pl.when(pl.col("_roster")).then(pl.lit("roster"))
            .when(pl.col("_stats") | pl.col("_snaps")).then(pl.lit("participation"))
            .when(pl.col("_activity")).then(pl.lit(OUTCOME_ADMITTED)))
    out = out.with_columns(path.alias("admission_path"))
    unexplained = out.filter(pl.col("admission_path").is_null())
    if unexplained.height:
        raise UniverseError(f"{unexplained.height} labelled player-game(s) match no universe "
                            f"rule: {unexplained.select(pg).head(5).rows()}")
    return out.select(*pg, "season", "admission_path",
                      (pl.col("admission_path") == OUTCOME_ADMITTED).alias("outcome_admitted")
                      ).sort(pg)
