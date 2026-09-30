"""Canonical postgame prop labels: one row per (game_id, player_id, market), completed games only.

Labels are ground truth (temporal_quality=postgame), never model inputs: assert_as_of rejects
them. Semantics, null rules and the label universe are documented in docs/adr/0002.
"""

import hashlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from lineseek.contracts import PROP_MARKETS, LabelStatus, TemporalQuality, validate_frame
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS, UTC_TS
from lineseek.grade.participation import build_pfr_gsis_map, derive_played, load_overrides
from lineseek.ingest.snapshot import Manifest, data_root, latest_manifest, write_snapshot

LABELS_ROOT_ENV = "LINESEEK_LABELS_ROOT"
DEFAULT_LABELS_ROOT = "data/labels"
LABEL_INPUTS = ("schedules", "player_stats", "snap_counts", "rosters_weekly", "players")
SKILL_POSITIONS = ("QB", "RB", "FB", "HB", "WR", "TE")  # snap/roster `position`
SKILL_GROUPS = ("QB", "RB", "WR", "TE")  # player_stats `position_group`
ROSTER_STATUSES = ("ACT", "INA")  # game-day roster: active or declared inactive
STAT_COLUMNS = [m.stat_column for m in PROP_MARKETS.values()]
LABEL_SOURCE = "player_stats+snap_counts"
# player_stats uses current codes for relocated teams (2018-19 Raiders are LV, schedules OAK).
PLAYER_STATS_TEAM_ALIASES = {"LV": "OAK"}


class LabelIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class LabelBuild:
    labels: pl.DataFrame
    # player_stats rows without a player_id (team-level rows); never labelled.
    quarantined: pl.DataFrame
    # Completed games missing stats or snap rows for a team; only non-empty if skipped.
    incomplete_games: list[str] = field(default_factory=list)
    # Labelled players whose snap-count team differs from the assigned team (reported only).
    snap_team_mismatches: int = 0


def labels_root() -> Path:
    return Path(os.environ.get(LABELS_ROOT_ENV, DEFAULT_LABELS_ROOT))


def _fail(msg: str, df: pl.DataFrame) -> None:
    if df.height:
        raise LabelIntegrityError(f"{df.height} {msg}: {df.head(10).rows()}")


def _any_market_activity() -> pl.Expr:
    return pl.any_horizontal(pl.col(c) != 0 for c in STAT_COLUMNS)


def _team_games(games: pl.DataFrame) -> pl.DataFrame:
    return pl.concat([
        games.select("game_id", "season", "week", pl.col(side).alias("team"))
        for side in ("home_team", "away_team")
    ])


def normalize_stats_teams(ps: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Map player_stats team codes onto schedule codes, only where the game has that team."""
    tg = _team_games(games).select("game_id", pl.col("team").alias("_sched_team"))
    for alias, canonical in PLAYER_STATS_TEAM_ALIASES.items():
        hit = tg.filter(pl.col("_sched_team") == canonical)
        ps = ps.join(hit, on="game_id", how="left").with_columns(
            pl.when(pl.col("team") == alias).then(pl.col("_sched_team").fill_null(pl.col("team")))
            .otherwise(pl.col("team")).alias("team")
        ).drop("_sched_team")
    return ps


def _check_stats(ps: pl.DataFrame, games: pl.DataFrame) -> None:
    _fail("duplicate player_stats (game_id, player_id)",
          ps.filter(pl.struct("game_id", "player_id").is_duplicated())
          .select("game_id", "player_id"))
    _fail("player_stats rows with a null market stat",
          ps.filter(pl.any_horizontal(pl.col(c).is_null() for c in STAT_COLUMNS))
          .select("game_id", "player_id"))
    _fail("player_stats rows for games that are not completed",
          ps.join(games, on="game_id", how="anti").select("game_id", "player_id"))
    _fail("player_stats rows whose season/week/team disagree with schedules",
          ps.join(_team_games(games), on=["game_id", "season", "week", "team"], how="anti")
          .select("game_id", "player_id", "season", "week", "team"))


def _incomplete_games(games: pl.DataFrame, ps: pl.DataFrame, sn: pl.DataFrame) -> list[str]:
    tg = _team_games(games).select("game_id", "team")
    missing = pl.concat([
        tg.join(src.select("game_id", "team").unique(), on=["game_id", "team"], how="anti")
        for src in (ps, sn)
    ])
    return sorted(missing["game_id"].unique().to_list())


def _universe(games, ps, sn, id_map, rosters) -> tuple[pl.DataFrame, int]:
    """Every (game_id, player_id, team) that gets labels: skill players plus any market activity.

    Team precedence: player_stats, then game-day roster, then snap counts (snap team is
    unreliable, e.g. swapped in the 2018 Super Bowl). Stats vs roster disagreement fails.
    """
    keys = ["game_id", "player_id"]
    from_stats = ps.filter(pl.col("position_group").is_in(SKILL_GROUPS) | _any_market_activity())
    snap_total = pl.col("offense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)
    from_snaps = (
        sn.filter(pl.col("position").is_in(SKILL_POSITIONS), snap_total > 0)
        .join(id_map.drop_nulls("gsis_id"), on="pfr_player_id")
        .rename({"gsis_id": "player_id"})
    )
    from_roster = (
        rosters.filter(pl.col("status").is_in(ROSTER_STATUSES),
                       pl.col("position").is_in(SKILL_POSITIONS), pl.col("gsis_id").is_not_null())
        .join(_team_games(games), on=["season", "week", "team"])
        .rename({"gsis_id": "player_id"})
    )
    uni = pl.concat([f.select(keys) for f in (from_stats, from_snaps, from_roster)]).unique()

    def team_of(df: pl.DataFrame, name: str) -> pl.DataFrame:
        t = df.select(*keys, pl.col("team").alias(name)).unique()
        _fail(f"players with several {name} values in a game",
              t.filter(pl.struct(keys).is_duplicated()).sort(keys))
        return t

    snaps_all = sn.join(id_map.drop_nulls("gsis_id"), on="pfr_player_id").rename(
        {"gsis_id": "player_id"})
    uni = (
        uni.join(team_of(ps, "stats_team"), on=keys, how="left")
        .join(team_of(from_roster, "roster_team"), on=keys, how="left")
        .join(team_of(snaps_all, "snap_team"), on=keys, how="left")
    )
    _fail("players whose player_stats team disagrees with the game-day roster",
          uni.filter(pl.col("stats_team") != pl.col("roster_team")).sort(keys))
    uni = uni.with_columns(pl.coalesce("stats_team", "roster_team", "snap_team").alias("team"))
    snap_mismatch = uni.filter(pl.col("snap_team") != pl.col("team")).height
    return uni.select(*keys, "team"), snap_mismatch


def _provenance_value(frames: list[pl.DataFrame], col: str) -> str:
    values = sorted(set().union(*(f[col].drop_nulls().unique().to_list() for f in frames)))
    if len(values) != 1:
        raise LabelIntegrityError(f"inputs disagree on {col}: {values}")
    return values[0]


def build_labels(
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    snap_counts: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    players: pl.DataFrame,
    overrides: pl.DataFrame | None = None,
    skip_incomplete: bool = False,
) -> LabelBuild:
    """Build canonical labels from one season's snapshot frames (with provenance columns)."""
    games = schedules.filter(pl.col("result").is_not_null()).select(
        "game_id", "season", "week", "home_team", "away_team")
    _fail("duplicate schedule game_id", games.filter(pl.col("game_id").is_duplicated()))

    quarantined = player_stats.filter(pl.col("player_id").is_null())
    _fail("player_stats rows without player_id but with market activity",
          quarantined.filter(_any_market_activity()).select("game_id", "team", *STAT_COLUMNS))
    ps = normalize_stats_teams(player_stats.filter(pl.col("player_id").is_not_null()), games)
    _check_stats(ps, games)

    incomplete = _incomplete_games(games, ps, snap_counts)
    if incomplete and not skip_incomplete:
        raise LabelIntegrityError(
            f"{len(incomplete)} completed game(s) lack stats or snap rows for a team: "
            f"{incomplete[:10]} (use skip_incomplete to exclude them)")
    games = games.filter(~pl.col("game_id").is_in(incomplete))
    ps = ps.filter(pl.col("game_id").is_in(games["game_id"].implode()))
    sn = snap_counts.filter(pl.col("game_id").is_in(games["game_id"].implode()))

    id_map = build_pfr_gsis_map(sn, players, rosters_weekly,
                                load_overrides() if overrides is None else overrides)
    observed_at = max(f["observed_at"].max() for f in
                      (schedules, player_stats, snap_counts, rosters_weekly, players))
    uni, snap_team_mismatches = _universe(games, ps, sn, id_map, rosters_weekly)
    played = derive_played(uni, sn, id_map, observed_at, "unused").select(
        "game_id", "player_id", "team", "played")

    stats = ps.select("game_id", "player_id", *STAT_COLUMNS, pl.lit(True).alias("has_row"))
    base = played.join(stats, on=["game_id", "player_id"], how="left").with_columns(
        pl.col("has_row").fill_null(False))
    _fail("players with market stats but played=False (snap/stat conflict)",
          base.filter(pl.col("has_row"), ~pl.col("played"), _any_market_activity())
          .select("game_id", "player_id", *STAT_COLUMNS))

    status = (
        pl.when(~pl.col("played")).then(pl.lit(LabelStatus.DID_NOT_PLAY.value))
        .when(pl.col("has_row")).then(pl.lit(LabelStatus.RECORDED.value))
        .otherwise(pl.lit(LabelStatus.PARTICIPANT_NO_STATS_ROW.value))
    )
    long = pl.concat([
        base.select(
            "game_id", "player_id", "team", "played", "has_row",
            pl.lit(key).alias("market"), pl.col(m.stat_column).cast(pl.Float64).alias("stat"),
        )
        for key, m in PROP_MARKETS.items()
    ]).with_columns(status.alias("label_status"))
    actual = (
        pl.when(pl.col("label_status") == LabelStatus.RECORDED).then(pl.col("stat"))
        .when(pl.col("label_status") == LabelStatus.PARTICIPANT_NO_STATS_ROW).then(pl.lit(0.0))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    ts = pl.lit(observed_at, dtype=UTC_TS)
    labels = (
        long.join(games, on="game_id")
        .with_columns(
            actual.alias("actual"),
            pl.when(pl.col("team") == pl.col("home_team")).then("away_team")
            .otherwise("home_team").alias("opponent"),
            pl.lit(LABEL_SOURCE).alias("source"),
            pl.lit(_provenance_value([player_stats, snap_counts], "source_version"))
            .alias("source_version"),
            pl.lit(TemporalQuality.POSTGAME.value).alias("temporal_quality"),
            ts.alias("available_at"),
            ts.alias("observed_at"),
        )
        .select(list(ARTIFACT_SCHEMAS["prop_labels"]))
        .sort("season", "week", "game_id", "team", "player_id", "market")
    )
    return LabelBuild(validate_frame(labels, "prop_labels"), quarantined, incomplete,
                      snap_team_mismatches)


def read_inputs(season: int, root: Path | None = None) -> tuple[dict[str, pl.DataFrame], list]:
    """Latest snapshot per label input for a season, checksum-verified against its manifest."""
    root = root or data_root()
    frames, inputs = {}, []
    for source in LABEL_INPUTS:
        m = latest_manifest(source, season, root)
        blob = (root / m["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != m["sha256"]:
            raise LabelIntegrityError(f"checksum mismatch for {m['path']}")
        frames[source] = pl.read_parquet(blob)
        inputs.append({"source": source, "path": m["path"], "sha256": m["sha256"]})
    return frames, inputs


def build_season(season: int, raw_root: Path | None = None, skip_incomplete: bool = False
                 ) -> tuple[LabelBuild, list[dict[str, str]]]:
    frames, inputs = read_inputs(season, raw_root)
    return build_labels(**frames, skip_incomplete=skip_incomplete), inputs


def write_labels(build: LabelBuild, season: int, inputs: list[dict[str, str]],
                 root: Path | None = None) -> Manifest:
    """Append-only Parquet + manifest (with input snapshot checksums) under the labels root."""
    return write_snapshot(
        build.labels, "prop_labels", "prop_labels", [season], None, datetime.now(UTC),
        {"outcomes": TemporalQuality.POSTGAME.value}, root or labels_root(), inputs,
    )
