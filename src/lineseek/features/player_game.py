"""Leakage-safe decision-time features: one row per (game_id, player_id, market, decision_time).

decision_time = forecast_time_utc(kickoff). A completed game feeds a later row only if its
stats were available (kickoff + POSTGAME_STATS_LAG) at or before that row's decision_time, so
the target game and anything after the lock can never contribute. The only other input is the
pre-published schedule. Definitions, windows and missing-data rules: docs/adr/0003.
"""

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from lineseek.contracts import (
    PLAYER_ID_RE,
    PROP_MARKETS,
    ContractViolation,
    forecast_time_utc,
    validate_frame,
)
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.contracts.temporal import POSTGAME_STATS_LAG
from lineseek.grade.labels import (
    FRANCHISE_CODES,
    STAT_COLUMNS,
    _fail,
    assign_teams,
    check_stats,
    correct_snap_teams,
    labels_root,
    normalize_stats_teams,
    read_inputs,
    roster_game_teams,
    team_games,
)
from lineseek.grade.participation import build_pfr_gsis_map, load_overrides
from lineseek.ingest.pipeline import HISTORY_SEASONS
from lineseek.ingest.snapshot import Manifest, latest_manifest, write_snapshot

FEATURE_SET_VERSION = "1.0.0"
FEATURES_ROOT_ENV = "LINESEEK_FEATURES_ROOT"
DEFAULT_FEATURES_ROOT = "data/features"
ARTIFACT = "player_game_features"
MARKET_WINDOWS = (3, 5, 8)
SNAP_WINDOWS = (3, 5)
RECENT = 5  # opportunity, team and opponent window
PRIOR_SEASONS = 1  # rolling windows may reach back into the previous season only
HISTORY_STATS = (*STAT_COLUMNS, "targets")
OPPORTUNITY = {"pass_att": "attempts", "carries": "carries", "targets": "targets"}
SHARES = {"pass_att": "pass_att_share_roll5", "carries": "carries_share_roll5",
          "targets": "target_share_roll5"}
LABEL_KEY_COLUMNS = ("game_id", "player_id", "team", "market")
HISTORY_META = ("game_id", "season", "week", "kickoff_utc", "available_at")
F64, I32 = pl.Float64, pl.Int32

FEATURE_COLUMNS: dict[str, pl.DataType] = {
    "prior_games_n": I32,
    "market_prev": F64,
    **{c: t for k in MARKET_WINDOWS
       for c, t in ((f"market_roll{k}_mean", F64), (f"market_roll{k}_n", I32))},
    "market_season_mean": F64,
    "market_season_median": F64,
    "market_season_std": F64,
    "market_season_games": I32,
    **{c: F64 for name in OPPORTUNITY
       for c in (f"{name}_prev", f"{name}_roll5_mean", SHARES[name])},
    "receptions_roll5_mean": F64,
    "snap_pct_prev": F64,
    **{c: t for k in SNAP_WINDOWS
       for c, t in ((f"snap_pct_roll{k}_mean", F64), (f"snap_pct_roll{k}_n", I32))},
    "team_games_n": I32,
    "team_pass_att_roll5_mean": F64,
    "team_carries_roll5_mean": F64,
    "team_plays_roll5_mean": F64,
    "team_pass_rate_roll5": F64,
    "team_rush_rate_roll5": F64,
    "opp_market_allowed_roll5_mean": F64,
    "opp_allowed_roll5_n": I32,
    "opp_market_allowed_season_mean": F64,
    "opp_allowed_season_games": I32,
    "is_home": pl.Boolean,
    "is_postseason": pl.Boolean,
    "team_rest_days": F64,
    "opp_rest_days": F64,
    "days_since_prev_game": F64,
}
COUNT_COLUMNS = [c for c, t in FEATURE_COLUMNS.items() if t == I32]
PARAMETERS = {
    "feature_set_version": FEATURE_SET_VERSION,
    "decision_time": "forecast_time_utc(kickoff): latest Wednesday 12:00 America/New_York "
                     "strictly before kickoff",
    "prior_game_available_at": f"kickoff_utc + {POSTGAME_STATS_LAG}",
    "market_windows": ",".join(map(str, MARKET_WINDOWS)),
    "snap_windows": ",".join(map(str, SNAP_WINDOWS)),
    "recent_window": str(RECENT),
    "prior_seasons_in_rolling_windows": str(PRIOR_SEASONS),
    "season_to_date": "target season only",
}


class FeatureIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class SeasonHistory:
    """Completed games of one season as eligible feature inputs (rows carry available_at)."""

    season: int
    # One row per player appearance (played, Phase 2 rule), with the player's team.
    appearances: pl.DataFrame
    # One row per (game_id, team): team totals and totals allowed (the other team's totals).
    team_games: pl.DataFrame
    # Completed games missing player_stats for a team: excluded from all history.
    skipped_games: tuple[str, ...]
    # Completed games missing snap counts for a team: kept, appearance = stats row, snap null.
    games_without_snaps: tuple[str, ...]


@dataclass(frozen=True)
class FeatureBuild:
    features: pl.DataFrame
    skipped_history_games: tuple[str, ...]
    history_games_without_snaps: tuple[str, ...]


def features_root() -> Path:
    return Path(os.environ.get(FEATURES_ROOT_ENV, DEFAULT_FEATURES_ROOT))


def _err(msg: str, df: pl.DataFrame) -> None:
    _fail(msg, df, FeatureIntegrityError)


def season_history(
    schedules: pl.DataFrame,
    player_stats: pl.DataFrame,
    snap_counts: pl.DataFrame,
    rosters_weekly: pl.DataFrame,
    players: pl.DataFrame,
    overrides: pl.DataFrame | None = None,
) -> SeasonHistory:
    """Build one season's appearance and team-game history from its snapshot frames."""
    seasons = schedules["season"].unique().to_list()
    if len(seasons) != 1:
        raise FeatureIntegrityError(f"schedules must cover one season, got {seasons}")
    games = schedules.filter(pl.col("result").is_not_null()).select(
        "game_id", "season", "week", "home_team", "away_team", "kickoff_utc")
    _err("duplicate schedule game_id", games.filter(pl.col("game_id").is_duplicated()))
    activity = pl.any_horizontal(pl.col(c) != 0 for c in HISTORY_STATS)
    _err("player_stats rows without player_id but with market/target stats",
         player_stats.filter(pl.col("player_id").is_null(), activity).select("game_id", "team"))
    ps = normalize_stats_teams(player_stats.filter(pl.col("player_id").is_not_null()), games)
    check_stats(ps, games, FeatureIntegrityError, HISTORY_STATS)
    _err("player_stats rows with a non-GSIS player_id",
         ps.filter(~pl.col("player_id").str.contains(PLAYER_ID_RE.pattern)).select("player_id"))

    sn = correct_snap_teams(snap_counts)
    tg = team_games(games).select("game_id", "team")

    def missing(src: pl.DataFrame) -> set[str]:
        present = src.select("game_id", "team").unique()
        return set(tg.join(present, on=["game_id", "team"], how="anti")["game_id"])

    skipped = sorted(missing(ps))
    no_snaps = sorted(missing(sn) - set(skipped))
    games = games.filter(~pl.col("game_id").is_in(skipped))
    kept = games["game_id"].implode()
    ps = ps.filter(pl.col("game_id").is_in(kept))
    sn = sn.filter(pl.col("game_id").is_in(kept))

    id_map = build_pfr_gsis_map(sn, players, rosters_weekly,
                                load_overrides() if overrides is None else overrides)
    snm = sn.join(id_map.drop_nulls("gsis_id"), on="pfr_player_id").rename(
        {"gsis_id": "player_id"})
    _err("several snap rows for one player in a game",
         snm.filter(pl.struct("game_id", "player_id").is_duplicated()).select("game_id",
                                                                             "player_id"))
    snaps = snm.select(
        "game_id", "player_id",
        ((pl.col("offense_snaps").fill_null(0) + pl.col("st_snaps").fill_null(0)) > 0)
        .alias("played"),
        pl.col("offense_pct").alias("snap_pct"),
    )
    stats = ps.select("game_id", "player_id", *HISTORY_STATS, pl.lit(True).alias("has_row"))
    cand = stats.join(snaps, on=["game_id", "player_id"], how="full", coalesce=True).with_columns(
        pl.col("has_row").fill_null(False), pl.col("played").fill_null(False))
    covered = ~pl.col("game_id").is_in(no_snaps)
    _err("players with stats but no offensive/special-teams snaps (snap/stat conflict)",
         cand.filter(covered, pl.col("has_row"), ~pl.col("played"), activity)
         .select("game_id", "player_id"))
    app = cand.filter(pl.when(covered).then(pl.col("played")).otherwise(pl.col("has_row")))
    app = app.with_columns(
        *[pl.col(c).fill_null(0) for c in HISTORY_STATS],  # played, no stats row: 0 (Phase 2)
        pl.when(covered).then(pl.col("snap_pct")).otherwise(None).alias("snap_pct"),
    )
    app, _ = assign_teams(app.select("game_id", "player_id", *HISTORY_STATS, "snap_pct"), ps,
                          roster_game_teams(rosters_weekly, games), snm, FeatureIntegrityError)

    meta = games.select("game_id", "season", "week", "kickoff_utc",
                        (pl.col("kickoff_utc") + POSTGAME_STATS_LAG).alias("available_at"))
    totals = ps.group_by("game_id", "team").agg(pl.col(c).sum() for c in HISTORY_STATS)
    allowed = (
        totals.join(totals, on="game_id", suffix="_opp")
        .filter(pl.col("team") != pl.col("team_opp"))
        .select("game_id", "team", *[pl.col(f"{c}_opp").alias(f"allowed_{c}")
                                     for c in HISTORY_STATS])
    )
    team_hist = (
        totals.rename({c: f"team_{c}" for c in HISTORY_STATS})
        .join(allowed, on=["game_id", "team"]).join(meta, on="game_id")
    )
    app = app.join(meta, on="game_id").join(
        team_hist.select("game_id", "team", *[f"team_{c}" for c in HISTORY_STATS]),
        on=["game_id", "team"], how="left")
    _err("appearances whose team has no totals in that game",
         app.filter(pl.col("team_attempts").is_null()).select("game_id", "player_id", "team"))
    return SeasonHistory(seasons[0], app, team_hist, tuple(skipped), tuple(no_snaps))


def _window(targets: pl.DataFrame, history: pl.DataFrame, on: str, group: list[str]
            ) -> pl.DataFrame:
    """History rows eligible for each target: same `on` key (player or franchise), available
    at or before decision_time, from the target season or PRIOR_SEASONS before it.
    rank 1 = most recent."""
    h = history.rename({c: f"h_{c}" for c in HISTORY_META})
    return (
        targets.join(h, on=on)
        .filter(pl.col("h_available_at") <= pl.col("decision_time"),
                pl.col("h_season").is_between(pl.col("season") - PRIOR_SEASONS, pl.col("season")))
        .with_columns(pl.col("h_kickoff_utc").rank("ordinal", descending=True).over(group)
                      .alias("rank"))
    )


def _recent(k: int) -> pl.Expr:
    return pl.col("rank") <= k


def _ratio(num: str, den: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den))


def _player_features(targets: pl.DataFrame, appearances: pl.DataFrame) -> pl.DataFrame:
    this_season = pl.col("h_season") == pl.col("season")
    aggs = [
        pl.len().alias("prior_games_n"),
        pl.col("h_available_at").max().alias("_player_available_at"),
        pl.col("h_kickoff_utc").max().alias("_last_kickoff"),
        this_season.sum().alias("market_season_games"),
        *[_recent(k).sum().alias(f"market_roll{k}_n") for k in MARKET_WINDOWS],
        pl.col("receptions").cast(F64).filter(_recent(RECENT)).mean()
        .alias("receptions_roll5_mean"),
        pl.col("snap_pct").filter(pl.col("rank") == 1).first().alias("snap_pct_prev"),
    ]
    for k in SNAP_WINDOWS:
        snap = pl.col("snap_pct").filter(_recent(k))
        aggs += [snap.mean().alias(f"snap_pct_roll{k}_mean"),
                 snap.is_not_null().sum().alias(f"snap_pct_roll{k}_n")]
    for c in STAT_COLUMNS:
        v = pl.col(c).cast(F64)
        aggs += [
            v.filter(pl.col("rank") == 1).first().alias(f"{c}__prev"),
            *[v.filter(_recent(k)).mean().alias(f"{c}__roll{k}_mean") for k in MARKET_WINDOWS],
            v.filter(this_season).mean().alias(f"{c}__season_mean"),
            v.filter(this_season).median().alias(f"{c}__season_median"),
            v.filter(this_season).std(ddof=1).alias(f"{c}__season_std"),
        ]
    for name, c in OPPORTUNITY.items():
        v = pl.col(c).cast(F64)
        aggs += [
            v.filter(pl.col("rank") == 1).first().alias(f"{name}_prev"),
            v.filter(_recent(RECENT)).mean().alias(f"{name}_roll5_mean"),
            v.filter(_recent(RECENT)).sum().alias(f"_{name}_num"),
            pl.col(f"team_{c}").cast(F64).filter(_recent(RECENT)).sum().alias(f"_{name}_den"),
        ]
    group = ["game_id", "player_id"]
    return (
        _window(targets.select(*group, "season", "decision_time"), appearances, "player_id",
                group)
        .group_by(group).agg(aggs)
        .with_columns(*[_ratio(f"_{n}_num", f"_{n}_den").alias(SHARES[n]) for n in OPPORTUNITY])
    )


def _franchise(col: str) -> pl.Expr:
    return pl.col(col).replace(FRANCHISE_CODES).alias("franchise")


def _team_features(targets: pl.DataFrame, team_hist: pl.DataFrame) -> pl.DataFrame:
    group = ["game_id", "team"]
    t = pl.col("team_attempts").cast(F64).filter(_recent(RECENT))
    r = pl.col("team_carries").cast(F64).filter(_recent(RECENT))
    tgt = targets.select(*group, _franchise("team"), "season", "decision_time").unique()
    return (
        _window(tgt, team_hist.with_columns(_franchise("team")).drop("team"), "franchise", group)
        .group_by(group).agg(
            _recent(RECENT).sum().alias("team_games_n"),
            pl.col("h_available_at").max().alias("_team_available_at"),
            t.mean().alias("team_pass_att_roll5_mean"),
            r.mean().alias("team_carries_roll5_mean"),
            (t + r).mean().alias("team_plays_roll5_mean"),
            t.sum().alias("_att"), r.sum().alias("_car"), (t + r).sum().alias("_plays"),
        )
        .with_columns(_ratio("_att", "_plays").alias("team_pass_rate_roll5"),
                      _ratio("_car", "_plays").alias("team_rush_rate_roll5"))
    )


def _opponent_features(targets: pl.DataFrame, team_hist: pl.DataFrame) -> pl.DataFrame:
    group = ["game_id", "team"]
    opp = targets.select("game_id", pl.col("opponent").alias("team"), _franchise("opponent"),
                         "season", "decision_time").unique()
    this_season = pl.col("h_season") == pl.col("season")
    aggs = [
        _recent(RECENT).sum().alias("opp_allowed_roll5_n"),
        this_season.sum().alias("opp_allowed_season_games"),
        pl.col("h_available_at").max().alias("_opp_available_at"),
    ]
    for c in STAT_COLUMNS:
        v = pl.col(f"allowed_{c}").cast(F64)
        aggs += [v.filter(_recent(RECENT)).mean().alias(f"opp_allowed_{c}__roll5_mean"),
                 v.filter(this_season).mean().alias(f"opp_allowed_{c}__season_mean")]
    history = team_hist.with_columns(_franchise("team")).drop("team")
    return (
        _window(opp, history, "franchise", group).group_by(group).agg(aggs)
        .rename({"team": "opponent"})
    )


def _rest_days(schedules: pl.DataFrame) -> pl.DataFrame:
    """Days since each team's previous scheduled game in the same season (null for its first)."""
    cal = team_games(schedules.select("game_id", "season", "week", "home_team", "away_team"))
    cal = cal.join(schedules.select("game_id", "kickoff_utc"), on="game_id").sort(
        "team", "kickoff_utc", "game_id")
    prev = pl.col("kickoff_utc").shift(1).over("team")
    return cal.select("game_id", "team",
                      ((pl.col("kickoff_utc") - prev).dt.total_seconds() / 86400).alias("rest"))


def build_features(
    season: int,
    keys: pl.DataFrame,
    histories: Mapping[int, SeasonHistory],
    schedules: pl.DataFrame,
) -> FeatureBuild:
    """Features for target keys (game_id, player_id, team, market) of one season.

    Only key columns are read from `keys`; label values never enter. Uses the histories of
    `season` and the PRIOR_SEASONS before it; any other season in `histories` is ignored.
    """
    keys = keys.select(LABEL_KEY_COLUMNS)
    _err("duplicate target keys (game_id, player_id, market)",
         keys.filter(pl.struct("game_id", "player_id", "market").is_duplicated()))
    _err("target keys with an unsupported market",
         keys.filter(~pl.col("market").is_in(list(PROP_MARKETS))))
    if season not in histories:
        raise FeatureIntegrityError(f"history for season {season} is required")
    used = [histories[s] for s in range(season - PRIOR_SEASONS, season + 1) if s in histories]
    sched = schedules.select("game_id", "season", "week", "game_type", "home_team", "away_team",
                             "kickoff_utc")
    _err("schedule rows from another season", sched.filter(pl.col("season") != season))

    tp = keys.select("game_id", "player_id", "team").unique()
    _err("players with several teams in a game",
         tp.filter(pl.struct("game_id", "player_id").is_duplicated()))
    tp = tp.join(sched, on="game_id", how="left")
    _err("target games missing from the season schedule",
         tp.filter(pl.col("kickoff_utc").is_null()).select("game_id"))
    _err("target team not playing in the game",
         tp.filter((pl.col("team") != pl.col("home_team")) & (pl.col("team") != pl.col("away_team")))
         .select("game_id", "team"))
    lock = {k: forecast_time_utc(k) for k in tp["kickoff_utc"].unique().to_list()}
    tp = tp.with_columns(
        pl.col("kickoff_utc").replace_strict(lock, return_dtype=pl.Datetime("us", "UTC"))
        .alias("decision_time"),
        pl.when(pl.col("team") == pl.col("home_team")).then("away_team").otherwise("home_team")
        .alias("opponent"),
        (pl.col("team") == pl.col("home_team")).alias("is_home"),
        (pl.col("game_type") != "REG").alias("is_postseason"),
    )

    appearances = pl.concat([h.appearances for h in used])
    team_hist = pl.concat([h.team_games for h in used])
    rest = _rest_days(schedules)
    wide = (
        keys.join(tp, on=["game_id", "player_id", "team"])
        .join(_player_features(tp, appearances), on=["game_id", "player_id"], how="left")
        .join(_team_features(tp, team_hist), on=["game_id", "team"], how="left")
        .join(_opponent_features(tp, team_hist), on=["game_id", "opponent"], how="left")
        .join(rest.rename({"rest": "team_rest_days"}), on=["game_id", "team"], how="left")
        .join(rest.rename({"team": "opponent", "rest": "opp_rest_days"}),
              on=["game_id", "opponent"], how="left")
        .with_columns(
            *[pl.col(c).fill_null(0).cast(I32) for c in COUNT_COLUMNS],
            ((pl.col("kickoff_utc") - pl.col("_last_kickoff")).dt.total_seconds() / 86400)
            .alias("days_since_prev_game"),
        )
    )
    per_market = []
    for key, m in PROP_MARKETS.items():
        c = m.stat_column
        per_market.append(wide.filter(pl.col("market") == key).with_columns(
            pl.col(f"{c}__prev").alias("market_prev"),
            *[pl.col(f"{c}__roll{k}_mean").alias(f"market_roll{k}_mean")
              for k in MARKET_WINDOWS],
            *[pl.col(f"{c}__season_{s}").alias(f"market_season_{s}")
              for s in ("mean", "median", "std")],
            pl.col(f"opp_allowed_{c}__roll5_mean").alias("opp_market_allowed_roll5_mean"),
            pl.col(f"opp_allowed_{c}__season_mean").alias("opp_market_allowed_season_mean"),
        ))
    max_available = pl.max_horizontal("_player_available_at", "_team_available_at",
                                      "_opp_available_at")
    out = (
        pl.concat(per_market)
        .with_columns(
            max_available.alias("max_available_at"),
            pl.when(max_available.is_null()).then(pl.lit("exact")).otherwise(pl.lit("inferred"))
            .alias("worst_temporal_quality"),
            pl.lit(FEATURE_SET_VERSION).alias("feature_set_version"),
        )
        .select([*ARTIFACT_SCHEMAS["features"], *FEATURE_COLUMNS])
        .sort("season", "week", "game_id", "team", "player_id", "market")
    )
    return FeatureBuild(
        validate_features(out),
        tuple(g for h in used for g in h.skipped_games),
        tuple(g for h in used for g in h.games_without_snaps),
    )


def validate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Contract checks plus the exact column set and value rules of this feature-set version."""
    expected = {**ARTIFACT_SCHEMAS["features"], **FEATURE_COLUMNS}
    if list(df.schema.items()) != list(expected.items()):
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        wrong = {c: str(df.schema[c]) for c, t in expected.items()
                 if c in df.columns and df.schema[c] != t}
        raise ContractViolation(f"features v{FEATURE_SET_VERSION}: missing={missing} "
                                f"extra={extra} wrong_dtype={wrong} (or column order differs)")
    validate_frame(df, "features")
    nonneg_market = pl.col("market").is_in([k for k, m in PROP_MARKETS.items() if m.nonnegative])
    market_values = ["market_prev", *[f"market_roll{k}_mean" for k in MARKET_WINDOWS],
                     "market_season_mean", "market_season_median",
                     "opp_market_allowed_roll5_mean", "opp_market_allowed_season_mean"]
    counts = ([f"{name}_prev" for name in OPPORTUNITY]
              + [f"{name}_roll5_mean" for name in OPPORTUNITY]
              + ["receptions_roll5_mean", "team_pass_att_roll5_mean", "team_carries_roll5_mean",
                 "team_plays_roll5_mean", "market_season_std"])
    unit = [*SHARES.values(), "snap_pct_prev", *[f"snap_pct_roll{k}_mean" for k in SNAP_WINDOWS],
            "team_pass_rate_roll5", "team_rush_rate_roll5"]
    floats = [c for c, t in FEATURE_COLUMNS.items() if t == F64]
    prior, season_n = pl.col("prior_games_n"), pl.col("market_season_games")
    checks = {
        "non-finite feature value": pl.any_horizontal(~pl.col(c).is_finite() for c in floats),
        "negative count feature": pl.any_horizontal(pl.col(c) < 0 for c in counts),
        "negative value for a count market": nonneg_market
        & pl.any_horizontal(pl.col(c) < 0 for c in market_values),
        "share/rate/snap pct outside [0, 1]": pl.any_horizontal(
            ~pl.col(c).is_between(0, 1) for c in unit),
        "non-positive rest/days": pl.any_horizontal(
            pl.col(c) <= 0 for c in ("team_rest_days", "opp_rest_days", "days_since_prev_game")),
        "rolling n not min(window, prior_games_n)": pl.any_horizontal(
            pl.col(f"market_roll{k}_n") != pl.min_horizontal(prior, pl.lit(k))
            for k in MARKET_WINDOWS),
        "season games outside [0, prior_games_n]": (season_n < 0) | (season_n > prior),
        "snap n outside [0, games in window]": pl.any_horizontal(
            (pl.col(f"snap_pct_roll{k}_n") < 0)
            | (pl.col(f"snap_pct_roll{k}_n") > pl.col(f"market_roll{k}_n"))
            for k in SNAP_WINDOWS),
        "team/opponent n outside [0, window]": ~pl.col("team_games_n").is_between(0, RECENT)
        | ~pl.col("opp_allowed_roll5_n").is_between(0, RECENT),
        "value present without history (or missing with history)": (
            (pl.col("market_prev").is_null() != (prior == 0))
            | (pl.col("market_season_mean").is_null() != (season_n == 0))
            | pl.any_horizontal((pl.col(f"snap_pct_roll{k}_mean").is_null()
                                 != (pl.col(f"snap_pct_roll{k}_n") == 0)) for k in SNAP_WINDOWS)
            | (pl.col("team_pass_att_roll5_mean").is_null() != (pl.col("team_games_n") == 0))
            | (pl.col("opp_market_allowed_roll5_mean").is_null()
               != (pl.col("opp_allowed_roll5_n") == 0))
        ),
        "worst_temporal_quality inconsistent with max_available_at": (
            pl.col("max_available_at").is_null() != (pl.col("worst_temporal_quality") == "exact")
        ),
    }
    counts_df = df.select(e.fill_null(False).sum().alias(k) for k, e in checks.items())
    problems = [f"{n} row(s) {k}" for k, n in counts_df.row(0, named=True).items() if n]
    if problems:
        raise ContractViolation(f"features v{FEATURE_SET_VERSION}: " + "; ".join(problems))
    return df


def read_label_keys(season: int, root: Path | None = None) -> tuple[pl.DataFrame, dict]:
    """Target keys from the latest prop_labels artifact (checksum-verified); keys only."""
    root = root or labels_root()
    m = latest_manifest("prop_labels", season, root)
    blob = (root / m["path"]).read_bytes()
    if hashlib.sha256(blob).hexdigest() != m["sha256"]:
        raise FeatureIntegrityError(f"checksum mismatch for {m['path']}")
    keys = pl.read_parquet(blob, columns=list(LABEL_KEY_COLUMNS))
    return keys, {"source": "prop_labels", "path": m["path"], "sha256": m["sha256"]}


def history_seasons(season: int) -> list[int]:
    """Seasons whose history a build of `season` reads; none before the first ingested one."""
    return [s for s in range(season - PRIOR_SEASONS, season + 1) if s >= HISTORY_SEASONS.start]


def build_season(season: int, raw_root: Path | None = None, label_root: Path | None = None
                 ) -> tuple[FeatureBuild, list[dict[str, str]]]:
    histories, inputs = {}, []
    for s in history_seasons(season):
        frames, ins = read_inputs(s, raw_root)
        histories[s] = season_history(**frames)
        inputs += ins
        if s == season:
            schedules = frames["schedules"]
    keys, label_input = read_label_keys(season, label_root)
    return build_features(season, keys, histories, schedules), [*inputs, label_input]


def write_features(build: FeatureBuild, season: int, inputs: list[dict[str, str]],
                   root: Path | None = None) -> Manifest:
    """Append-only Parquet + manifest (inputs, checksums, feature-set parameters)."""
    return write_snapshot(
        build.features, ARTIFACT, f"{ARTIFACT}/v{FEATURE_SET_VERSION}", [season], None,
        datetime.now(UTC), {"prior_game_stats": "inferred", "schedule_calendar": "exact"},
        root or features_root(), inputs, PARAMETERS,
    )


def summarize(df: pl.DataFrame) -> dict[str, object]:
    key = ["market_prev", "market_roll3_mean", "market_season_mean", "snap_pct_roll3_mean",
           "team_pass_att_roll5_mean", "opp_market_allowed_roll5_mean", "team_rest_days",
           "days_since_prev_game"]
    return {
        "rows": df.height,
        "players": df["player_id"].n_unique(),
        "games": df["game_id"].n_unique(),
        "markets": df["market"].n_unique(),
        "cold_start_rows (no prior games)": df.filter(pl.col("prior_games_n") == 0).height,
        "rows with <3 prior games": df.filter(pl.col("prior_games_n") < 3).height,
        "null_counts": {c: df[c].null_count() for c in key},
        "leakage_violations": df.filter(pl.col("max_available_at") > pl.col("decision_time"))
        .height,
    }
