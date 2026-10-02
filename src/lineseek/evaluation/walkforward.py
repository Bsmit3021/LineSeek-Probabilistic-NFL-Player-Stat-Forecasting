"""Walk-forward (rolling-origin) folds on exact decision_time; docs/adr/0004.

One fold per unique decision_time of the evaluation seasons. Test rows: decision_time equal to
the fold's. Training rows: any row whose outcome was available by then, i.e.
kickoff_utc + POSTGAME_STATS_LAG <= decision_time (the Phase 3 rule,
`postgame_stats_available_at`). Week numbers are never used, rows are never shuffled, and the
training set only grows. Phase 4B models must use these folds unchanged.
"""

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from lineseek.contracts import validate_frame
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS
from lineseek.contracts.temporal import POSTGAME_STATS_LAG

ROW_COLUMNS = ("game_id", "season", "week", "kickoff_utc", "decision_time", "outcome_admitted")


class FoldIntegrityError(ValueError):
    pass


def outcome_available_at() -> pl.Expr:
    """Vectorised `postgame_stats_available_at(kickoff_utc)`."""
    return (pl.col("kickoff_utc") + POSTGAME_STATS_LAG).alias("outcome_available_at")


def train_rows_filter(decision_time: datetime) -> pl.Expr:
    return pl.col("kickoff_utc") + POSTGAME_STATS_LAG <= pl.lit(decision_time)


def evaluation_rows_filter(decision_time: datetime) -> pl.Expr:
    return pl.col("decision_time") == pl.lit(decision_time)


@dataclass(frozen=True)
class Split:
    fold_id: int
    decision_time: datetime
    train: pl.DataFrame
    test: pl.DataFrame


def split(rows: pl.DataFrame, fold: dict) -> Split:
    """Train/test rows of one fold (a row of walk_forward_folds), with invariants enforced."""
    dt = fold["decision_time"]
    train, test = rows.filter(train_rows_filter(dt)), rows.filter(evaluation_rows_filter(dt))
    if train.height and train.select((pl.col("kickoff_utc") + POSTGAME_STATS_LAG).max()).item() > dt:
        raise FoldIntegrityError(f"fold {fold['fold_id']}: training outcome after decision_time")
    if test.height and test["kickoff_utc"].min() <= dt:
        raise FoldIntegrityError(f"fold {fold['fold_id']}: test kickoff not after decision_time")
    return Split(fold["fold_id"], dt, train, test)


def walk_forward_folds(rows: pl.DataFrame, eval_seasons: tuple[int, ...]) -> pl.DataFrame:
    """Fold table from all modelling rows (every season, warm-up included).

    `rows` needs ROW_COLUMNS, one row per (game_id, player_id, market); `outcome_admitted` marks
    the F1 cohort so counts can be reported, it does not change the folds.
    """
    rows = rows.select(ROW_COLUMNS).with_columns(outcome_available_at())
    test = rows.filter(pl.col("season").is_in(list(eval_seasons)))
    per = test.group_by("decision_time").agg(
        pl.col("season").unique().alias("_seasons"),
        pl.col("week").min().alias("first_week"), pl.col("week").max().alias("last_week"),
        pl.col("game_id").n_unique().alias("test_games"), pl.len().alias("test_rows"),
        (~pl.col("outcome_admitted")).sum().alias("test_rows_primary"),
        pl.col("outcome_admitted").sum().alias("test_rows_outcome_admitted"),
    ).sort("decision_time")
    multi = per.filter(pl.col("_seasons").list.len() != 1)
    if multi.height:
        raise FoldIntegrityError(f"decision_time spanning seasons: {multi['decision_time'].to_list()}")
    folds = []
    for i, r in enumerate(per.iter_rows(named=True)):
        dt = r["decision_time"]
        avail = rows.filter(pl.col("outcome_available_at") <= dt)
        started = rows.filter(pl.col("kickoff_utc") < dt, pl.col("outcome_available_at") > dt)
        folds.append({
            "fold_id": i, "decision_time": dt, "season": r["_seasons"][0],
            **{k: r[k] for k in ("first_week", "last_week", "test_games", "test_rows",
                                 "test_rows_primary", "test_rows_outcome_admitted")},
            "train_rows": avail.height,
            "train_rows_primary": avail.filter(~pl.col("outcome_admitted")).height,
            "train_first_kickoff": avail["kickoff_utc"].min(),
            "train_last_available_at": avail["outcome_available_at"].max(),
            "started_unavailable_rows": started.height,
        })
    schema = ARTIFACT_SCHEMAS["walk_forward_folds"]
    return validate_frame(pl.DataFrame(folds, schema=schema), "walk_forward_folds")
