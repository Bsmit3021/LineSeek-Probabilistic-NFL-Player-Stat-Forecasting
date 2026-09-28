"""Pull orchestration: one-time history backfill and the weekly current-season snapshot."""

import logging
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from lineseek.contracts.raw import schema_variant
from lineseek.ingest.snapshot import Manifest, list_manifests, read_latest, write_snapshot
from lineseek.ingest.sources import SOURCES, Source

log = logging.getLogger(__name__)

HISTORY_SEASONS = range(2018, 2027)
WEEKLY_SOURCES = ("schedules", "injuries", "depth_charts", "rosters_weekly", "player_stats",
                  "snap_counts")
RETRIES = 3


def _load(source: Source, season: int) -> pl.DataFrame:
    """Download with retries on transient network errors (nflverse release downloads flake)."""
    for attempt in range(1, RETRIES + 1):
        try:
            return source.loader(season)
        except OSError as e:
            if attempt == RETRIES:
                raise
            log.warning("%s %s attempt %d failed: %s", source.name, season, attempt, e)
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def pull(
    name: str,
    season: int,
    *,
    weeks: list[int] | None = None,
    schedules: pl.DataFrame | None = None,
    completed_game_ids: Iterable[str] | None = None,
    root: Path | None = None,
) -> tuple[Manifest, pl.DataFrame]:
    source = SOURCES[name]
    pulled_at = datetime.now(UTC)
    df = source.build(_load(source, season), season, pulled_at, schedules)
    if completed_game_ids is not None and source.completed_games_only:
        df = df.filter(pl.col("game_id").is_in(list(completed_game_ids)))
    manifest = write_snapshot(
        df, name, schema_variant(name, season), [season], weeks, pulled_at,
        source.temporal_quality(season), root,
    )
    log.info("%s %s: %d rows -> %s", name, season, manifest.row_count, manifest.path)
    return manifest, df


def backfill(
    seasons: Iterable[int] = HISTORY_SEASONS,
    names: Iterable[str] = tuple(SOURCES),
    force: bool = False,
    root: Path | None = None,
) -> list[Manifest]:
    """Snapshot each (source, season) once; skips pairs already on disk unless force=True."""
    names = list(names)
    out: list[Manifest] = []
    def todo(name: str, season: int) -> bool:
        return force or not list_manifests(name, season, root)

    for season in seasons:
        schedules = None
        # Injuries depend on schedules: reuse the stored snapshot or pull it first.
        if "schedules" in names or "injuries" in names:
            if todo("schedules", season):
                manifest, schedules = pull("schedules", season, root=root)
                out.append(manifest)
            else:
                schedules = read_latest("schedules", season, root)
        for name in names:
            if name != "schedules" and todo(name, season):
                out.append(pull(name, season, schedules=schedules, root=root)[0])
    return out


def weekly(season: int, week: int, root: Path | None = None) -> list[Manifest]:
    """Snapshot the current week; stats and snaps keep completed games only."""
    out: list[Manifest] = []
    manifest, schedules = pull("schedules", season, weeks=[week], root=root)
    out.append(manifest)
    completed = schedules.filter(pl.col("result").is_not_null())["game_id"].to_list()
    for name in WEEKLY_SOURCES[1:]:
        m, _ = pull(name, season, weeks=[week], schedules=schedules,
                    completed_game_ids=completed, root=root)
        out.append(m)
    return out
