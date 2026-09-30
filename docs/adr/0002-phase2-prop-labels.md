# ADR 0002: Phase 2 canonical prop labels

Status: accepted, 2026-09-30

## Context

Grading, backtests and training need one reproducible ground-truth value per player, game and
market. Labels are **postgame** information: they describe what happened and must never be a
prediction-time feature. Built by `lineseek.grade.labels`; contract in
`contracts/artifacts.py` (`prop_labels`, contracts 0.5.0).

## Decisions

1. **Grain:** exactly one row per `(game_id, player_id, market)`, completed games only
   (`schedules.result` not null). `player_id` is the GSIS id; names are never keys.
2. **Markets and source columns** (nflreadpy weekly `player_stats`, the label source of record):

   | market | player_stats column | may be negative |
   |---|---|---|
   | `pass_attempts` | `attempts` | no |
   | `rush_attempts` | `carries` | no |
   | `rush_yds` | `rushing_yards` | yes |
   | `receptions` | `receptions` | no |
   | `reception_yds` | `receiving_yards` | yes |

   Market keys are the frozen Phase 0 keys (they pair with the Odds API keys in
   `PROP_MARKETS`). No fantasy, PBP, or derived substitutes.
3. **Label universe** per game: skill players (QB/RB/FB/HB/WR/TE) who have a stats row, a snap
   (offense or special teams), or a game-day roster entry (`ACT` or `INA`), plus any player of
   any position with non-zero activity in one of the five markets. Defenders and linemen with
   no market activity get no labels.
4. **Participation:** `played` comes from the existing `grade.participation.derive_played`
   ((offense_snaps + st_snaps) > 0). `actual` and `played` are independent columns, and
   `label_status` records which case applies:

   | label_status | played | actual |
   |---|---|---|
   | `recorded` | true | official stat value (may be 0) |
   | `participant_no_stats_row` | true | 0 (zero-touch participant, Phase 0 rule) |
   | `did_not_play` | false | null (never 0) |

   Grading is unchanged: the forecast track scores `did_not_play` as 0 at grading time; the
   market track voids it. A participant who recorded 0 is graded on both tracks.
5. **Null and integrity rules** (build fails with the offending keys unless stated):
   - null market stat on a stats row: fail (never coerced to 0)
   - stats row without `player_id`: quarantined (returned, reported, not labelled); fails if it
     has market activity. In 2018–2025 these are 21–22 team-level rows per season.
   - stats row with market activity but `played` false: fail (snap/stat conflict)
   - unmatched PFR id in snap counts: fail (`UnmatchedPlayerError`)
   - completed game missing stats or snap rows for a team: fail; `--skip-incomplete` excludes
     such games and lists them
   - duplicate stats `(game_id, player_id)` or schedule `game_id`: fail
   - stats season/week/team not matching the schedule: fail
6. **Team:** player_stats team, then game-day roster, then snap-count team. Stats vs roster
   disagreement fails. Snap-count team is unreliable (swapped for the 2018 Super Bowl, 24 rows)
   and disagreements are only counted. player_stats writes the 2018–19 Raiders as `LV`; it is
   mapped to the schedule's `OAK` only in games where the schedule has `OAK`.
7. **Validation** (`validate_frame(df, "prop_labels")`): exact dtypes, no nulls except
   `actual`, unique key, known markets, GSIS/game id formats, season/week and team/opponent
   consistent with `game_id`, `temporal_quality == postgame`, finite integer values, no negative
   attempts/receptions, and `label_status` consistent with `played`/`actual`.
8. **Provenance:** `source = player_stats+snap_counts`, `source_version` from the inputs (must
   agree), `observed_at = available_at =` latest `observed_at` of the input snapshots,
   `temporal_quality = postgame`. `assert_as_of` therefore rejects labels as model inputs.
9. **Inputs and reproducibility:** one season's latest snapshots of schedules, player_stats,
   snap_counts, rosters_weekly and players, each checksum-verified. Nothing from other seasons
   is joined. The nflverse `players` table is now an ingest source (UNKNOWN quality, id mapping
   only, snapshotted per season and in the weekly run) so labels rebuild from snapshots alone.
   Same inputs give byte-identical Parquet.
10. **Storage:** append-only Parquet at
    `$LINESEEK_LABELS_ROOT/prop_labels/season=YYYY/{built_at}_{sha12}.parquet` (default
    `data/labels`, git-ignored) with a manifest listing the input snapshots and checksums.
    This departs from ADR 0001's DuckDB plan for derived tables; DuckDB reads these files
    directly.
11. **CLI:** `lineseek-labels build --season 2025` (or `2018-2025`, `--skip-incomplete`) and
    `lineseek-labels validate <file.parquet>` (contract + manifest checksum).

## Consequences

- Contract 0.5.0: `prop_labels` gained `season`, `week`, `opponent`, `label_status` and domain
  validation; `PropMarket` gained `nonnegative`; `LabelStatus` added; `players` and
  `prop_labels` temporal qualities added.
- Override `CartNa00` now maps to `00-0040547` (it was excluded, which made two games with
  carries look like non-participation).
- Downstream projections for players outside the label universe need a universe extension.
