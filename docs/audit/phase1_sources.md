# Phase 1 source audit (nflreadpy 0.1.5, audited 2026-09-27)

Scope: schedules, play-by-play, player stats, rosters (weekly), depth charts, injuries.
Seasons checked: 2024, 2025 (full) and 2026 (in progress, week 3). Reproduce with
`python scripts/audit_phase1.py`.

| Source | Key | Integrity | Timestamp fields | Temporal quality |
|---|---|---|---|---|
| `load_schedules` | `game_id` | unique; 285 games/season; 0 null `gametime` | `gameday` + `gametime` in **US/Eastern local** (verified vs Odds API: 13:00 ET = 17:00Z) | kickoff EXACT; scores POSTGAME; lines UNKNOWN |
| `load_player_stats` (week) | `(player_id, game_id)` | unique; 44 null `player_id` rows (2024-25) | none | POSTGAME (label source of record) |
| `load_pbp` | `(game_id, play_id)` | gsis ids per role | `time_of_day` (UTC ISO per play), `start_time` (local string) | POSTGAME |
| `load_rosters_weekly` | `(gsis_id, season, week)` | 8 duplicate keys, 25 null `gsis_id` | none (week granularity only) | UNKNOWN |
| `load_depth_charts` ≤2024 | `(season, week, club_code, gsis_id, ...)` | weekly | none | UNKNOWN |
| `load_depth_charts` ≥2025 | `(dt, team, gsis_id, pos_abb)` | 221 snapshots in 2025 (Aug 3 → Mar 14) | `dt` ISO UTC **string** | EXACT (snapshot) |
| `load_injuries` 2024 | `(gsis_id, season, week)` | 2 duplicate keys | `date_modified` UTC (last modified) | EXACT (upper bound) |
| `load_injuries` 2025 | same | | `date_modified` all null | INFERRED |
| `load_injuries` 2026 | same | | `date_modified` column absent | INFERRED |

## Findings that affect ingestion

1. **Schema drift in depth charts.** 2025+ is a different table (timestamped snapshots, no
   season/week). `load_depth_charts([2024, 2025])` silently concatenates both with nulls.
   Always load per season and branch on schema.
2. **Injury timestamps disappeared after 2024.** For 2025+ use
   `contracts.temporal.injury_report_available_at` (16:00 ET the day before kickoff), which is
   never earlier than the NFL's final-report publication time. `season_type` is null in 2024;
   use `game_type`.
3. **Labels come from player_stats, not PBP.** Receiving-yard totals differ for 14/502 receivers
   in 2025 REG (laterals, stat corrections).
4. **Schedule lines are not usable as market data.** No capture time; treat as UNKNOWN.
   Timestamped lines come from the odds provider.
5. **Duplicates/nulls must be handled at ingest:** drop null-id rows from player_stats and
   rosters, and dedupe injuries/rosters on key, keeping the latest record.
6. **Team codes:** nflverse uses `LA` for the Rams; `load_teams()` also contains `LAR` with the
   same name, so name→abbr maps must prefer `LA`.
7. **Data is revised post-game.** POSTGAME sources can change after first publication; ingest
   records `observed_at` per pull so labels are reproducible.

## Phase 1 ingest findings (2018–2026, 2026-09-27)

Frozen raw schemas are in `src/lineseek/contracts/raw_schemas.json`; regenerate them with
`scripts/freeze_raw_schemas.py`. Snap counts were added as a source (participation).

- **Dtype drift between seasons** is cast explicitly (`contracts/raw.py:KNOWN_CASTS`); any other
  difference fails the pull:
  - injuries `season`/`week`: Float64 in 2018–2020, Int32 from 2021
  - rosters_weekly `height`: Float64 before 2025, Int32 from 2025
  - pbp `goal_to_go`: Int32 in 2021–2023, Float64 in 2020 and 2024+
- **Format variants:**
  - injuries: `date_modified` present through 2024, gone from 2025 (`season_type` added)
  - depth charts: legacy weekly table through 2024, timestamped `dt` snapshots from 2025
- **Cancelled game:** 2022 wk 17 BUF@CIN (suspended, then cancelled) is missing from schedules
  but has injury rows. It is listed explicitly in `ingest/sources.py:CANCELLED_GAMES`.
- **pbp downloads fail transiently** (connection errors on 2018/2019), so the loader retries.

### Backfill (one snapshot per source × season, 2018–2026)

| Source | Rows | Parquet MB |
|---|---:|---:|
| pbp | 397,312 | 106.6 |
| depth_charts | 1,385,231 | 4.6 |
| rosters_weekly | 387,978 | 4.8 |
| snap_counts | 208,441 | 2.0 |
| player_stats | 150,425 | 5.4 |
| injuries | 46,071 | 1.0 |
| schedules | 2,499 | 0.3 |
| **Total (63 snapshots)** | **2,577,957** | **124.7** |

### Injury rule validation (4pm ET the day before kickoff)

`lineseek-ingest injury-rule-report` compares the rule with the real `date_modified` on all
39,269 rows from 2018–2024.

| Season | Rows | After rule | Share after rule | Share after kickoff |
|---|---:|---:|---:|---:|
| 2018 | 5,133 | 8 | 0.16% | 0.04% |
| 2019 | 5,392 | 6 | 0.11% | 0.09% |
| 2020 | 5,661 | 14 | 0.25% | 0.23% |
| 2021 | 5,587 | 90 | 1.61% | 0.05% |
| 2022 | 5,682 | 126 | 2.22% | 0.00% |
| 2023 | 5,599 | 116 | 2.07% | 0.00% |
| 2024 | 6,215 | 75 | 1.21% | 0.02% |
| **All** | **39,269** | **435** | **1.11%** | **0.06%** |

**The result is above the ~1% threshold. The rule is unchanged pending review.**

- **Thursday games cause most of it:** 311 of 435 late rows (13.0% of Thursday rows). Sunday is
  0.33% and Monday 0.56%. Wednesday reports for Thursday games often post just after 16:00 ET.
- **Most late rows are only slightly late:** median 0.55 h after the rule, p90 9.7 h, p99 63 h.
- **`date_modified` is a last-modified time,** so some late rows are later edits rather than late
  first publication. That makes 1.11% an upper bound on real violations.

