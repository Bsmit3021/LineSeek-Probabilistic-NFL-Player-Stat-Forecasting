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
