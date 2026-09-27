# The Odds API feasibility (checked 2026-09-27)

Probe: `python scripts/odds_api_probe.py` (needs `ODDS_API_KEY`; spent 3 credits).

## Verified live

- NFL (`americanfootball_nfl`) active; 29 upcoming events; `/sports` and `/events` cost 0.
- `/events/{id}/odds` with 3 player-prop markets, region `us`: 6 books (draftkings, fanduel,
  betrivers, betonlineag, betmgm, bovada), cost 3 (1/market/region).
- All 5 v1 targets have Odds API market keys: `player_pass_attempts`, `player_rush_attempts`,
  `player_rush_yds`, `player_receptions`, `player_reception_yds`.
- Market-level `last_update` UTC timestamps are provided, so lines get EXACT temporal quality.
- Historical endpoints return `401 HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN` on the current key.

## From docs

- Historical player props exist from **2023-05-03** (so NFL 2023, 2024, 2025 seasons), 5-minute
  snapshots, `date` returns the closest snapshot at or before the requested time.
- Historical cost: 10 × markets × regions per event snapshot. Plans: free 500/mo (no history),
  $30 20K, $59 100K, $119 5M.

## Budget math (5 markets, 1 region)

- Live: one full-slate snapshot ≈ 16 games × 5 = 80 credits. Free tier ≈ 6 slates/month;
  20K ≈ 250 slates/month.
- Backfill for 2023–2025: 855 games × 5 markets × 10 ≈ 42.75K credits per snapshot time.
  Opening + closing ≈ 85.5K, which fits one month of the $59 100K plan and supports both
  CLV-style and closing-line backtests.

## Integration risks

- **Players are identified by name only** (`outcome.description`). Need name → gsis_id matching,
  restricted to the two teams' rosters for that event (for example, "Josh Allen" is ambiguous
  league-wide).
- **Teams use full names** ("Buffalo Bills"). Map via `load_teams()` (prefer `LA` for the Rams).
- **Events → game_id** by (home, away, kickoff date in ET). The provider's `commence_time`
  matches `kickoff_utc`.
- Suspended markets vanish about 15 minutes after their last update, so closing lines must be
  captured before kickoff.
- "Opening" is the first snapshot in which a market appears, so finding it means scanning
  backwards with `previous_timestamp`. Discovery calls cost extra on top of the 42.75K.

**Verdict:** feasible. Live capture works on the current free key for spot checks. Backtesting
against historical lines needs a paid plan: one month of the 100K tier covers opening and
closing snapshots.
