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
- Backfill for 2023–2025 uses two fixed snapshots per game (see below):
  855 games × 5 markets × 2 snapshots × 10 = **85,500 credits**. Add about 130 credits for
  `/historical/.../events` lookups (1 credit each; 3 seasons × 22 weeks × 2 snapshot times). That fits
  one month of the $59 100K plan and supports both CLV-style and closing-line backtests.

## Backfill snapshot definition

Each game gets exactly two historical snapshots, requested with the `date` parameter (the
provider returns the latest snapshot at or before `date`):

| Snapshot | Requested `date` | Definition (`lineseek.contracts.temporal`) |
|---|---|---|
| `forecast_time` | latest Wednesday 12:00 ET strictly before kickoff, stored in UTC | `forecast_time_utc(kickoff)` |
| `closing` | kickoff − 1 s | `closing_snapshot_time_utc(kickoff)` |

There is no first-appearance search. True opening lines come from our own live polling going
forward, not from the backfill.

## Integration risks

- **Players are identified by name only** (`outcome.description`). Need name → gsis_id matching,
  restricted to the two teams' rosters for that event (for example, "Josh Allen" is ambiguous
  league-wide).
- **Teams use full names** ("Buffalo Bills"). Map via `load_teams()` (prefer `LA` for the Rams).
- **Events → game_id** by (home, away, kickoff date in ET). The provider's `commence_time`
  matches `kickoff_utc`.
- Suspended markets vanish about 15 minutes after their last update, so closing lines must be
  captured before kickoff.
- Props may not be posted yet at the Wednesday forecast time, especially for Sunday/Monday games.
  Those games get no `forecast_time` line; empty responses cost nothing. Coverage has to be
  measured once the backfill runs.

**Verdict:** feasible. Live capture works on the current free key for spot checks. Backtesting
against historical lines needs a paid plan: one month of the 100K tier covers forecast-time
and closing snapshots.
