# ADR 0003: Phase 3 leakage-safe decision-time feature store

Status: accepted, 2026-09-30

## Context

Models (Phase 4) need pregame inputs that were knowable at the forecast lock. Built by
`lineseek.features.player_game`; contract `features` in `contracts/artifacts.py` (0.6.0).
No models are trained here.

## Decisions

1. **Grain:** one row per `(game_id, player_id, market, decision_time)`; `player_id` is GSIS.
   Rows are built for the keys of the season's latest `prop_labels` artifact
   (`game_id, player_id, team, market`; label values are never read). Features for unplayed
   games need a pregame player universe and are deferred.
2. **Decision time:** the existing forecast lock, `decision_time = forecast_time_utc(kickoff)`
   = latest Wednesday 12:00 America/New_York strictly before kickoff, stored in UTC.
3. **Availability of completed games (new rule, INFERRED):** player_stats and snap counts of a
   completed game may feed a later row once `available_at = kickoff_utc + 36 h`
   (`postgame_stats_available_at`). A history row contributes only if
   `available_at <= decision_time`; everything else is excluded (never back-filled). Because
   `decision_time < kickoff`, a game can never feed its own row. This refines ADR 0001 (6):
   POSTGAME data stays out of its own game and becomes an input to later games only after the
   lag. Monday night + 36 h is Wednesday 08:15 ET, so a normal weekly cadence always sees the
   previous game; COVID-era Tuesday/Wednesday games (10 team-games in 2020–21) are excluded
   from the next game's features.
4. **Calendar data:** season, week, home/away, postseason flag and scheduled kickoffs (for
   rest days) come from the schedule, treated as pre-published (EXACT). The Phase 1 schedule
   row's `available_at` is the kickoff itself, so schedule rows are not counted in
   `max_available_at`. Late reschedules are a known limitation.
5. **History (per season, `season_history`):** completed games only. An appearance is Phase 2
   participation: (offense_snaps + st_snaps) > 0, mapped PFR→GSIS with the Phase 2 map. Stats
   come from player_stats (0 for a participant without a stats row, as in labels). Stats with
   market/target activity but no snaps fail (conflict). Team per appearance uses the Phase 2
   precedence (player_stats, game-day roster, snap counts) via `assign_teams`, after
   `normalize_stats_teams` (LV→OAK 2018–19) and `correct_snap_teams` (2018 Super Bowl swap).
   Team totals are sums of player_stats rows per (game, team); "allowed" totals of a team are
   the other team's totals in that game. Games missing player_stats for a team are excluded
   from all history; games missing snap counts keep stats-row appearances with a null snap %.
6. **Windows.** Eligible player history E = appearances with `available_at <= decision_time`
   in the target season or the one before, most recent first (all positions and teams, so
   trades carry history). Team and opponent histories use the same rule keyed by franchise
   (OAK and LV are one franchise, from the same relocation table).
   - rolling windows (`roll3/5/8`, `roll5`) use the k most recent eligible games and may cross
     one season boundary; `_n` is the number of games used, `min(k, |E|)`
   - season-to-date (`season_*`, `opp_*_season_*`) uses target-season games only (resets)
   - games more than one season back never contribute; 2018 has no 2017 history
7. **Feature definitions** (v = the row market's stat: `pass_attempts`→attempts,
   `rush_attempts`→carries, `rush_yds`→rushing_yards, `receptions`→receptions,
   `reception_yds`→receiving_yards):

   | column | definition |
   |---|---|
   | `prior_games_n` | \|E\| |
   | `market_prev` | v in the most recent game of E |
   | `market_roll{3,5,8}_mean`, `_n` | mean of v over the k most recent games of E; games used |
   | `market_season_mean/median/std`, `_games` | over E ∩ target season; std is sample (ddof 1), null for < 2 games |
   | `pass_att_prev`, `carries_prev`, `targets_prev` | attempts / carries / targets in the most recent game |
   | `*_roll5_mean`, `receptions_roll5_mean` | mean over the 5 most recent games |
   | `pass_att_share_roll5`, `carries_share_roll5`, `target_share_roll5` | Σ player stat / Σ that player's team total, same 5 games; null if Σ team = 0 |
   | `snap_pct_prev`, `snap_pct_roll{3,5}_mean`, `_n` | offense_pct (0–1) in the most recent game; mean over non-null values of the k most recent games; non-null count |
   | `team_games_n` | team games in the window (≤ 5) |
   | `team_pass_att_roll5_mean`, `team_carries_roll5_mean`, `team_plays_roll5_mean` | per-game team attempts, carries, attempts + carries over the team's 5 most recent games |
   | `team_pass_rate_roll5`, `team_rush_rate_roll5` | Σ attempts / Σ (attempts + carries), Σ carries / Σ (attempts + carries) |
   | `opp_market_allowed_roll5_mean`, `opp_allowed_roll5_n` | per-game team total of v by teams facing the opponent, opponent's 5 most recent games (team totals, not position-specific) |
   | `opp_market_allowed_season_mean`, `opp_allowed_season_games` | same over the opponent's target-season games |
   | `is_home`, `is_postseason` | team is the home team; game_type ≠ REG |
   | `team_rest_days`, `opp_rest_days` | days between the team's previous scheduled kickoff in the season and this kickoff; null for its first game |
   | `days_since_prev_game` | days between the player's most recent game in E and this kickoff |

8. **Cold start and missing data:** rows are never dropped for short history. Counts are 0 and
   values null when no eligible game exists; partial windows use the games available. No
   value is filled with 0 unless 0 is the observed stat. Snap % is null when snap data is
   missing for that game (windows average the non-null values and report their count).
9. **Provenance:** row columns `decision_time`, `max_available_at` (latest `available_at` of
   any contributing player, team or opponent row; null if none), `worst_temporal_quality`
   (`inferred`, or `exact` when only the schedule contributed) and `feature_set_version`.
   The manifest lists every input snapshot and the labels artifact with checksums, the
   contract version and `parameters` (feature-set version, decision-time and availability
   rules, windows).
10. **Validation:** the contract checks the grain, identifiers, `decision_time < kickoff`,
    `max_available_at <= decision_time`, usable temporal quality and rejects label-only columns
    (`actual`, `played`, `label_status`). `validate_features` also checks the exact column set,
    dtypes, count/share/snap ranges (yards may be negative, counts may not), window counts and
    null consistency.
11. **Feature-set version** `FEATURE_SET_VERSION = "1.0.0"` (independent of
    `CONTRACT_VERSION`). Bump major for removed/renamed features or changed window, cutoff or
    missing-value semantics; minor for added features; patch for fixes that don't change any
    value. Output is deterministic: same inputs give byte-identical Parquet.
12. **Storage and CLI:** append-only Parquet at
    `$LINESEEK_FEATURES_ROOT/player_game_features/season=YYYY/{built_at}_{sha12}.parquet`
    (default `data/features`, git-ignored). `lineseek-features build --season 2025` (or
    `2018-2025`) reads the season's and previous season's raw snapshots plus the season's
    labels; `lineseek-features validate <file>` checks contract, feature set and checksum.

## Deferred

Route participation, air-yards/aDOT shares, red-zone usage, injuries/depth charts as features,
weather, betting lines, and live-game universes. None are proxied.

## Known limitations

- Values are from the latest snapshot; stat corrections published after a decision time are
  used for that row (no point-in-time copies of nflverse stats exist).
- The 36 h lag is a conservative rule, not an observed publication time.
- The players table is a present-day id reference; it maps ids only, never feature values.
- 2018 rows have no prior-season history (2017 is not ingested).
