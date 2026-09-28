# ADR 0001: Phase 0 foundation

Status: accepted, 2026-09-27

## Context

lineseek models NFL player-prop outcomes and compares them to sportsbook lines. The main
failure mode for this kind of system is look-ahead leakage: training or backtesting on
information that was not public at bet time. Phase 0 fixes the contracts before any
ingestion is built.

## Decisions

1. **Layout:** `src/lineseek/{ingest,features,models,sim,grade,alerts,app,contracts}`, built
   with hatchling. Scripts that hit the network live in `scripts/`, outside the package.
2. **Targets (v1, frozen, per PRD):** exactly five full-game stats: passing attempts, rushing
   attempts, rushing yards, receptions, receiving yards (`contracts/targets.py`). Other props
   (passing yards, TDs, completions, combos) are PRD non-goals. Labels come from nflreadpy
   weekly `player_stats`.
3. **Grading has two tracks** (PRD survivorship rule):
   - *Forecast:* every locked projection is scored. A player who didn't play scores 0.
   - *Market:* sportsbook rules. Non-participants are void. Participants without a stats row
     score 0.
   Participation is an explicit input (`prop_labels.played`), never inferred from a missing
   stats row, because zero-touch participants may have no row. `played` =
   (offense_snaps + st_snaps) > 0 from nflreadpy snap counts, with POSTGAME quality
   (`lineseek.grade.participation`). PFR ids are mapped to GSIS ids via the nflverse players
   table, then an exact roster name/team/season match, then the reviewed
   `grade/pfr_gsis_overrides.csv`. Any other unmatched id raises.
4. **Identifiers:** nflverse `game_id` and GSIS `player_id` are canonical. Provider IDs and
   names are stored raw and mapped onto them.
5. **Time:** all timestamps are tz-aware UTC. Every record carries `available_at` and
   `observed_at`; features carry `decision_time`, and `max_available_at <= decision_time` is
   enforced by `validate_frame` and `assert_as_of`.
6. **Temporal-quality provenance:** each source/field group gets a `TemporalQuality`
   (EXACT/OBSERVED/INFERRED/POSTGAME/UNKNOWN). POSTGAME is never used as a feature. UNKNOWN is
   excluded from features unless a later ADR defines a rule for it. INFERRED requires a
   written, conservative rule.
7. **Storage:** raw pulls are Parquet in `data/raw/{source}/season=YYYY/` with provenance
   columns. Derived artifacts (games, prop_labels, prop_lines, features, predictions, grades)
   are DuckDB tables whose schemas are in `contracts/artifacts.py`. Polars is the dataframe
   engine.
8. **Phase 1 sources:** nflreadpy schedules, PBP, player stats, weekly rosters, depth charts,
   injuries (audit: `docs/audit/phase1_sources.md`).
9. **Odds provider:** The Odds API (feasibility: `docs/audit/odds_api_feasibility.md`).
   Full ingestion is deferred. Historical backtests are blocked until a paid plan is chosen.
   The backfill takes two fixed snapshots per game: `forecast_time` (Wednesday 12:00 ET) and
   `closing` (last snapshot before kickoff). Opening lines come only from our live polling.
10. **External reference architectures:** BetBetter was considered and dropped; no
   external architecture is adopted.

## Consequences

- Any change to `lineseek.contracts` bumps `CONTRACT_VERSION`.
- 2024-and-earlier depth charts and all weekly roster statuses cannot be used as features
  until a timestamp rule is justified.
- Name-based player matching for odds is a required Phase 1+ component.
- Snap counts are added to the Phase 1 sources to supply participation.
- `assert_as_of` rejects any row tagged POSTGAME or UNKNOWN, whatever its timestamp
  (contracts 0.2.0).
