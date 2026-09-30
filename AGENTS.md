# lineseek

- Package lives in `src/lineseek/`; install with `.venv/bin/pip install -e '.[dev]'`.
- Verify: `.venv/bin/pytest -q` and `.venv/bin/ruff check .` (DTZ rules are active: use tz-aware datetimes).
- Contracts (targets, IDs, time semantics, artifact + raw schemas) are frozen in `src/lineseek/contracts/`;
  bump `CONTRACT_VERSION` on any change. Decisions: `docs/adr/`.
- Ingest (network, writes to `$LINESEEK_DATA_ROOT`, default `data/raw`, never committed):
  - `.venv/bin/lineseek-ingest backfill [--seasons 2018-2026] [--sources a,b] [--force]` (skips existing source/season)
  - `.venv/bin/lineseek-ingest weekly [--season S --week W]`; scheduled via `scripts/ingest_weekly.sh`
    (example launchd plist in `scripts/launchd/`). Schedules are full-season; other sources are
    filtered to the week (depth charts: latest snapshot; stats/snaps: completed games only).
  - `.venv/bin/lineseek-ingest injury-rule-report`
- Labels (postgame ground truth, never features; ADR 0002), from the latest raw snapshots to
  `$LINESEEK_LABELS_ROOT` (default `data/labels`, never committed):
  - `.venv/bin/lineseek-labels build --season 2025` (or `2018-2025`; `--skip-incomplete`)
  - `.venv/bin/lineseek-labels validate <file.parquet>`
  - Needs schedules, player_stats, snap_counts, rosters_weekly and players snapshots for the season.
- Features (decision-time inputs; ADR 0003) to `$LINESEEK_FEATURES_ROOT` (default `data/features`,
  never committed); needs the season's labels plus raw snapshots for the season and the one before:
  - `.venv/bin/lineseek-features build --season 2025` (or `2018-2025`)
  - `.venv/bin/lineseek-features validate <file.parquet>`
  - Invariant: a row may only use history with `available_at <= decision_time`; bump
    `FEATURE_SET_VERSION` when feature definitions change.
- Schema drift fails pulls: review, then regenerate with `scripts/freeze_raw_schemas.py` or add a cast in `contracts/raw.py`.
- Network scripts: `scripts/audit_phase1.py` (nflreadpy), `scripts/odds_api_probe.py` (needs `ODDS_API_KEY`, spends credits).
- Tests use fixtures in `tests/fixtures/`, never live downloads.
