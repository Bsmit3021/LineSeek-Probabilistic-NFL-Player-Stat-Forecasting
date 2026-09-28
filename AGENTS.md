# lineseek

- Package lives in `src/lineseek/`; install with `.venv/bin/pip install -e '.[dev]'`.
- Verify: `.venv/bin/pytest -q` and `.venv/bin/ruff check .` (DTZ rules are active: use tz-aware datetimes).
- Contracts (targets, IDs, time semantics, artifact + raw schemas) are frozen in `src/lineseek/contracts/`;
  bump `CONTRACT_VERSION` on any change. Decisions: `docs/adr/`.
- Ingest (network, writes to `$LINESEEK_DATA_ROOT`, default `data/raw`, never committed):
  - `.venv/bin/lineseek-ingest backfill [--seasons 2018-2026] [--sources a,b] [--force]` (skips existing source/season)
  - `.venv/bin/lineseek-ingest weekly [--season S --week W]`; scheduled via `scripts/ingest_weekly.sh`
    (example launchd plist in `scripts/launchd/`)
  - `.venv/bin/lineseek-ingest injury-rule-report`
- Schema drift fails pulls: review, then regenerate with `scripts/freeze_raw_schemas.py` or add a cast in `contracts/raw.py`.
- Network scripts: `scripts/audit_phase1.py` (nflreadpy), `scripts/odds_api_probe.py` (needs `ODDS_API_KEY`, spends credits).
- Tests use fixtures in `tests/fixtures/`, never live downloads.
