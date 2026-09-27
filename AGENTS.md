# lineseek

- Package lives in `src/lineseek/`; install with `.venv/bin/pip install -e '.[dev]'`.
- Verify: `.venv/bin/pytest -q` and `.venv/bin/ruff check .` (DTZ rules are active: use tz-aware datetimes).
- Contracts (targets, IDs, time semantics, artifact schemas) are frozen in `src/lineseek/contracts/`;
  bump `CONTRACT_VERSION` on any change. Decisions: `docs/adr/`.
- Network scripts: `scripts/audit_phase1.py` (nflreadpy), `scripts/odds_api_probe.py` (needs `ODDS_API_KEY`, spends credits).
