# Phase 3 feature store: adversarial verification audit

Date: 2026-10-02. Audited commit `f1ced0c` (contract 0.6.0, feature set 1.0.0). The working tree
was clean before the audit. No implementation code was changed.

**Result: CONDITIONAL GO for Phase 4A.** The audit found no Critical or High defects. One Medium
evaluation-universe issue and one unquantifiable revision risk must stay tracked (F1, F2).

## Method

- **Inputs.** Real data for 2018–2025: the raw snapshots used by the Phase 3 build
  (`players` included), checksum-verified. Labels and features were rebuilt from scratch into
  scratch roots, twice.
- **Independent reference.** A pandas/plain-Python reimplementation that imports nothing from
  `lineseek`. It re-derives kickoff, the Wednesday lock, availability, the PFR→GSIS map,
  appearances, team totals, windows and rest days from raw rows. It compared **every row**
  (357,170) on all 42 feature columns plus `decision_time`, `max_available_at` and `opponent`.
  Its sensitivity was checked by perturbing it: a wrong lag, no OAK→LV, a two-season lookback
  and no lag each produced mismatches.
- **Mutation tests.** 40 checks on real data using in-memory copies, covering:
  - the target game, Week 10 and a future season;
  - the availability boundary at −1 s, exactly at the lock and +1 s, on player, team and
    opponent history;
  - label scrambling;
  - an exact cross-player dependency chain.
- **Cross-checks against independent sources.** Team attempts and carries rebuilt from
  play-by-play, and snap-team labels compared with stats/roster teams for every game.

## Results

| Area | Result |
|---|---|
| Independent reference, all rows × 45 columns | **0 mismatches** |
| Same-game / future-game / future-season mutations | own rows unchanged; earlier rows unchanged; later rows change |
| Availability boundary | `lock − 1 s` included, `== lock` included (ADR: `<=`), `lock + 1 s` excluded, for player, team and opponent history |
| Label isolation | scrambled `actual`/`played`/`label_status` give identical output; label columns are rejected by validation |
| Unrelated player | changes reach other rows only through team totals (shares of that game's teammates, team rates) and opponent-allowed totals |
| Two-seasons-back | never contributes, even when passed to the builder |
| Determinism | byte-identical across 4 builds over 3 days and with 1, 2, 3, 7 and 8 threads; manifests differ only in `pulled_at`/`path` |
| Team totals vs play-by-play | attempts and carries equal in all 4,504 team-games (QB kneels count as carries in both) |
| Join with labels | 1:1 on `(game_id, player_id, market)`; 0 unmatched on either side; team/opponent/season/week agree |
| Counts | 357,170 = sum of the 8 season counts = label rows, for every season × market |
| Domain | shares, rates and snap % within [0, 1]; counts ≥ 0; `_n` = `min(k, prior_games_n)`; rest > 0; lock is always Wed 12:00 ET and < kickoff; `max_available_at <= decision_time`; no duplicate keys |
| COVID / reschedules | exactly 10 team-games (2020–21) whose previous game was unavailable at the lock; all excluded |
| OAK→LV | team and opponent history carry over (2020 Wk1 LV uses OAK 2019 Wk13–17); the only alias in the period |
| 2018 Super Bowl | the only game whose snap teams disagree with stats/roster teams (90/90 rows swapped); the correction touches that game only |
| Trades | history follows the GSIS id; share denominators use the team of each historical game; team context is the new team |
| Manual sample | 27 stratified rows (seed `20261002`) checked against raw history; no anomalies |

## Findings

**F1 — Medium: the evaluation universe uses post-lock and outcome information (potential
selection bias; not feature-value leakage).** Rows exist only for label keys, so membership is
decided by information from after the Wednesday lock.

| Membership path | Player-games | Share |
|---|---|---|
| Game-day ACT/INA roster entry at a skill position (published about 90 min before kickoff) | 71,052 | 99.47% |
| Non-skill player admitted **only because he recorded market activity** in that game (SPEC 116, DB 80, OL 59, LB 19, DL 8) | 282 (1,410 rows) | 0.39% |
| Stats or snaps without an ACT/INA skill roster entry | 100 | |
| **Total** | **71,434** | |

In addition, 2.8–6.4% of each previous game's skill participants are missing from their team's
next universe (IR, release, COVID list). A Wednesday-lock live system would not know about many
of these moves. Feature *values* are unaffected (row independence was verified).

Classification: **potential selection bias**. Acceptable as a training universe, but not
equivalent to live inference. Phase 4 should exclude or flag the 1,410 outcome-admitted rows,
state its handling of `did_not_play` rows, and report metrics on a universe that can be
reconstructed pregame. No rebuild is needed.

**F2 — Known limitation: revision leakage risk is unknown because historical source versions
are unavailable.** Every stored 2018–2025 stats and snap snapshot was pulled on 2026-09-28. Two
versions roughly 20 h apart are identical, but that shows only present-day stability. No
point-in-time versions exist, so corrected values that may have leaked backward cannot be
counted. The feature store must not be described as strictly point-in-time. Weekly snapshots
from 2026 onward will let this be measured.

**F3 — Low (latent; 0 rows today): snap coverage is keyed by game, not by team.** In
`season_history`, `covered` uses `game_id`. If only one team's snap rows are missing, the
other team also loses its valid snap % and switches to "stats row = appearance". That drops
zero-touch blockers and adds defense-only players.

- Reproduce: drop PIT's snap rows for `2023_07_PIT_LA`. LA snap features become null and the
  game's appearances go from 85 to 65.
- No 2018–2025 game lacks snaps, so no artifact is affected.
- This matches ADR 0003's game-level wording, so it was not changed. It should be revisited
  before live in-season builds, where one team's snaps may lag.

**F4 — Low: the manifest does not record polars/pyarrow versions,** although byte determinism
depends on them. They are pinned only indirectly, through `git_revision` → `pyproject.toml`.

**F5 — Low: ADR 0003 versioning has a gap.** "Patch" covers fixes that change no value, and
"major" covers semantic changes. There is no rule for a bug fix that changes values but keeps
the intended semantics.

**F6 — Low (documentation).** ADR 0002's "24 rows" for the Super Bowl swap counts label-universe
mismatches. The raw swap covers all 90 snap rows of `2018_21_NE_LA`.

**Observation (not a defect).** Means and std are float-order sensitive at about 1e-14. Output is
deterministic for fixed inputs, but exact-equality comparisons after unrelated input changes
can show last-bit differences.

**Reconciliation.** The earlier "333,550 comparable rows" figure is the set of rows whose entire
eligible history consists of label-universe `played` rows. The other 2,005 rows with history
include appearances outside the label universe. This audit's reference covers all 357,170 rows.

## Versions and tests

- No implementation change, so neither `CONTRACT_VERSION` nor `FEATURE_SET_VERSION` changes.
- Four regression tests were added to `tests/test_features.py`. Each was shown to be the only
  test that fails under an injected regression:
  - two-seasons-back exclusion, including through the window's season filter;
  - team/opponent availability boundary (a strict `<` regression);
  - the exact cross-player dependency chain (opponent-allowed computed from own production);
  - row independence from the key universe (team totals computed from labelled players only).
