# ADR 0004: Phase 4A deterministic baselines and walk-forward evaluation

Status: accepted, 2026-10-02

## Context

Phase 4B will train ML point-forecast models. Their value can only be judged against an
uncontaminated, fixed benchmark under a fixed protocol. This phase builds that protocol and
benchmark. It has no ML, no sportsbook data and no probabilities. The Phase 3 audit
(`docs/audit/phase3_feature_store_audit.md`) ended CONDITIONAL GO. Its findings F1
(evaluation universe), F2 (stat revisions) and F3 (snap coverage) must stay visible.

Code:

- `lineseek.models.baselines` (predictions);
- `lineseek.evaluation.{walkforward,universe,metrics,phase4a,cli}`.

Contracts (0.7.0): `baseline_predictions` and `walk_forward_folds` in `contracts/artifacts.py`.

## Decisions

1. **Baselines** (`BASELINE_SET_VERSION = "1.0.0"`). Names are frozen, in reporting order. v is
   the row market's stat.

   | name | prediction | history_n |
   |---|---|---|
   | `prev_game` | `market_prev` | `min(prior_games_n, 1)` |
   | `roll3_mean` | `market_roll3_mean` | `market_roll3_n` |
   | `roll5_mean` | `market_roll5_mean` | `market_roll5_n` |
   | `roll8_mean` | `market_roll8_mean` | `market_roll8_n` |
   | `season_mean` | `market_season_mean` (resets each season) | `market_season_games` |
   | `ewma_span5` | EWMA of v over the Phase 3 eligible history | `prior_games_n` |

   B1–B5 copy audited Phase 3 columns. They read only key and feature columns, never labels.

   **EWMA:** alpha = 2/(5+1) = 1/3, fixed before any evaluation and never tuned. The history is
   the Phase 3 eligible set E, selected with the same `_window` helper:
   - player appearances only;
   - `available_at <= decision_time`;
   - the target season or the one before.

   E is ordered oldest first. `EWMA_1 = v_1` and `EWMA_t = alpha * v_t + (1 - alpha) * EWMA_(t-1)`
   (`ewm_mean(adjust=False)`). The prediction is the value after the most recent game in E.

   Each EWMA row is cross-checked against its feature row: the history length must equal
   `prior_games_n` and the most recent value must equal `market_prev`. If either differs, the
   build fails, because the histories were not the ones the features were built from. The raw
   snapshots are loaded from the exact paths and checksums in each feature manifest.

   A prediction is null exactly when `history_n == 0`. Nothing is imputed.
2. **Prediction is separate from scoring.** `evaluate` generates predictions from features and
   history, serialises them to Parquet and records the SHA-256. Only then does it read labels;
   it scores exactly the frozen bytes. The universe class is kept out of the prediction
   artifact so it can never become an input.
3. **Evaluation period.** The period is frozen and recorded in the manifest. The CLI has no
   option to change it.
   - Out-of-sample: 2021–2025.
   - Warm-up and training history: 2018–2020.
4. **Walk-forward folds** (`walkforward.py`) are reused unchanged by Phase 4B.
   - There is one fold per unique `decision_time` of the evaluation seasons; `fold_id` is in
     time order. This gives 22 per season: 18 regular-season locks and 4 playoff rounds.
   - Test rows: `decision_time ==` the fold's lock.
   - Training rows: rows from any season with
     `outcome_available_at = kickoff_utc + 36 h <= decision_time`. This is the Phase 3 rule
     `postgame_stats_available_at`.
   - Week numbers are never used and nothing is shuffled. The training set only grows.
   - `split()` re-asserts that every training outcome was available and every test kickoff
     follows the lock.
   - Games that kicked off before the lock but were not yet available are counted
     (`started_unavailable_rows`) and kept out of training. The only such case in 2021–2025
     is the 2021 Week 16 fold (three Tuesday games).
   - The deterministic baselines need no fitting. The folds still define the protocol, and
     every score is attributed to its fold.
5. **Universe policy (F1).** `universe.classify_universe` re-applies the Phase 2 universe rules
   to each labelled player-game. It is evaluation-only, uses postgame information and never
   feeds predictions. Paths:
   - `roster`: an ACT/INA game-day roster entry at a skill position;
   - `participation`: no such entry, but a skill-group stats row or a skill-position snap;
   - `outcome_only`: admitted only because the player recorded activity in a target market.

   2018–2025 counts are 71,052 / 100 / **282** player-games, matching the audit (1,410 rows).
   - **Primary cohort:** excludes `outcome_only`. It is *game-day-roster-conditioned*: it still
     depends on ACT/INA information published after the Wednesday lock, so it is **not** a
     live Wednesday-lock backtest.
   - **Diagnostic cohort** (`full_label_universe`): keeps every row and is reported only to
     show the effect of the selection.
6. **Participation tracks.**
   - `participant`: `played` rows only, with `actual` as labelled. A participant's 0 is a real 0.
     Point values equal the contract's market track (`resolve_actual(..., MARKET, played)`),
     so a separate market diagnostic would be identical and is not reported.
   - `forecast_lock`: every row, with `resolve_actual(actual, Track.FORECAST)`, so a
     non-participant scores 0. This is the existing Phase 0 rule. `resolve_actual` is applied
     once per distinct (actual, played) pair, and no other null is turned into 0.
7. **Metrics** (`metrics.py`). Metrics are computed per cohort × track × view × market ×
   baseline. The period is each season or 2021–2025; for 2021–2025 they are also split by
   history bucket (0, 1, 2, 3–4, 5–7, 8+ prior games).
   - `error = prediction - actual`, so a positive `mean_error` means overprediction.
   - Reported: `n_rows`, `n_eligible` (rows scored on the track), `n_predicted`,
     `coverage_rate = n_predicted / n_eligible`, MAE, RMSE, median AE, mean error, P90 AE
     (linear interpolation, R type 7) and max AE (diagnostic).
   - There is no MAPE: outcomes of 0 are legitimate. There is no cross-market score: the
     units differ.
   - Views:
     - `native`: each baseline wherever it predicts.
     - `common_support`: only rows where all six baselines predict, which in practice is
       `market_season_games >= 1`.
   - A null prediction is uncovered, not an error.
8. **Comparison and uncertainty.** Each baseline is compared with `prev_game` on MAE, RMSE and
   coverage, with the number of OOS seasons in which its MAE is better or worse. For the
   primary cohort on common support, a 95% percentile interval for the paired MAE difference
   comes from 2,000 resamples of whole **games** (seed 20261002). Rows within a game share a
   game script, so rows are never resampled independently.
9. **Benchmarks** (`benchmarks.parquet`). For each market and track (primary cohort), four
   baselines are frozen; ties are listed:
   - best MAE on common support;
   - best RMSE on common support;
   - best native MAE;
   - highest native coverage.
10. **Artifacts.** Each run is append-only, in
    `$LINESEEK_EVALUATION_ROOT/phase4a/{built_at}_{content12}/` (default `data/evaluation`,
    git-ignored). It contains:
    - `baseline_predictions`, `cohorts`, `folds`, `baseline_metrics`,
      `baseline_comparison`, `benchmarks` (`.parquet`);
    - `manifest.json`, which records:
      - the contract, feature-set and baseline-set versions;
      - the checksum of every feature, label and file;
      - the seasons, timing, fold, universe and DNP rules, and the metric definitions;
      - the research sources and runtime versions (python, polars, pyarrow, numpy,
        nflreadpy, git revision);
      - the warnings.

    The same inputs give byte-identical Parquet files and the same content id.
11. **CLI.**
    - `lineseek-baselines build` runs the frozen protocol and prints the summary and warnings.
    - `lineseek-baselines validate <run dir>` checks checksums, contracts, one row per
      baseline per key, fold coverage and metric counts.
12. **Versioning.** `BASELINE_SET_VERSION` is independent of `CONTRACT_VERSION` and
    `FEATURE_SET_VERSION`.
    - Major: a baseline removed or renamed, a formula changed, or the scoring universe, folds
      or period changed materially.
    - Minor: a baseline or non-breaking metric added.
    - Patch: a bug fix that does not change intended formulas.

    Adding the two artifact contracts bumped `CONTRACT_VERSION` 0.6.0 → 0.7.0.
    `FEATURE_SET_VERSION` is unchanged (1.0.0).

## Phase 4B promotion rule (defined, not yet applied)

A model counts as an improvement for a market only if all of the following hold:

1. It is evaluated on these exact folds, with the same primary cohort, the same DNP tracks and
   the same targets.
2. It is trained only on each fold's training rows.
3. Its coverage is not lower than the benchmark's, unless justified and reported in both views.
4. Its MAE is lower than the frozen best-MAE benchmark on common support (the model's rows ∩
   the benchmark's). The game-clustered interval for the difference must exclude 0.
5. It wins in a majority of the OOS seasons (at least 3 of 5), not only in aggregate.
6. P90 AE and RMSE do not degrade materially without an explicit trade-off analysis.

It must also be reported on the forecast-lock track and the diagnostic cohort.

## Research consulted

| Source | Technique | Adopted | Not adopted |
|---|---|---|---|
| Hyndman & Athanasopoulos, *FPP3* §5.10 | Evaluation on a rolling forecasting origin: train only on observations before each test point | One origin per decision batch; an expanding training set | Fixed-size sliding windows |
| scikit-learn `TimeSeriesSplit` docs | Expanding train set, a `gap` between train and test, no shuffling | Expanding, unshuffled folds; the gap is LineSeek's availability rule (kickoff + 36 h), not a fixed sample count | The class itself (row-count folds ignore real time) |
| *FPP3* §5.8 | MAPE is undefined at zero; MAE/RMSE are not comparable across units | No MAPE; per-market reporting only | Scaled errors (MASE), deferred |

The cluster bootstrap is standard practice (resample independent units, here games). No
external code was copied and no dependency was added.

## Known limitations (shown with every result)

- **F2:** Historical nflverse statistics are reconstructed from later snapshots. True
  point-in-time historical revisions are unavailable, so statistical revision leakage cannot
  be completely ruled out.
- **F1:** the primary universe is conditioned on the game-day roster, not a Wednesday-lock
  universe.
- **F3:** snap coverage is tracked per game, not per team. It affected no 2018–2025 rows and
  must be hardened before live use.
- `pass_attempts` is scored over every skill player, so most rows are non-QBs with an actual of
  0 and the pooled MAE is dominated by them. Phase 4B should also report a QB-role diagnostic.
