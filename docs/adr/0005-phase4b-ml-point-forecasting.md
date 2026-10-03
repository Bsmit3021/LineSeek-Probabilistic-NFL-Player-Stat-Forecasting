# ADR 0005: Phase 4B walk-forward ML point forecasting

Status: accepted, 2026-10-03

## Context

Phase 4A (ADR 0004) froze the evaluation protocol and benchmarks:

- 110 walk-forward folds;
- the primary game-day-roster-conditioned cohort;
- participant and forecast-lock tracks;
- common support;
- game-clustered intervals;
- per-market benchmarks: `pass_attempts` and `rush_yds` use `season_mean`,
  `rush_attempts` and `receptions` use `ewma_span5`, and `reception_yds` uses `roll8_mean`.

Phase 4B asks whether a reasonably configured learned model beats those benchmarks on exactly
that playing field. It is point forecasting only: no probabilities, no sportsbook data and no
tuning.

Code:

- `lineseek.models.learned` (estimators);
- `lineseek.evaluation.phase4b` (orchestration);
- `lineseek.evaluation.models_cli` (`lineseek-models`).

Contract 0.8.0 adds `model_predictions`.

## Decisions

1. **Model set** (`MODEL_SET_VERSION = "1.0.0"`). Exactly two families, `ridge` and
   `lightgbm`, one independent model per market per fold: 2 × 5 × 110 = 1,100 fits.
   - Versioning:
     - Major: a family removed or renamed, training-target semantics changed, or
       preprocessing semantics changed.
     - Minor: a model or diagnostic added.
     - Patch: a correction that preserves semantics.
   - The ablation `lightgbm_history_only` is a diagnostic, not part of the model set.
2. **Features.** All 42 Phase 3 feature columns, in contract order (`learned.FEATURES`).
   Booleans are cast to 0/1 and nulls become NaN. The list is identical in every fold.
   - `learned.FORBIDDEN` refuses the following, so they never enter a matrix: every feature
     key column (`game_id`, `player_id`, `team`, `opponent`, `market`, timestamps,
     `max_available_at`, `worst_temporal_quality`, versions), `season`, `week`, label columns,
     cohort and QB flags, fold IDs and names.
   - There are no ID encodings and no target encoding.
3. **Ridge.** `FeatureUnion[SimpleImputer(median, keep_empty_features=True),
   MissingIndicator(features="all")] → StandardScaler → Ridge(alpha=1.0)`.
   - Every statistic is fitted inside the fold's pipeline on its training rows.
   - `MissingIndicator(features="all")` is used instead of `SimpleImputer(add_indicator=True)`,
     whose indicators cover only columns missing at fit time. That would make the width vary
     by fold.
   - `alpha=1.0` is fixed and was not tuned.
4. **LightGBM** (4.7.0, the version already pinned in the `mr2` extra):

   ```
   objective=l1, n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
   reg_lambda=1.0, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, max_bin=255,
   deterministic=True, force_row_wise=True, n_jobs=1, random_state=20261002, verbose=-1
   ```

   - The L1 objective matches the primary MAE metric (a conditional-median forecast).
   - Missing values use native handling.
   - The number of rounds is fixed: no early stopping and no validation set drawn from
     evaluation data.
   - Every parameter was fixed before evaluation. There was no search.
5. **Determinism.** Each fit is single-threaded, seeded and deterministic, with no bagging or
   feature subsampling. For speed, markets and models run in separate processes. Each task is
   a pure function of its own data, so the output does not depend on process count; a test
   compares 1 and 2 workers.
   - Aggregates are taken from frames in a fixed order with `group_by(maintain_order=True)`.
     Polars' bare `mean()` fast path proved non-deterministic in the last bit (11 distinct
     results in 20 runs). The shared `error_metrics` was hardened the same way, and the
     Phase 4A artifacts were verified byte-identical afterwards (content `c21d61557c9d`).
6. **Frozen Phase 4A inputs.** `load_phase4a` validates the Phase 4A run (checksums).
   - Its folds, cohorts, baseline predictions and benchmark choices are read, never rebuilt.
   - `walk_forward_folds` is recomputed and must equal the frozen folds exactly.
   - The feature and label artifacts must have the checksums the Phase 4A run used.
   - The benchmark's metrics are recomputed by the Phase 4B scorer and must equal the frozen
     Phase 4A metrics.
7. **Training population per fold.** Rows from any season qualify if
   `kickoff_utc + 36h <= decision_time` (the Phase 3/4A availability rule), the row is in the
   primary cohort (not `outcome_only`) and `played` is true. The target is the labelled actual.
   - DNP rows are never zero targets.
   - Estimators receive only `(X_train, y_train, X_test)`.
   - `run_market` re-asserts availability and that training and test rows are disjoint.
8. **Prediction population.** Every row with the fold's `decision_time`, including eventual
   DNPs, which the model cannot know about. Outcome-only rows are predicted for the diagnostic
   cohort only. Predictions are serialised and checksummed before labels are joined.
9. **Count clipping.** For `pass_attempts`, `rush_attempts` and `receptions`,
   `prediction = max(0, raw)`. Yards are never clipped and nothing is rounded. The contract
   checks the rule and the artifact keeps `prediction_raw`.
10. **Scoring.** The unchanged Phase 4A metric code (`compute_metrics`, now parameterised by
    predictor names) is used with the predictors `benchmark`, `ridge`, `lightgbm` and
    `lightgbm_history_only`.
    - Common support is Phase 4A's: rows where all six baselines predict. Every predictor is
      scored on identical rows there.
    - The game-clustered bootstrap (2,000 resamples, seed 20261002) is reused for the paired
      MAE delta against the benchmark.
11. **QB diagnostic.** `pass_attempts` rows whose game-day (ACT/INA) roster position is QB. It
    is roster-conditioned, used for evaluation only, and never a feature.
12. **Diagnostics.**
    - LightGBM gain share and Ridge standardised coefficients per fold, aggregated: mean,
      standard deviation and share of folds with a positive coefficient.
    - Raw and clipped counts and MAE.
    - The history-only LightGBM ablation.
    - Fit wall-clock times are stored apart from the deterministic content.

## Promotion rule

Fixed in code (`phase4b.py`) before any result was computed. It is evaluated on the headline
view: primary cohort, participant track, common support, 2021–2025, frozen folds, per market.

- **PROMOTE** requires all of the following:
  1. MAE lower than the frozen benchmark;
  2. the game-clustered 95% CI of the MAE delta entirely below 0;
  3. MAE better in at least 3 of 5 seasons, counting a season as tied if its delta is within
     ±0.5% of the benchmark's MAE;
  4. native coverage not more than 1 percentage point below the benchmark's;
  5. P90 AE not more than 5% worse;
  6. no temporal or protocol violation;
  7. the MAE delta also negative on the full label universe.
- **PROMISING:** MAE is lower, but not every PROMOTE condition holds.
- **NO IMPROVEMENT:** MAE is equal or higher.
- **REJECT:** an integrity failure, or P90 AE more than 10% worse.
- **Selection** for Phase 4C: the lowest-MAE PROMOTE model per market, otherwise the frozen
  Phase 4A benchmark.

## Research consulted

| Source | Influenced |
|---|---|
| LightGBM Parameters documentation | `objective="l1"` (alias of `regression_l1`); `deterministic=true` must be paired with `force_row_wise`/`force_col_wise`; `seed`; a fixed `num_threads` |
| scikit-learn `SimpleImputer` documentation | `add_indicator` marks only features missing at fit time (variable width), and all-missing columns are dropped unless `keep_empty_features=True`. This led to `MissingIndicator(features="all")` and `keep_empty_features=True` |
| scikit-learn `Pipeline`/`FeatureUnion`/`StandardScaler`/`Ridge` | The fold-local pipeline: every statistic is fitted on the training rows |

No recipe was copied, no new dependency was added (LightGBM was already pinned), and there was
no tuning.

## Results (real data, frozen Phase 4A run `c21d61557c9d`, Phase 4B content `064978d1cc1c`)

- **Scale:** 110 folds and 1,650 fits: 1,100 for the model set plus 550 for the ablation.
  There are 683,340 predictions, 0 temporal violations and 0 duplicate keys. Two runs from
  identical inputs gave byte-identical files.
- **Headline view:** primary cohort, participant track, common support, 2021–2025,
  n = 33,741 per market. Δ is the change in MAE versus the benchmark; the CI is a
  game-clustered 95% interval.

| Market | Benchmark (MAE) | Ridge MAE (Δ) | LightGBM MAE (Δ, 95% CI) | Ridge | LightGBM |
|---|---|---|---|---|---|
| pass_attempts | season_mean 0.771 | 0.830 (+0.060) | 0.710 (−0.060, [−0.076, −0.044]) | REJECT¹ | PROMOTE |
| rush_attempts | ewma_span5 1.071 | 1.131 (+0.061) | 1.002 (−0.069, [−0.076, −0.061]) | NO IMPROVEMENT | PROMOTE |
| rush_yds | season_mean 6.251 | 6.596 (+0.345) | 5.686 (−0.565, [−0.624, −0.507]) | NO IMPROVEMENT | PROMOTE |
| receptions | ewma_span5 1.055 | 1.046 (−0.009) | 0.988 (−0.067, [−0.072, −0.061]) | PROMOTE | PROMOTE |
| reception_yds | roll8_mean 13.404 | 13.371 (−0.033) | 12.243 (−1.162, [−1.241, −1.084]) | PROMISING | PROMOTE |

¹ The P90 ratio is 4.55 on a near-zero benchmark P90 (0.143). Most `pass_attempts` rows are
non-QBs with an actual of 0, and Ridge gives them small positive values. On the QB-role
cohort, Ridge *beats* the benchmark (MAE 7.93 vs 8.18). The label stands under the frozen rule.

- **LightGBM:**
  - It wins 5 of 5 seasons in every market.
  - Its P90 is not worse in any market (P90 ratio 0.18–0.98).
  - Its native coverage is 100% vs the benchmark's 91–98%.
  - It also wins on the forecast-lock track and the full label universe.
  - Its RMSE is lower than the benchmark's in every market, though for receptions and both
    yardage markets Ridge has the lowest RMSE.
- **QB-role `pass_attempts`** (common support, n = 3,150): MAE is benchmark 8.181, Ridge 7.932
  and LightGBM 7.558 (RMSE 10.26 vs 11.14). LightGBM is better in every season.
- **Bias:** the L1 objective produces a conditional median, so LightGBM sits well below the
  mean: −2.25 rushing yards, −4.66 receiving yards and −0.31 rush attempts, vs benchmark biases
  of −0.43, +0.10 and −0.02.
- **How much of the gain is the median effect** (analysis only, not a benchmark): the
  deterministic `market_season_median` improves MAE only modestly. Rushing yards go to 6.086
  (about 30% of LightGBM's gain), receiving yards to 13.150 (about 22%), and it is no better on
  attempts. Most of LightGBM's gain is therefore usage and history signal, not loss alignment.
  Phase 4C must treat LightGBM's point forecast as a median, not an expectation.
- **Ablation:** history-only LightGBM gets 75–80% of the full model's gain, for example rushing
  yards −0.453 vs −0.565. The Phase 3 context (shares, snaps, team, opponent) adds a smaller
  but clearly separable improvement.
- **Diagnostics:**
  - Top gain: `target_share_roll5` for receptions and receiving yards, `market_prev` for pass
    attempts, `carries_share_roll5` for rushing yards, `market_roll8_mean` for rush attempts.
    No identifier or calendar proxy appears.
  - Ridge coefficients are stable in sign. The only sign-split pairs are identical columns
    within a market (for example `market_roll5_mean` and `carries_roll5_mean` in rush attempts).
  - Ridge clips 26,746 count predictions, which helps its MAE by up to 0.048. LightGBM clips
    7,515, with an MAE effect under 0.001.
- **Largest errors** are legitimate: backup QB starts, injuries and early exits, role changes,
  breakouts and explosive games. LightGBM shrinks stars toward the median (for example Barkley
  81 vs the benchmark's 114 against an actual of 255).
- **Selected point model for Phase 4C:** `lightgbm` for all five markets (PROMOTE, lowest MAE).

## Limitations (carried with every result)

- **F1:** the primary cohort is conditioned on the game-day roster, not a Wednesday-lock
  universe.
- **F2:** Historical nflverse statistics are reconstructed from later snapshots. True
  point-in-time historical revisions are unavailable, so statistical revision leakage cannot
  be completely ruled out.
- **F3:** snap coverage is tracked per game, not per team. This is a pre-live hardening item.
- The QB diagnostic depends on game-day roster position.
