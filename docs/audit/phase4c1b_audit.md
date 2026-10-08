# Phase 4C.1b independent audit (analysis only)

Date: 2026-10-08. Scope: committed code at `f966ec4`, ADR 0008, and both frozen runs:

- Run 1: `data/evaluation/phase4c1b/20261008T173929762154Z_0a9365e1dbba`
- Run 2: `data/evaluation/phase4c1b/20261008T213040228527Z_0a9365e1dbba`

Ground rules:

- No retraining, no threshold changes, no code changes.
- **No correctness defect was found.**
- Phase 5 was not started.

Scripts and outputs are git-ignored and live in `data/evaluation/phase4c1b_audit/`:

| Script | Covers |
|---|---|
| `scripts/audit_integrity.py` (`integrity.json`) | Hashes, immutability, structure, leakage |
| `scripts/audit_scoring.py` | Independent scoring from frozen predictions and labels |
| `scripts/audit_compare.py` | Comparison with the run artifacts and re-application of the rule |

These scripts do not call `lineseek.evaluation.phase4c1b`. They reuse only the input loader and
the `qb_role` cohort definition.

## Results

| # | Check | Verdict |
|---|---|---|
| 1a | Tests and Ruff | **PASS** |
| 1b | Artifact hashes, both runs | **PASS** |
| 1c | Validation and determinism | **PASS** |
| 2 | Temporal leakage; split before threshold expansion | **PASS** |
| 3 | 550 fits, folds, unique out-of-fold rows | **PASS** |
| 4 | 0.33 pp vs 4.08 pp ECE definitions | **PASS** (definitions correct; the aggregate is diluted, see §4) |
| 5 | Brier improvements and correlated-observation CIs | **PASS** |
| 6a | Monotonicity | **PASS** |
| 6b | PMF mass conservation | **PASS** |
| 6c | Tail treatment | **PASS**, with a Phase 5 limitation |
| 6d | Exact-zero calibration | **PASS**, weaker than the zero model |
| 7 | Readiness follows the pre-registered rule | **PASS** |
| 8 | Phase 4C.1 artifacts unmodified | **PASS** |

No check is FAIL or UNVERIFIED. Evidence for each follows.

### 1. Tests, Ruff, hashes, validation

- **Tests and Ruff:**
  - `pytest -q`: 265 passed (265.8 s).
  - `ruff check .`: all checks passed.
  - The working tree was clean at `f966ec4`.
- **Artifact hashes:**
  - For both runs, all 12 file SHA-256s match `manifest.json`.
  - Recomputing `content_sha256` gives `0a9365e1dbbaa3592d8dc305325abeeac0c75333b866620c867bb058873080b6`
    for both runs, equal to their manifests.
  - All 12 files are byte-identical across the two runs.
- **Validation:** `lineseek-cdf-tc validate` reports OK for both runs: 5,922,280 rows, CDF set
  v2.0.0, CONDITIONAL GO.

### 2. Temporal leakage

**Split before expansion.** This is confirmed by code review:

- `phase4c1b.py:145`: `train = ok & (task.available <= dt)` selects player-game rows.
- `phase4c1b.py:149`: only `x[train]` and `y[train]` reach `fit_predict_tc`.
- `cdf_tc.py:43,46`: `stack(...)` and the targets are built from those rows only.
- `cdf_tc.py:47`: test rows are expanded separately.
- **Alignment:** `stack` repeats rows and tiles thresholds, which matches the row-major reshape
  of `z = y[:,None] <= t`. Expanded row `i*T+j` is therefore paired with `1{y_i <= t_j}`.

**Independent recount**, rebuilt from labels, the Phase 4A cohorts and `kickoff + 36 h`:

- Training rows match `fold_metadata` in all 550 fits: 21,772,500 player-game rows in total;
  0 mismatches; 0 nulls.
- The last available training time matches in all 550 fits.
- `train_rate_k0` matches with a maximum difference of 0.
- No fit trained after its decision time. The minimum gap between the lock and the last
  available training row is 2.0 h.
- Test-row exposure:
  - 0 test rows were available by their own fold's lock;
  - 0 test decision times are at or after kickoff;
  - no non-participant rows are in the eligible population.

### 3. Fits, folds, uniqueness

- **Fits:** 550 rows in `fold_metadata`, 550 unique (fold, market) pairs, and 110 fits per
  market.
- **Folds:**
  - the fold IDs equal the 110 Phase 4A folds;
  - fold decision times are strictly increasing;
  - every prediction's `fold_id` maps to its own `decision_time` (0 mismatches, 0 nulls).
- **Cells:** 5,922,280 cells, all unique on (game, player, market, threshold), covering 227,780
  unique player-game-market rows. Each row has exactly its market's grid count of cells.
- **Out-of-fold uniqueness:** `test_rows` summed over fits equals the row count, and it matches
  per (fold, market). Every row is predicted by exactly one fit.
- **Other:** seasons 2021–2025; only predictor `cdf_tc`; stacked rows equal rows × T in every
  fit, for 566,085,000 in total.

### 4. Pass-attempt ECE: 0.33 pp (all players) vs 4.08 pp (QBs)

Both use the same definitions:

- **ECE:** 20 equal-width bins, `Σ n_b/n |mean p_b − rate_b|`, per threshold.
- **Averaging:** a plain mean over that cohort's informative thresholds.
- **Informative rule:** ≥ 200 outcomes on each side and an observed rate in [5%, 95%], fixed per
  cohort over 2021–2025.

**Independent recomputation agrees exactly.** Comparing with `threshold_metrics.parquet` over
161 cohort-thresholds × 3 predictors × {ECE, Brier, log loss, bias} gives a maximum difference
of 0.0, and the informative flags match.

| Cohort | Rows | Informative k | Mean ECE (pp) | Worst (pp) |
|---|---|---|---|---|
| Primary (all players) | 36,997 | 11 (k = 0 … 28) | 0.3257 | 0.515 |
| QB role | 3,567 | 27 (k = 0 … 44) | 4.0800 | 5.965 |
| Non-QB | 33,430 | — | 0.13 on the primary k, 0.07 on the QB k | — |

**Why the two numbers differ.** The gap is dilution, not threshold choice:

- Non-QBs are 90.4% of pass-attempt rows. Their predicted F(0) is 0.992 against an observed zero
  rate of 0.996, so they are trivially well calibrated.
- On the *same* k sets:
  - all players score 0.35 pp on the QB-informative k;
  - QBs score 4.14 pp on the primary-informative k.

**What the gate depends on.** The 0.33 pp figure is valid but says nothing about QB lines. The
gate correctly applies the QB criteria, so pass attempts rests on 4.08 / 5.96 pp. QB ECE is
above 3 pp at 20 of 27 thresholds and above 5 pp at 6 (k = 28–32 and 35). The noise floor is
1.78 pp.

### 5. Comparative Brier and CIs

The independent per-row differences (mean over informative thresholds) match the run's
`comparisons.parquet`:

- Brier: maximum difference 1.7e-18.
- Log loss: maximum difference 2.8e-17.
- CI upper bounds: within 4e-5 (1,000 reps vs the run's Phase 4A reps).

**Correlation handling.**

- **Thresholds within a row:** averaged per row before resampling.
- **Rows:** clustered three ways, each with a separate percentile bootstrap:
  - by game: 1,424 clusters;
  - by player: 1,224 clusters (142 for QBs);
  - by fold/week: 110 clusters.

| Market | Δ Brier vs empirical | CI upper bound (game / player / fold) | Δ vs 4C.1 | CI upper bound (game / player / fold) |
|---|---|---|---|---|
| pass_attempts | −0.00244 | −0.00206 / −0.00176 / −0.00199 | −0.00035 | −0.00019 / −0.00016 / −0.00019 |
| rush_attempts | −0.00569 | −0.00513 / −0.00463 / −0.00514 | −0.00048 | −0.00031 / −0.00031 / −0.00031 |
| rush_yds | −0.00544 | −0.00491 / −0.00455 / −0.00488 | −0.00039 | −0.00023 / −0.00021 / −0.00023 |
| receptions | −0.01013 | −0.00949 / −0.00930 / −0.00932 | −0.00082 | −0.00064 / −0.00061 / −0.00065 |
| reception_yds | −0.00941 | −0.00875 / −0.00858 / −0.00858 | −0.00042 | −0.00025 / −0.00025 / −0.00023 |
| QB pass_attempts | −0.01600 | −0.01297 / −0.01148 / −0.01280 | −0.00416 | −0.00284 / −0.00277 / −0.00288 |

- **Every CI lies below 0** for every market, comparator, metric and clustering. Log loss behaves
  the same way.
- **Per-threshold wins:** 83/83 against empirical and 80/83 against 4C.1 for primary; 27/27 and
  27/27 for QBs.
- **Seasons:** mean ECE is 0.48–1.46 pp per season. The worst season-threshold is 2.08 pp, and
  the worst |bias| is 1.11 pp.
- **Limitation:** two-way (game × player) clustering was not run. Every one-way interval clears 0
  by a wide margin, so it would not change the conclusion.

### 6. Monotonicity, PMF, tails, exact zero

- **Monotonicity:**
  - All 5,922,280 cells: the minimum difference between adjacent thresholds is 0.0, with 0
    negative differences.
  - `monotonic_repair` is never set, and `p_le == p_le_raw` everywhere, so no repair was
    applied.
  - Ties (zero differences) occur, mostly in saturated non-QB pass-attempt rows (155,235).
  - The run summary prints `max_raw_decrease = -0.0`. This comes from `max(-0.0, 0.0)` in
    `check_monotone`, is cosmetic, and means 0.
- **PMF mass conservation:** there are no negative interval masses, and the maximum
  `|lower + Σ interior + upper − 1|` is 2.2e-16 (`pmf.parquet`). Probabilities lie in
  [0.00036, 1.0].
- **Tails:** the lower tail P(Y ≤ k_min) and upper tail P(Y > k_max) are explicit lumped masses.

| Market | k range | Upper tail: predicted / observed | Rows above k_max |
|---|---|---|---|
| pass_attempts | 0–55 | 0.085% / 0.059% | 22 |
| rush_attempts | 0–26 | 0.29% / 0.22% | 83 |
| receptions | 0–10 | 0.50% / 0.41% | 152 |
| rush_yds | −5–144 | 0.41% / 0.27% | 101 |
| reception_yds | −1–146 | 0.64% / 0.44% | 164 |

  - **Yardage lower tails:** rush yards 0.43% predicted vs 0.30% observed; receiving yards
    0.99% vs 0.90%.
  - **Limitation:** the tails are slightly conservative but small. There is no probability for
    lines beyond the grid; for example, receptions above 10.5 cannot be priced.
- **Exact zero:**

  | Market | cdf_tc P(0): bias, ECE (pp) | Zero model ECE (pp) |
  |---|---|---|
  | pass_attempts | +0.03, 0.28 | 0.38 |
  | rush_attempts | +0.23, 1.06 | 0.61 |
  | receptions | +0.07, 0.92 | 0.80 |
  | rush_yds | +0.58, 1.77 | 0.80 |
  | reception_yds | +0.28, 1.14 | 0.77 |

  - For counts, P(0) is F(0); for yardage it is F(0) − F(−1).
  - Exact-zero probabilities are calibrated (|bias| ≤ 0.6 pp) but worse than the Phase 4C zero
    model, by up to 1 pp of ECE in rush yards.
  - This is not a gate in ADR 0008. Line-at-0.5 probabilities (CDF thresholds) are what the gate
    scores.

### 7. Readiness against ADR 0008

The rule was re-applied from the independent numbers, using the thresholds as written in ADR
0008 and committed in `0a235ef` before any result.

| Market | Independent status | Run status | Failing gates | Gate mismatches |
|---|---|---|---|---|
| pass_attempts | CONDITIONAL | CONDITIONAL | `qb_mean_ece_ok` (4.08 > 3), `qb_worst_threshold_ok` (5.96 > 5) | 0 |
| rush_attempts | READY | READY | — | 0 |
| rush_yds | READY | READY | — | 0 |
| receptions | READY | READY | — | 0 |
| reception_yds | READY | READY | — | 0 |

- **Pass attempts meets CONDITIONAL:** it is leak-free, monotone, has a valid PMF and beats the
  empirical baseline, primary mean ECE is 0.33 ≤ 5 pp, and QB mean ECE is 4.08 ≤ 5 pp.
- **QB seasonal bias:** no persistent bias; the worst single season |bias| is 4.88 pp, with no
  season-threshold at or above 5 pp.
- **Exceptions:** none.
- **Phase 5 mapping:** 4 of 5 markets are READY, which gives **CONDITIONAL GO**.

### 8. Phase 4C.1 immutability

- **Files:** all 11 files match the Phase 4C.1 manifest. Recomputing `content_sha256` gives
  `6e1568a6…`, unchanged.
- **Timestamps:** every file was last modified 2026-10-03 19:29 UTC, before Phase 4C.1b existed.
- **Directory:** no extra files.
- **Official result:** still NOT READY in all five markets, Phase 5 NO-GO. Phase 4C.1b records
  that result unchanged in its manifest.
- **Code:** `phase4c1.py`, `models/cdf.py` and `cdf_cli.py` are unchanged since `ed76ee2`. The
  only change to shared code adds `cdf_tc` to the `cdf_predictions` predictor whitelist
  (contract 0.11.0), which loosens validation without changing any Phase 4C.1 result.

## Conclusion

**CONDITIONAL GO is independently justified.** Every reported number reproduces and every gate
decision follows the pre-registered rule.

**Minimum corrections: none to code, artifacts or decisions.** The following must accompany any
Phase 5 use:

1. **Pass attempts.**
   - Report QB calibration (4.08 / 5.96 pp), never the diluted 0.33 pp, for any pass-attempt
     line.
   - QB pass-attempt lines stay CONDITIONAL. They are miscalibrated in resolution at k = 28–32,
     which covers typical sportsbook lines.
2. **Conditioning.** All probabilities are conditional on participation and roster-conditioned
   (`UNIVERSE_WARNING`). Phase 5 must combine them with a participation or void rule; the
   evaluation is not a live Wednesday-lock backtest.
3. **Coverage.**
   - Lines above k_max or below k_min have no direct CDF value.
   - Exact-zero probabilities are coarser than the Phase 4C zero model.
4. **Cosmetic, optional.** `max_raw_decrease` prints −0.0.
