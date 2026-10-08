# ADR 0008: Phase 4C.1b threshold-conditioned monotone CDF (final Phase 4C gate)

Status: pre-registered, 2026-10-08 (results to be appended after the real-data runs)

## Context

Phase 4C.1 (ADR 0007) is officially NOT READY / Phase 5 NO-GO. That result stands and is not
revised. The audit found two problems:

- **Incoherence.** The 14,300 independent threshold classifiers are incoherent at row level.
- **QB miscalibration.** QB pass attempts are miscalibrated by 6–9 pp.

Phase 4C.1b replaces the independent classifiers with one coherent model. It is the **final**
Phase 4C experiment: there will be no Phase 4C.1c, whatever the result.

Code:

- `lineseek.models.cdf_tc`
- `lineseek.evaluation.phase4c1b`
- `lineseek-cdf-tc` (`lineseek.evaluation.cdf_tc_cli`)

The contract moves to 0.11.0, which only allows predictor `cdf_tc` in `cdf_predictions`.
`TC_CDF_SET_VERSION = "2.0.0"`.

## Decisions (fixed before any Phase 4C.1b result)

1. **Model.** One LightGBM binary classifier per market and fold, for 5 × 110 = 550 fits.
   - Each Phase 4B training row (primary cohort, played, `kickoff + 36h <= decision_time`) is
     repeated once per frozen grid threshold k. Its target is `1{y <= k}`.
   - **The only new input is a numeric `threshold`**, appended to the 42 Phase 3 features.
   - Parameters equal Phase 4C.1's: the Phase 4B configuration with objective `binary`,
     300 rounds, learning rate 0.05, 31 leaves, `min_child_samples` 100, deterministic, seed
     20261002.
   - `monotone_constraints` is +1 on `threshold` and 0 on every other feature, with method
     `basic`.
   - LightGBM runs single-threaded in 2 worker processes. Results do not depend on the worker
     count.
2. **No repair.**
   - The constraint makes every tree non-decreasing in k, so the CDF is monotone by
     construction.
   - A raw decrease above 1e-12 is an error, and the run fails.
   - Smaller decreases are floating-point noise. A running maximum removes them, and each one is
     recorded.
3. **Frozen protocol.** The run reuses:
   - the Phase 4A folds (recomputed and asserted equal);
   - the cohorts;
   - the Phase 4B population and features;
   - the Phase 4C.1 threshold grids, which the warm-up rule must reproduce exactly or the run
     fails;
   - the durable `data/` roots.

   DNP is never a zero, and outcome-only rows are never trained or scored.
4. **Comparators on identical rows** (asserted by key alignment):
   - the Phase 4C.1 independent CDF (`cdf_lgbm`, repaired), as `cdf_indep`;
   - the Phase 4C.1 empirical CDF.

   Both are read from the frozen Phase 4C.1 run, which is never rewritten.
5. **Scoring.**
   - **Rows:** primary cohort, participant rows, not outcome-admitted, 2021–2025.
   - **Informative thresholds:** n(Y ≤ k) ≥ 200, n(Y > k) ≥ 200, and an observed rate in
     [0.05, 0.95]. The rule is fixed per cohort over all evaluation seasons.
   - **Metrics:** Brier, log loss (p clipped to 1e-15) and 20-bin ECE per threshold, cohort and
     season.
   - **Comparisons:** game-clustered bootstrap of the per-row informative-mean Brier and log
     loss differences (Phase 4A reps and seed).
   - **Diagnostics:**
     - an ECE noise floor (200 Bernoulli draws);
     - central 50% / 80% interval coverage (rush attempts in particular);
     - zero consistency against the Phase 4C zero model (yardage: CDF(0) − CDF(−1));
     - yardage coherence (CDF(0) against observed P(Y ≤ 0));
     - PMF validity and tails.
   - **QB-role cohort:** game-day roster QBs, pass attempts only. It is an evaluation diagnostic
     and never a model input.
6. **Readiness rule.**
   - **Every market is READY only if all of the following hold:**
     - Mean informative Brier is below the empirical baseline's, and the game-clustered 95% CI
       of the difference is below 0.
     - Mean informative ECE ≤ 3 pp.
     - Every informative ECE ≤ 5 pp.
     - The CDF is monotone (no raw decrease > 1e-12).
     - It is stable:
       - Brier beats the empirical baseline in ≥ 4 of 5 seasons;
       - every season's mean informative ECE is ≤ 5 pp;
       - no informative threshold has a bias of |b| ≥ 5 pp with the same sign in ≥ 3 seasons.
     - It is leak-free: no fit trained on data unavailable at its decision time.
     - The PMF is valid: no negative mass, and total mass error ≤ 1e-9.
   - **Pass attempts, additionally (QB-role cohort, QB informative thresholds):**
     - QB mean informative ECE ≤ 3 pp;
     - QB worst informative ECE ≤ 5 pp;
     - no persistent QB seasonal bias (same definition as above).
   - **CONDITIONAL:** leak-free, monotone, valid PMF, beats empirical, and mean informative
     ECE ≤ 5 pp. For pass attempts, the QB mean informative ECE must also be ≤ 5 pp.
   - **NOT READY:** otherwise.
   - **Phase 5 (the mapping is unchanged from Phase 4C.1):** GO if all five markets are READY,
     CONDITIONAL GO if at least one is, otherwise NO-GO.
7. **Determinism.** The build runs twice, and the two manifests' `content_sha256` must be equal.
   The hash covers all artifact checksums and excludes the build time.
