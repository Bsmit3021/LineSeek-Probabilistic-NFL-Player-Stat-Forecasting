# ADR 0008: Phase 4C.1b threshold-conditioned monotone CDF (final Phase 4C gate)

Status: accepted, 2026-10-08 (rule pre-registered in `0a235ef` before any result)

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

## Result

Run 1 is `20261008T173929762154Z_0a9365e1dbba` (content `0a9365e1dbba…`): 110 folds and 550
fits. Training used 566,085,000 stacked rows in total and produced 5,922,280 prediction rows.
It took 69 min 52 s wall time (8,111 CPU-s) on 2 workers, with no swap pressure. There were 0
temporal violations, and the run followed the pre-registered protocol and code (`0a235ef`)
without changes.

Determinism: Run 2 (`20261008T213040228527Z_0a9365e1dbba`) produced an identical `content_sha256`
(`0a9365e1dbbaa3592d8dc305325abeeac0c75333b866620c867bb058873080b6`). All 12 artifact files
are byte-identical (`cmp`), and both runs pass `lineseek-cdf-tc validate`. Run 2 used 8,591
CPU-s; its wall time of 3 h 51 min includes pauses while the laptop slept on battery.

### Monotonicity and PMF

- **Raw decreases.** None among the 5,922,280 cells (maximum decrease 0). No floating-point
  cleanup was needed, so no repair of any kind was applied. In Phase 4C.1, 43–100% of rows
  violated monotonicity.
- **PMF.** No negative masses, and the total mass error is ≤ 2.3e-16.
- **Tails outside the grid** are now close to the observed rates:
  - upper tail: 0.08–0.64% predicted vs 0.06–0.44% observed;
  - rush yards P(Y ≤ −5): 0.43% predicted vs 0.30% observed. Phase 4C.1 predicted 0.04%.

### Primary cohort (informative thresholds; ECE and bias in pp)

| Market | Inf. thr | Brier tc / indep / emp | Log loss tc / emp | Mean ECE tc (indep) | Worst ECE tc | Noise floor | Status |
|---|---|---|---|---|---|---|---|
| pass_attempts | 11 | 0.0131 / 0.0135 / 0.0156 | 0.044 / 0.148 | 0.33 (0.54) | 0.52 | 0.19 | CONDITIONAL (QB) |
| rush_attempts | 13 | 0.0424 / 0.0429 / 0.0481 | 0.140 / 0.381 | 0.49 (0.62) | 1.06 | 0.37 | READY |
| receptions | 6 | 0.1003 / 0.1011 / 0.1104 | 0.315 / 0.750 | 0.74 (0.74) | 1.01 | 0.55 | READY |
| rush_yds | 22 | 0.0561 / 0.0565 / 0.0615 | 0.186 / 0.512 | 0.54 (0.67) | 1.16 | 0.42 | READY |
| reception_yds | 31 | 0.1037 / 0.1042 / 0.1131 | 0.327 / 0.766 | 0.71 (0.74) | 1.15 | 0.55 | READY |

**Against the empirical CDF.**

- **Pooled Brier:** better in every market. The game-clustered 95% CI of the per-row
  difference lies wholly below 0 (−0.0024 to −0.0101).
- **Per threshold:** better on Brier at 83 of 83 informative thresholds.

**Against the Phase 4C.1 independent CDF.**

- **Pooled Brier and log loss:** better in every market, with every CI below 0.
- **Per threshold:** better on Brier at 80 of 83 informative thresholds.

**Seasons (2021–2025).**

- Brier beats the empirical CDF in 5 of 5 seasons in every market.
- Per-season mean informative ECE is 0.48–1.46 pp, and the worst single season-threshold is
  2.08 pp.
- No threshold has a persistent ≥ 5 pp bias; the largest single-season |bias| is 1.11 pp.

### QB pass attempts (roster-position QBs, 3,567 rows, 27 informative thresholds)

| | cdf_tc | Phase 4C.1 indep | Empirical |
|---|---|---|---|
| Mean informative ECE (pp) | **4.08** | 6.41 | 7.32 |
| Worst informative ECE (pp) | **5.96** (k = 30) | 8.93 | 8.97 |
| Thresholds > 5 pp | 6 (k = 28–32, 35) | 19 | 24 |
| Brier (informative mean) | 0.1426 | 0.1468 | 0.1586 |
| Log loss | 0.448 | 0.472 | 1.282 |

- **Noise floor:** 1.78 pp.
- **Bias is small.** Mean |bias| is 0.85 pp, and no threshold exceeds 3.7 pp. No threshold shows
  a persistent ≥ 5 pp seasonal bias; the largest single-season |bias| is 4.88 pp.
- **What remains is resolution/dispersion error** at the central lines (k = 28–32), not a level
  shift.
- **Brier:** better than Phase 4C.1 at 27/27 thresholds and better than empirical at 27/27.
- **Seasons:** mean ECE was 6.9, 5.5, 5.3, 5.4 and 4.7 pp for 2021–2025.

**Verdict:** the QB gates require a mean ≤ 3 pp and a worst threshold ≤ 5 pp. **They fail**
(4.08 / 5.96), even though QB calibration improved substantially.

### Diagnostics

- **Central interval coverage** (observed minus predicted mass, pp; 50% / 80%; Phase 4C.1 in
  parentheses):
  - rush attempts: −1.5 / −1.3 (−9.6 / −9.3);
  - QB pass attempts: −6.2 / −3.4 (−17.5 / −16.7);
  - receptions: −1.1 / +0.1;
  - rush yards: 0.0 / −0.2;
  - receiving yards: −0.3 / 0.0.
- **Zero consistency.** CDF(0) is no longer equal to the Phase 4C zero model by construction:
  - **Count markets:** the mean |difference| is 0.8–3.4 pp, and the P95 is 2.7–12.0 pp.
  - **Calibration:** the implied P(Y = 0) is still calibrated (|bias| ≤ 0.23 pp, ECE ≤ 1.06 pp)
    with Brier within ±6e-4 of the zero model's (better for pass attempts and receptions).
- **Yardage coherence.**
  - **CDF(0) against observed P(Y ≤ 0):** +0.56 pp for rush yards and +0.37 pp for receiving
    yards. In Phase 4C.1 they were −0.50 and −1.53, and the near-tie sorting bias is gone.
  - **Implied P(Y = 0) = CDF(0) − CDF(−1):** this point mass is less sharp than in Phase 4C.1.
    It differs from the zero model by a mean of 3.7 / 4.0 pp (P95 13.7 / 12.5 pp). Its ECE is
    1.77 pp for rush yards (Phase 4C.1: 0.76) and 1.14 pp for receiving yards (Phase 4C.1:
    1.21).
  - **Effect on the gate:** none, because the rule scores CDF thresholds. Yardage lines at 0.5
    remain well calibrated, but "exactly 0 yards" probabilities are coarser.

## Decision

| Market | Status |
|---|---|
| pass_attempts | **CONDITIONAL**. It passes every all-player gate, but QB mean informative ECE is 4.08 pp (> 3) and the worst QB threshold is 5.96 pp (> 5). |
| rush_attempts | **READY** |
| rush_yds | **READY** |
| receptions | **READY** |
| reception_yds | **READY** |

**Phase 5: CONDITIONAL GO** (frozen mapping: at least one READY market, not all). Phase 4C ends
here; there is no Phase 4C.1c.

- **Pass attempts:** under this rule it is not READY. Any Phase 5 use must treat it as
  conditional, with the QB miscalibration above disclosed.
- **Phase 4C.1:** its official NOT READY / NO-GO result (ADR 0007) is unchanged.
- **Next phase:** Phase 5 has not been started.
