# ADR 0007: Phase 4C.1 discrete-aware CDF

Status: accepted, 2026-10-03 (result recorded 2026-10-08; frozen status unchanged)

## Context

Phase 4C (ADR 0006) gave Phase 5 a NO-GO: quantiles were miscalibrated by about 10 pp as
probabilities for integer outcomes. Phase 4C.1 models `P(Y <= k | X, participated)` directly at
integer thresholds k, which gives half-point lines `k + 0.5`.

Code (commit `ed76ee2`, committed before any real-data evaluation):

- `lineseek.models.cdf`
- `lineseek.evaluation.phase4c1`
- `lineseek-cdf`

The contract moved to 0.10.0 (`cdf_predictions`), with `CDF_SET_VERSION = "1.0.0"`.

## Decisions

1. **Model.** One LightGBM binary classifier per market, threshold and fold, for 14,300 fits.
   - Training uses the Phase 4B population and the 42 Phase 3 features.
   - Parameters are the Phase 4C zero-model ones.
2. **Grid.** One integer grid per market, taken only from warm-up (2018–2020) outcomes:
   - pass attempts: 31 thresholds;
   - rush attempts: 20;
   - receptions: 10;
   - rush yards: 33;
   - receiving yards: 36.
3. **Monotone repair.**
   - Row-wise rearrangement.
   - Count markets are anchored at CDF(0), so CDF(0) equals the Phase 4C zero model exactly.
4. **Comparators.**
   - Empirical-history CDF.
   - Phase 4C quantile-derived CDF (diagnostic only).
5. **Readiness rule.** Frozen before any result. It includes a monotonicity gate: the share of
   rows with raw violations must be ≤ 0.25.

## Result (run `20261003T192903167544Z_6e1568a6a014`, 1 h 44 min)

- 110 folds, 14,300 fits, 11,844,560 prediction rows, 0 temporal violations.
- Mean ECE was 0.53–0.72 pp ("strong") in every market. Brier beat the empirical CDF in every
  market and in 5/5 seasons.
- Raw monotonicity-violating row shares were:
  - pass attempts: 1.00;
  - rush attempts: 0.996;
  - rush yards: 1.00;
  - receptions: 0.43;
  - receiving yards: 1.00.

  Every market therefore failed the frozen monotonicity gate.

**Official frozen status: NOT READY in all five markets. Phase 5: NO-GO.** This result is final
for Phase 4C.1 and is not revised.

## Audit (analysis only, `docs/audit/phase4c1_run1_audit.md`)

- The all-rows ECE dilutes the QB signal: non-QBs are 91% of pass-attempt rows and predicted
  F ≈ 1.
- **QB pass attempts.** On roster-position QBs, mean informative ECE is 6.41 pp (noise floor
  1.71 pp) and the worst threshold is 8.93 pp. The error is stable across seasons. The QB CDF is
  no better than Phase 4C.
- **Coherence.** Independent classifiers are row-level incoherent. Repairs above 5 pp are common
  in yardage and QB rows. Unanchored sorting biases receiving yards CDF(0) by about −1.5 pp.
- **Next step.** The audit recommends a coherent (shared, monotone) construction. Phase 4C.1b
  (ADR 0008) is that experiment and the final Phase 4C gate.
