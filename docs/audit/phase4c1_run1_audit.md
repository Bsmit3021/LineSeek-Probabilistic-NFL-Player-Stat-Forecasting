# Phase 4C.1 Run 1 audit: discrete CDF calibration (analysis only)

Date: 2026-10-07. No retraining, no change to the frozen readiness rule, no Phase 4C.1b or
Phase 5 work. Inputs: frozen Run 1 artifacts, Phase 4C distribution predictions, labels.
Scripts and raw output (git-ignored): `data/evaluation/phase4c1_audit/<run>/scripts/`
(`audit4c1.py`, `audit4c1_report.py`, `audit4c1_extra.py`, `audit4c1_active.py`) with
`report_output.txt`, `extra_output.txt` and `active_output.txt`. The per-threshold Brier and ECE
reproduce the frozen `cdf_calibration` to within 1e-14.

## 1. Run analysed

- Phase 4C.1 run `20261003T192903167544Z_6e1568a6a014` (CDF_SET 1.0.0, contract 0.10.0).
- Phase 4C run `20261003T094049738795Z_8e29d9e3fc15`, `quantile_lgbm` (manifest hash checked).

## 2. Rows and thresholds

- **Rows:** 36,997 scored rows per market (primary cohort, participants, not outcome-admitted,
  2021–2025, 110 folds). The QB-role subset has 3,567 rows.
- **Thresholds:** 130 in total: pass attempts 31, rush attempts 20, receptions 10, rush yards 33,
  receiving yards 36.
- **Alignment:** the CDF and empirical predictions sit on identical rows (asserted). Each row has
  exactly one value per grid threshold (asserted).

## 3. Informative-threshold rule

Fixed before any per-threshold result was computed. It uses outcome counts only:
`n(Y <= k) >= 200`, `n(Y > k) >= 200`, and `0.05 <= observed rate <= 0.95`.

- This leaves 83/130 primary thresholds (pass 11, rush attempts 13, receptions 6, rush yards 22,
  receiving yards 31) and 27/31 QB thresholds.
- **Calibration error:** 20-bin ECE plus |CITL| (absolute difference between the mean prediction
  and the observed rate).
- **Noise floor:** the ECE a perfectly calibrated forecaster would score with the same
  predictions and row count (parametric bootstrap, 200 draws).
- **Dilution check:** Run 1's per-threshold ECE averages over all rows, so near-certain rows
  dilute it. Non-QBs make up 91% of the pass-attempt rows. Two cuts address this:
  - the QB-role cohort;
  - a post hoc *active* subgroup chosen from predictions only: repaired `F(0) <= 0.25`.

## 4. Mean calibration error (repaired CDF, pp)

| Market | Inf. thr | ECE (all) | ECE (inf.) | Noise floor | \|CITL\| (inf.) | Active-subgroup ECE (floor) |
|---|---|---|---|---|---|---|
| pass_attempts | 11 | 0.53 | 0.54 | 0.18 | 0.21 | **6.78** (1.93) |
| rush_attempts | 13 | 0.61 | 0.62 | 0.36 | 0.04 | 2.41 (1.13) |
| receptions | 6 | 0.66 | 0.74 | 0.54 | 0.23 | 1.65 (0.91) |
| rush_yds | 22 | 0.64 | 0.67 | 0.41 | 0.20 | 2.52 (1.32) |
| reception_yds | 31 | 0.72 | 0.74 | 0.55 | 0.28 | 1.51 (0.97) |
| pass_attempts, QB role | 27 | 5.83 | **6.41** | 1.71 | 1.71 | — |

Excluding QBs, informative-threshold ECE is 0.2–0.4 pp above the noise floor overall and
0.7–1.3 pp above it in the active subgroup.

## 5. Worst informative threshold (ECE)

| Market | k | Support (pos/neg) | Observed | Predicted | ECE | Raw ECE |
|---|---|---|---|---|---|---|
| reception_yds | 0 | 17,173 / 19,824 | 46.42 | 44.89 | 1.70 | 1.03 |
| receptions | 1 | 22,919 / 14,078 | 61.95 | 61.59 | 0.88 | 0.89 |
| rush_yds | 1 | 26,686 / 10,311 | 72.13 | 72.38 | 0.88 | 1.03 |
| rush_attempts | 2 | 29,512 / 7,485 | 79.77 | 79.77 | 0.77 | 0.90 |
| pass_attempts | 27 | 34,978 / 2,019 | 94.54 | 94.32 | 0.74 | 0.70 |
| QB pass_attempts | 31 | 2,073 / 1,494 | 58.12 | 57.04 | **8.93** | 9.20 |

- **Where the errors are:** outside QBs, no primary threshold has ECE above 1.7 pp. The two that
  exceed 1 pp are high-support, central yardage thresholds (receiving yards k = 0 and 1). They
  are not low-support or extreme ones.
- **QBs:** the errors are largest at the central, highest-support thresholds (k = 26–33,
  observed 40–65%).

### 5a. Repair regressions (receiving yards)

Monotone repair raised ECE by more than 0.1 pp only for receiving yards, at three thresholds:

- **k = 0 (1.03 → 1.70 pp).** Support 17,173/19,824; observed 46.42%; raw mean 45.98%; repaired
  44.89%.
  - Repair changed 50.7% of rows, every one of them downward (mean −2.15 pp). On those rows: raw
    47.49%, repaired 45.34%, observed 46.69%. The raw values were 0.8 pp too high; after repair
    they are 1.35 pp too low.
  - The new values came from neighbouring thresholds: k = 1 (11,211 rows), k = 3 (4,945), k = 5
    and beyond.
  - **Cause:** yardage repair is plain rearrangement with no anchor. Only 0.7% of outcomes fall
    in (0, 1], so F(0), F(1) and F(3) are nearly tied. Independent classifier noise crosses them
    in about half of rows. Sorting then always gives k = 0 the minimum of the tied cluster, which
    is a systematic order-statistic bias.
  - Brier still improves very slightly (−1e-5), because repair cuts variance while adding bias.
- **k = 1 (0.83 → 1.04 pp)** and **k = 19 (0.78 → 0.92 pp; |CITL| 0.27 → 0.63):** same mechanism.
  k = 19 and k = 20 are one yard apart.
- **Count markets:** repair is anchored at CDF(0), so there is no k = 0 effect.
- **Rush yards:** no regression above 0.1 pp.

## 6. Versus the empirical baseline (identical rows, informative thresholds)

Mean of CDF minus empirical; negative means the CDF is better.

| Market | ΔBrier ×1e3 | ΔLog loss | ΔECE pp | Brier wins |
|---|---|---|---|---|
| pass_attempts | −2.09 | −0.10 | −0.19 | 11/11 |
| rush_attempts | −5.21 | −0.24 | −0.84 | 13/13 |
| receptions | −9.32 | −0.43 | −2.93 | 6/6 |
| rush_yds | −5.06 | −0.33 | −1.20 | 22/22 |
| reception_yds | −8.98 | −0.44 | −2.99 | 31/31 |
| QB pass_attempts | −11.85 | −0.81 | −0.91 | 25/27 |

- In the active subgroup the CDF wins on Brier at 108/111 informative thresholds.
- The empirical baseline's log loss is inflated by hard 0/1 probabilities from short histories,
  so treat Brier as the primary comparison.

## 7. Versus Phase 4C quantile-derived probabilities

Phase 4C probabilities come from `quantile_approx_cdf` on the same rows and integer thresholds k
(a line of k + 0.5). Values are informative-threshold means.

| Market | 4C \|CITL\|, k>0 (worst) | 4C.1 \|CITL\|, k>0 (worst) | 4C Brier | 4C.1 Brier |
|---|---|---|---|---|
| pass_attempts | 4.65 (4.78) | 0.2 (0.45) | 0.0164 | 0.0135 |
| rush_attempts | 4.04 (5.10) | 0.04 (0.07) | 0.0548 | 0.0429 |
| receptions | 5.72 (9.16) | 0.2 (0.36) | 0.1279 | 0.1011 |
| rush_yds | 2.95 (4.57) | 0.2 (0.45) | 0.0639 | 0.0565 |
| reception_yds | 2.06 (8.45) | 0.3 (0.67) | 0.1095 | 0.1042 |
| **QB pass_attempts** | **1.31 (2.71)** | **1.71 (4.05)** | **0.1441** | **0.1468** |

- **Outside QBs, the ~10 pp Phase 4C failure is fixed.** Mean ECE drops from 3.3–9.4 pp to
  0.5–0.7 pp, and 4C.1 wins on Brier at 81/83 informative thresholds. At k = 0 the 4C quantiles
  were 27–32 pp off.
- **For QBs it is not fixed.** Phase 4C is better on Brier (wins 23 of 27 thresholds), on log loss
  (0.452 vs 0.472) and on ECE (4.62 vs 6.41).
- **Central-interval coverage** (grid intervals; predicted mass vs observed coverage):
  - QB 50% interval: −17.5 pp for 4C.1 vs −12.0 pp for 4C. QB 80% interval: −16.7 vs −17.6.
  - Rush attempts: about −9.5 pp for both.
  - Receptions: −4.4 / −4.8 pp for 4C.1 (4C: −24.6 / −7.2).
  - Yardage: about −2 pp for 4C.1 (4C receiving yards: −23.9 / −4.9).

## 8. Zero consistency

- **Count markets:** CDF(0) equals the Phase 4C zero model exactly (mean, median, P95 and max
  difference all 0), so Brier and ECE are identical. This is true by construction: the k = 0
  classifier is the zero model, and repair is anchored there. It is a consistency check, not
  independent evidence.
  - Zero-model ECE: pass attempts 0.38, rush attempts 0.61, receptions 0.80 pp.
- **Yardage:** compare `CDF(0) − CDF(−1)` (= P(Y = 0)) with the zero model, not CDF(0).

| Market | Mean | Median | P95 | Max | Brier (zero model / CDF) | ECE (zero model / CDF) |
|---|---|---|---|---|---|---|
| rush_yds | 2.04 | 0.72 | 8.64 | 48.7 | 0.0816 / 0.0817 | 0.79 / 0.76 |
| reception_yds | 2.25 | 1.45 | 7.08 | 24.5 | 0.1398 / 0.1395 | 0.77 / 1.21 |

- The two estimates agree on average but differ by more than 7 pp in 5% of rows. That is the same
  row-level incoherence that shows up as repair size.
- Yardage CDF(0) against observed P(Y ≤ 0): −0.50 pp for rush yards, −1.53 pp for receiving yards
  (this is the k = 0 repair bias from §5a).

## 9. PMF validity (repaired CDF, `P(k_i) = F(k_i) − F(k_{i−1})`)

- **Valid:** no negative masses (the smallest is 0). The largest single mass is 0.99997, the
  largest F is 0.9999996, and masses sum to 1 within 2.2e-16.
- **Mass on the grid:** the grid holds everything except the lower and upper tails.
  - On sparse stretches of the grid (for example pass attempts 3 → 13), "masses" are interval
    masses, not point masses.
- **Tails outside the grid are underpredicted by 2–20×, but they are small in absolute terms:**
  - Lower tail: P(Y ≤ −5) for rush yards is 0.04% predicted vs 0.30% observed; P(Y ≤ −1) for
    receiving yards is 0.38% vs 0.90%.
  - Upper tail (above k_max): 0.003–0.12% predicted vs 0.06–0.44% observed.

## 10. QB pass attempts (roster-position QBs)

- **Calibration:** mean informative ECE is 6.41 pp (noise floor 1.71 pp). The worst threshold is
  k = 31 at 8.93 pp. All 27 informative thresholds exceed twice the noise floor.
- **Pattern:** overconfident and S-shaped. Pooled pass-attempt reliability (mid-range bins are almost all QB rows) shows predictions of 0.05–0.25
  occurring 8–11 pp more often than predicted, and predictions of 0.70–0.95 occurring 3–5 pp less
  often. The model also leans persistently negative at k = 20–26 (it underpredicts low counts)
  and positive at k = 42–44.
- **Seasons:** mean ECE of 9.2 / 7.6 / 7.0 / 7.3 / 6.0 pp for 2021–2025. The worst threshold in
  every season sits at 9–12 pp.
- **Repair:** 64.6% of QB rows have a row-max repair above 10 pp (mean 13.6 pp).
- **Conclusion:** the QB CDF is not better calibrated than the Phase 4C QB quantiles; it is
  slightly worse.

## 11. Season stability (informative thresholds)

- **Primary cohort:** no persistent bias, no one-season anomalies, no sign reversals.
  - Worst |CITL| in any season is 2.23 pp (receiving yards k = 0, 2021; 2.07 pp in 2022).
  - Per-season mean ECE is 0.56–1.74 pp, highest in 2021 for every market.
- **QB cohort:** 9 persistent-bias thresholds, 2 sign reversals (k = 33 and 34), and 12
  thresholds above 3 pp in at least one season.

## 12. Monotone repair severity (row-max |repaired − raw|, all scored rows)

| Market | Mean | P95 | P99 | >1 pp | >2.5 pp | >5 pp | >10 pp |
|---|---|---|---|---|---|---|---|
| pass_attempts | 1.38 | 12.5 | 23.8 | 11.0% | 9.8% | 8.8% | 6.3% |
| ↳ QB rows | 13.6 | 28.4 | 36.2 | 96.9% | 94.0% | 87.7% | 64.6% |
| ↳ non-QB rows | 0.08 | 0.07 | 2.1 | 1.9% | 0.8% | 0.3% | 0.1% |
| rush_attempts | 1.73 | 10.4 | 18.4 | 25.1% | 18.9% | 12.7% | 5.4% |
| receptions | 0.31 | 1.6 | 7.1 | 6.5% | 3.5% | 1.7% | 0.5% |
| rush_yds | 3.57 | 13.5 | 20.1 | 53.2% | 39.5% | 27.0% | 10.9% |
| reception_yds | 4.86 | 12.2 | 16.7 | 86.0% | 68.1% | 40.6% | 10.1% |

- Repair hardly moves calibration in aggregate: Brier and pooled ECE fall slightly.
- But it changes individual probabilities a lot: by more than 5 pp in 27–41% of yardage rows and
  88% of QB rows.
- **Yardage is materially less coherent than receptions.** QB pass attempts is the least coherent
  slice of all.

## 13. Key defects

1. **QB pass attempts: overconfident by 6–9 pp at central lines, stable across all five seasons.**
   It is no better than Phase 4C. Run 1's per-threshold ECE hides this because non-QBs are
   predicted F ≈ 1 at every threshold.
2. **Active-subgroup calibration is 1.5–2.5 pp for the other markets** (about 1 pp above the
   noise floor). Rush attempts central intervals undercover by about 9.5 pp, so tail mass is too
   thin for high-usage rushers.
3. **Independent per-threshold classifiers are row-level incoherent.** Repairs above 5 pp are
   common in yardage and QB rows. Unanchored rearrangement adds an order-statistic bias at clusters
   of near-tied thresholds: receiving yards CDF(0) is 1.5 pp low.
4. **Tails outside the grid are underpredicted by 2–20×.** Absolute mass is at most 0.5 pp.
5. **Yardage CDF(0) − CDF(−1) differs from the zero model by more than 7 pp in 5% of rows.**

No implementation bug was found. The analysis script only gained alignment asserts.

## 14. Official frozen status

**Official frozen Phase 4C.1 status: NOT READY / Phase 5 NO-GO** (all five markets). Unchanged.

## 15. Diagnostic assessment: **PROMISING FOR 4C.1b**

| Question | Answer |
|---|---|
| A. Calibration within 1–3 pp at informative thresholds? | Yes for rush attempts, receptions, rush yards and receiving yards (0.6–0.7 pp overall, 1.5–2.5 pp in the active subgroup). No for QB pass attempts (6.4 pp). |
| B. Did 4C.1 fix the ~10 pp Phase 4C problem? | Yes outside QBs. No for QBs. |
| C. Does it beat the empirical baseline? | Yes: 83/83 primary informative thresholds on Brier, and 25/27 for QBs. |
| D. Where are the worst errors? | Central, high-support thresholds (receiving yards k = 0–1; QB k = 26–33), not low-support extremes. |
| E. Is repair mostly correcting harmless noise? | Aggregate scores barely move, but row-level probabilities change materially for yardage and QBs. |
| F. Is yardage less coherent than count markets? | Yes (except the QB slice). |
| G. Is the PMF valid? | Yes. |
| H. Is CDF(0) consistent for count markets? | Yes, exactly, by construction. |
| I. Are QB pass-attempt probabilities good enough to continue as is? | No. |
| J. Is the method promising enough for 4C.1b? | Yes for four markets. Pass attempts needs a QB-specific redesign. |

**Inputs for a pre-registered 4C.1b** (proposals, not rules):

- judge readiness on informative thresholds, measured as excess over the noise floor and including
  the QB and active subgroups;
- replace the all-rows monotonicity flag with a repair-severity measure (for example P95 row-max
  repair);
- fit a QB-specific or recalibrated pass-attempts CDF;
- use a more coherent CDF construction, such as shared or ordinal models or anchored/isotonic
  repair, to remove the near-tie bias.
