# ADR 0006: Phase 4C distributional forecasting and calibration

Status: accepted, 2026-10-03

## Context

The Phase 4B audit (`docs/audit/phase4b_audit.md`) showed that the selected LightGBM point model
(`objective="l1"`) estimates the **conditional median**. Most of its MAE advantage comes from
targeting the median, which MAE rewards; it is not an expected value. Over/under probabilities
(Phase 5) need a calibrated conditional *distribution*, not a single point.

Phase 4C builds and evaluates that distribution:

- on the frozen protocol (110 folds, primary cohort, Phase 4B training population, the audited
  42-feature matrix);
- without sportsbook data and without tuning;
- from durable `data/` roots only.

Code:

- `lineseek.models.distribution` (models, crossing repair, empirical baseline);
- `lineseek.evaluation.scoring` (pinball, WIS, calibration);
- `lineseek.evaluation.phase4c` (orchestration);
- `lineseek.evaluation.distributions_cli` (`lineseek-distributions`).

Contract 0.9.0 adds `distribution_predictions`.

## Decisions

1. **Why quantiles.** The targets are integer-valued, zero-heavy (a median of 0 for most
   `pass_attempts`, `rush_attempts` and `rush_yds` rows), right-skewed (skew 1.6–3.5), and
   yardage can be negative. No single parametric family fits all five markets, so no Normal,
   Gamma, Poisson or negative-binomial distribution is imposed. The distribution is represented
   by five conditional quantiles: Q10, Q25, Q50, Q75 and Q90.
2. **Components** (`DISTRIBUTION_SET_VERSION = "1.0.0"`): per market and fold, on Phase 4B's
   exact training rows and features. Every parameter equals the audited Phase 4B LightGBM
   configuration except the objective: 300 rounds, learning rate 0.05, 31 leaves,
   `min_child_samples` 100, `reg_lambda` 1, no subsampling, deterministic, `force_row_wise`,
   1 thread, seed 20261002.
   - **Quantile LightGBM:** `objective="quantile"`, `alpha` ∈ {0.10, 0.25, 0.50, 0.75, 0.90}.
   - **`mean_lgbm`:** `objective="regression"` (squared error), the conditional-mean
     comparator. It is not automatically selected.
   - **Zero model:** `LGBMClassifier(objective="binary")` for P(actual == 0 | X,
     participated), trained on participant rows only. DNP is never a zero, and no roster,
     label-status or DNP information is a feature.

   That is 7 components × 5 markets × 110 folds = 3,850 fits.
3. **Crossing.** Raw quantile crossings are recorded first: the crossing flag, the maximum
   repair, and WIS before and after. Each row is then repaired by monotone rearrangement
   (sorting; Chernozhukov, Fernández-Val & Galichon 2010).
4. **Domain.** Count markets (`pass_attempts`, `rush_attempts`, `receptions`) clip quantiles
   and the mean at 0 after repair. Yards are never clipped, and nothing is rounded.
5. **Empirical baseline** (the naive probabilistic benchmark, not optimised):
   - It uses the player's Phase 3 eligible history E, via the same `_window` as the feature
     store: `available_at <= decision_time`, the target season or the one before, appearances
     only.
   - Quantiles of E use linear (type 7) interpolation; mean = mean(E); p_zero = the share of
     zeros in E.
   - When E is empty, it falls back to the same statistics of the fold's training targets for
     the market.
   - |E| must equal the feature row's `prior_games_n`.
6. **Scores** (primary cohort, participant rows; the distribution is conditional on playing):
   - **Pinball loss:** ρ_τ(y, q) = (1{y ≤ q} − τ)(q − y).
   - **WIS, the primary aggregate** (Bracher, Ray, Gneiting & Reich 2021, eq. 1–2):
     WIS = 1/(K+½)·(½|y − m| + Σ_k (α_k/2)·IS_{α_k}), with K = 2: the 50% interval [Q25, Q75]
     (α = 0.5) and the 80% interval [Q10, Q90] (α = 0.2). The interval score is
     IS_α = (u − l) + 2/α·(l − y)·1{y < l} + 2/α·(y − u)·1{y > u}. This equals
     2/(2K+1)·Σ pinball (eq. 4), and a test checks the identity. The raw, crossing scores use
     that pinball form.
   - **Calibration for integer outcomes** (Czado, Gneiting & Held 2009). The ordinary PIT is
     not uniform for discrete data, so the coverage of a quantile is the bracket
     [P(y < Q), P(y ≤ Q)], and for an interval it is [P(l < y < u), P(l ≤ y ≤ u)]. The
     calibration error is the distance from the nominal level to the bracket (0 inside it). The
     midpoint is also reported. When ties are heavy the bracket is wide; it is always reported
     with the error.
   - **Zero mass:** Brier score, log loss (p clipped to [1e-15, 1 − 1e-15]), calibration bins
     fixed in advance (0, .05, .1, .2, …, .9, .95, 1), and the observed vs predicted zero rate.
   - **Uncertainty:** the WIS delta vs the baseline, with 2,000 game-cluster resamples
     (seed 20261002).
7. **Calibration tolerances** (fixed before any result): aggregate interval error ≤ 3 pp is
   strong, 3–5 pp acceptable, more than 5 pp material. A per-season error above 7 pp is
   investigated.
8. **Promotion rule** (per market; fixed in `phase4c.py` before any result).
   - **PROMOTE** requires all of the following:
     1. lower WIS than the empirical baseline, with the game-clustered 95% CI entirely below 0;
     2. lower pinball loss on at least 4 of the 5 quantiles;
     3. identical row coverage;
     4. aggregate 50% and 80% interval error ≤ 5 pp;
     5. every season's interval error ≤ 7 pp;
     6. a raw crossing rate ≤ 10%;
     7. no integrity violation.
   - **PROMISING:** lower WIS, but not every condition holds.
   - **NO IMPROVEMENT:** WIS not lower than the baseline.
   - **REJECT:** an integrity failure, a crossing rate above 10%, or aggregate interval error
     above 10 pp.
9. **Separation and determinism.** Predictions are serialised and checksummed before labels
   are joined. The artifact has no outcome columns, and the contract checks the grain, the
   quantile order, the count domain and that p_zero ∈ [0, 1]. Each market × component task is a
   pure single-threaded function of its own data, so the output does not depend on the worker
   count (a test checks this).
10. **Versions.**
    - `CONTRACT_VERSION` 0.8.0 → 0.9.0, because a new artifact contract was added.
    - `FEATURE_SET_VERSION`, `BASELINE_SET_VERSION` and `MODEL_SET_VERSION` are unchanged at
      1.0.0.
    - `DISTRIBUTION_SET_VERSION` 1.0.0. Major: a quantile level, component, objective, training
      population or repair/clipping rule changed. Minor: a component or diagnostic added.
      Patch: a correction that preserves semantics.

## Research consulted

| Source | Influenced |
|---|---|
| LightGBM Parameters documentation | `objective="quantile"` with `alpha`; `regression`; `binary`; determinism settings |
| Bracher, Ray, Gneiting & Reich (2021), *PLOS Comput Biol* 17(2): e1008618 | WIS definition, weights w0 = ½ and w_k = α_k/2, the 1/(K+½) normaliser, and the pinball equivalence (eq. 4). Checked against the paper; a secondary summary had the weights wrong |
| Czado, Gneiting & Held (2009), *Biometrics* 65(4) | The PIT is not uniform for count data, hence bracketed calibration |
| Chernozhukov, Fernández-Val & Galichon (2010), *Econometrica* | Monotone rearrangement (sorting) to repair crossing quantiles |

## Results

From the frozen real-data run, content `8e29d9e3fc15`, built on Phase 4A `c21d61557c9d` and
Phase 4B `064978d1cc1c`.

- **Scale:** 110 folds, 3,850 fits, 455,560 prediction rows, 0 temporal violations.
- **Scored rows:** primary cohort, participant rows, 2021–2025, n = 36,997 per market.

| Market | Empirical WIS | Quantile LGBM WIS | Δ (95% CI) | Mean pinball (emp. → LGBM) | 50% cover (open–closed) | 80% cover (open–closed) | Frozen status |
|---|---|---|---|---|---|---|---|
| pass_attempts | 0.649 | 0.559 | −0.090 (−0.105, −0.076) | 0.324 → 0.279 | 0.04–0.94 | 0.06–0.97 | PROMOTE |
| rush_attempts | 0.809 | 0.726 | −0.083 (−0.091, −0.075) | 0.405 → 0.363 | 0.14–0.79 | 0.23–0.91 | PROMOTE |
| rush_yds | 4.486 | 4.126 | −0.360 (−0.401, −0.321) | 2.243 → 2.063 | 0.14–0.80 | 0.26–0.90 | PROMOTE |
| receptions | 0.743 | 0.686 | −0.057 (−0.061, −0.053) | 0.371 → 0.343 | 0.28–0.67 | 0.45–0.89 | PROMOTE |
| reception_yds | 9.217 | 8.590 | −0.627 (−0.678, −0.577) | 4.609 → 4.295 | 0.27–0.67 | 0.46–0.87 | PROMOTE |

- **Skill.** Quantile LightGBM beats the empirical baseline in every market and every season,
  at all five quantile levels, on identical rows. It also wins on the history-only rows.
- **Zero model (calibrated).**
  - Observed vs predicted zero rate: `pass_attempts` 0.907/0.908, `rush_attempts`
    0.681/0.682, `rush_yds` 0.690/0.691, `receptions` 0.450/0.446, `reception_yds`
    0.455/0.452.
  - The largest bin gap (bins with n ≥ 200) is ≤ 2.3 pp.
  - Brier beats the empirical baseline in 24 of 25 market-seasons, and log loss is 2.5–4×
    lower.
  - QB-role `pass_attempts`: zero rate 0.079 vs 0.098 predicted, Brier 0.072.
- **Mean vs median.**
  - Mean bias: Q50 −0.07 / −0.34 / −2.47 / −0.27 / −4.64; `mean_lgbm` +0.06 / +0.02 / +0.14 /
    +0.03 / +0.26 (pass attempts, rush attempts, rush yds, receptions, rec yds).
  - Q50 has the lower MAE and `mean_lgbm` the lower RMSE.
  - Q50 tracks the Phase 4B L1 median closely (correlation 0.996–0.999, mean |Δ| 0.06–0.56).
    It is not identical: LightGBM's `l1` and `quantile` objectives are implemented
    differently.
- **Crossing.** 2.4–4.5% of rows cross before repair, with a mean repair of 0.04–0.85 and a WIS
  change ≤ 0.002. Not material.
- **Clipping.** Count quantiles are clipped in 883–1,760 rows per market, and the mean in
  12,764 `pass_attempts` rows (squared error gives non-QBs negative values).

### Calibration: material quantile-level miscalibration (integer outcomes)

- **The frozen interval tests pass only weakly.** Ties make the discrete brackets wide; for
  example, the `pass_attempts` 50% interval bracket is [0.04, 0.94]. The bracket test is a
  necessary condition, not a sufficient one.
- **Individual quantiles are materially miscalibrated**, consistently in every season (9–12 pp
  in each season):

  | Market | Quantile | [P(y < Q), P(y ≤ Q)] | Error |
  |---|---|---|---|
  | receptions | Q50 | [0.605, 0.625] | 10.5 pp |
  | reception_yds | Q50 | [0.609, 0.633] | 10.9 pp |
  | rush_attempts | Q75 | [0.856, 0.863] | 10.6 pp |
  | rush_yds | Q75 | [0.861, 0.872] | 11.1 pp |

  Q75 for receptions and receiving yards is off by about 4 pp.
- **Mechanism (diagnosed).** The pattern is present **in-sample** on training rows too, so it is
  not shift or leakage. For integer outcomes, pinball loss is piecewise-linear and its optimum
  is an integer, but boosted predictions land slightly *above* integers (97–100% of the flagged
  quantiles are non-integer). A Q50 of 1.03 therefore implies P(y ≤ 1) = 0.5 when the truth is
  0.61. Evaluated at its integer floor, every bracket contains the nominal level (a post-hoc
  diagnostic only, not adopted).
- **Inconsistency with the zero model.** On count markets, the zero model and the quantiles
  disagree about where the zero mass sits: 42% of `receptions` rows at Q50, and 62%/50% of
  `rush_attempts` rows at Q75/Q90.
- **QB-role `pass_attempts` intervals are too narrow.** The 50% interval covers 0.39–0.42
  (error 7.8 pp) and the 80% interval 0.65–0.71 (error 9.2 pp). Q90 is off by 5.6 pp.

## Selection

- **Median signal:** Phase 4B L1 LightGBM. Q50 is equivalent.
- **Mean signal:** `mean_lgbm` (bias ≈ 0). The Phase 4B audit found Ridge's RMSE comparable, so
  keep both in view.
- **Quantile distribution:** quantile LightGBM. It has real skill, but its quantile-level
  calibration is not threshold-ready.
- **Zero probability:** the LightGBM classifier (calibrated).

## Phase 5 readiness: NO-GO

- **Ready:** P(y = 0), which serves 0.5 lines.
- **Not ready:** P(y > line) at the half-integer lines typical of props. Those lines sit near Q50
  and Q75, where the quantiles misstate the CDF by about 10 pp. For quarterbacks the intervals
  are also too narrow.

**Remediation (Phase 4C.1, to be pre-registered before it is evaluated):** a discrete-aware CDF
layer.

- One option is to estimate P(y ≤ k) directly at integer thresholds, extending the calibrated
  zero classifier.
- Another is a recalibration of quantile levels fitted on training folds only.
- Either way, the zero mass and the quantiles must be made consistent.
- It must be evaluated on the same 110 folds with PIT/bracket diagnostics, and per threshold.

## Limitations

- **F1:** the primary cohort is conditioned on the game-day roster.
- **F2:** historical nflverse statistics come from later snapshots, so revision leakage cannot
  be ruled out.
- **F3:** snap coverage is tracked per game, not per team.
- Ten 2024–2025 raw manifests were reconstructed; their data checksums are verified.
- The distribution is conditional on participation: P(play) is not modelled.
- Five quantiles represent the body, not the extreme tails; outside [Q10, Q90] the shape is
  unknown.
