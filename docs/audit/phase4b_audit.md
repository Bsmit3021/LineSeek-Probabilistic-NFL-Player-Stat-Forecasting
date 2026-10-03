# Phase 4B audit: are the LightGBM gains genuine?

- **Date:** 2026-10-03.
- **Audited:** `b447e93`. The tree was clean; 212 tests passed; Ruff was clean.
- **Runs:** Phase 4A content `c21d61557c9d` (folds `db03f6ee4710`) and Phase 4B content
  `064978d1cc1c`.
- **Code:** unchanged. The only change is a correction to ADR 0005.

**Result: CONDITIONAL GO.** The LightGBM gains are reproducible, leak-free, stable across seasons
and pass the frozen rule. They are mainly a gain from forecasting the *median*, which MAE
rewards; they are not mainly better information about the mean.

## Method

- **Independent fold reproduction** (no `lineseek` imports):
  - Training sets were rebuilt from raw feature and label Parquet files, with outcome-only rows
    taken from the independent Phase 3 audit classification.
  - The feature list and model configurations were hand-written from the ADR.
  - Seven folds (2021 Wk1, 2021 Wk16 with Tuesday games, 2022 Wk12, 2023 Wk14, 2024 Wk11,
    2025 Wk14, the Super Bowl) × all five markets × both models.
- **Independent metrics:** recomputed from the prediction, label and baseline artifacts, with
  common support rebuilt independently, the QB flag taken from raw roster rows, a separate
  game-clustered bootstrap (seed 7), and the frozen promotion rule reapplied from its written
  definition.
- **Diagnostic L2 LightGBM:** `objective="regression"`, otherwise the same folds, features and
  parameters (550 fits). It was not promoted, and nothing was changed.
- **Real-data mutations:** run through the audited `build_tasks`/`run_market` on folds 60–62.

## Results

| Check | Result |
|---|---|
| Folds | Mine equal the frozen 110; Phase 4B uses exactly those fold IDs and lock times |
| Fold reproduction | Training row counts identical; LightGBM predictions identical (0.0); Ridge ≤ 2e-10; imputer statistics equal the training-row medians; training rows always available by the lock, primary cohort, played, no DNP zeros, no outcome-only rows |
| Metrics | MAE identical to the artifact; status, seasons won and the promotion classification agree; intervals agree within 0.006 |
| Common support | Identical rows for every predictor in every market (33,741); 0 outcome-only rows in primary; on the 16,280 rows the benchmark cannot predict, LightGBM is still best, so nothing is gained by dropping rows |
| Seasons | LightGBM beats the benchmark in 5/5 seasons in all 5 markets |
| QB `pass_attempts` | Reproduced exactly from roster position only (n = 3,150): benchmark 8.181, Ridge 7.932, LightGBM 7.558 |
| Mutations | Target-fold outcomes changed: folds 60–61 byte-identical (fold 62 changes legitimately). Future labels and features changed: earlier folds identical. Historical DNP features scrambled: identical. DNP flipped to played: enters training. Team codes renamed: LightGBM identical, Ridge ≤ 1.7e-12 (row order only, also seen on a pure shuffle). Repeat run: identical |
| Determinism | Three full runs share content `064978d1cc1c`; repeated subset runs are byte-identical |
| Importance | No identifier or calendar proxies. Calendar and rest features take ≤ 7% of gain, history counts ≤ 3%. The top feature is stable: `target_share_roll5` in 95/110 receptions folds, `market_prev` in 110/110 `pass_attempts` folds |

### Median vs mean

Primary cohort, participant track, common support.

| Market | Actual mean / median | Bias: benchmark / L1 / L2 | MAE: benchmark / L2 / L1 | RMSE: benchmark / Ridge / L1 / L2 |
|---|---|---|---|---|
| pass_attempts | 2.58 / 0 | −0.05 / −0.05 / +0.06 | 0.771 / 0.749 / 0.710 | 3.40 / 3.23 / 3.14 / 3.13 |
| rush_attempts | 2.11 / 0 | −0.02 / −0.31 / +0.01 | 1.071 / 1.091 / 1.002 | 2.43 / 2.38 / 2.41 / 2.40 |
| rush_yds | 9.24 / 0 | −0.43 / −2.25 / +0.10 | 6.251 / 6.341 / 5.686 | 15.04 / 14.34 / 14.73 / 14.51 |
| receptions | 1.70 / 1 | 0.00 / −0.26 / +0.02 | 1.055 / 1.052 / 0.988 | 1.60 / 1.54 / 1.57 / 1.55 |
| reception_yds | 18.60 / 5 | +0.10 / −4.66 / +0.13 | 13.404 / 13.323 / 12.243 | 21.70 / 21.02 / 21.61 / 21.19 |

- Targets are right-skewed (skew 1.6–3.5) and zero-heavy.
- The L1 median residual is about 0 in every market, as expected for a median forecast.
- The L1-vs-L2 objective accounts for 64% (`pass_attempts`) and 93–130% (other markets) of
  LightGBM's MAE gain.
- With a squared-error objective, LightGBM does not beat the benchmark on MAE for
  `rush_attempts` or `rush_yds`, ties for `receptions`, and gains only slightly for
  `reception_yds` and `pass_attempts`.

## Findings

| Severity | Finding | Status |
|---|---|---|
| High (interpretation) | ADR 0005 said LightGBM's gain was mostly signal rather than loss alignment. The controlled L2 experiment shows the opposite | **Fixed:** ADR 0005 corrected |
| Medium (protocol design) | The Phase 4A benchmarks are all mean-type while the primary metric (MAE) favours medians, so the frozen promotion rule cannot separate estimand alignment from signal | Known limitation; Phase 4C must use proper scoring rules |
| Medium (operational) | Every evaluation input and frozen artifact used here lives under `/private/tmp`. The macOS cleaner already deleted the 2024–2025 raw *manifest* files on 2026-10-03; the Parquet files survived and verify against the feature-manifest checksums | **Resolved 2026-10-03**: see Storage below |
| Low | `predict_models` assumes the full fold set; a partial set fails loudly at the contract | Fail-safe; noted |
| Low | Ridge output depends on training-row order at the ~1e-10 level; the order is deterministic. LightGBM is order-independent while training sets stay under 200,000 rows (`bin_construct_sample_cnt`); the maximum now is about 58,000 per market | Noted |
| Low | Ridge's `pass_attempts` REJECT comes from a P90 ratio on a near-zero base | Known |
| Known limitations | F1 (roster-conditioned universe), F2 (revision risk), F3 (snap coverage per game) | Unchanged |

## Implications for Phase 4C

- **Treat LightGBM's L1 output as a conditional median.** It sits 2.3 rushing yards and 4.7
  receiving yards below the mean.
  - Do not use it as an expected value.
  - Do not rescale it into one without a calibrated distribution.
  - Do not assume the mean-type benchmarks' MAE ranking transfers to probability quality.
- **Zero inflation:** the median is 0 for most `rush_attempts`, `rush_yds` and
  `pass_attempts` rows. A predictive distribution needs an explicit mass at zero, or a
  participation/usage component.
- **Compare candidates with proper scoring rules** (for example pinball/CRPS) on the frozen folds
  and cohorts.
- **Keep the mean-type learned models (Ridge, L2) in view.** They have better RMSE.

## Storage (resolved 2026-10-03)

The canonical artifacts now live in the repository's git-ignored `data/` tree. Nothing Phase 4C
depends on is read from `/private/tmp`.

```
data/raw/                      snapshots (the canonical 2026-09-28 20:41-20:56Z pulls are the latest per source/season)
data/labels/prop_labels/       labels 2018-2025
data/features/player_game_features/   features 2018-2025
data/evaluation/phase4a/20261002T230720728867Z_c21d61557c9d/   frozen Phase 4A run (copied byte for byte)
data/evaluation/phase4b/20261003T075455758784Z_064978d1cc1c/   Phase 4B run rebuilt from data/ (same content)
data/evaluation/phase4b_audit/  audit evidence + scripts, with MANIFEST.json checksums
```

- Every file was copied, not moved, and checksum-verified.
- The ten raw manifests for 2024–2025 that the macOS cleaner deleted were **reconstructed** and
  carry a `"reconstructed"` block. Their data checksums equal those recorded in the feature and
  label manifests. `git_revision`, `contract_version`, `schema_variant` and `temporal_quality` are
  taken from a sibling pull made in the same backfill and are marked unverified.
- `data/MIGRATION_2026-10-03.{py,json}` is the copy script and its log; both are git-ignored.
- Rebuilding from `data/` alone reproduces every artifact exactly:
  - labels and features are byte-identical for all 8 seasons;
  - Phase 4A content is `c21d61557c9d`;
  - Phase 4B content is `064978d1cc1c`.
- The `/private/tmp` copies were left in place and are no longer authoritative.
