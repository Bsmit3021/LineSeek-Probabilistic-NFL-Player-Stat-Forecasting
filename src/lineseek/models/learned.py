"""Phase 4B learned point models: Ridge and LightGBM, one per market per fold; docs/adr/0005.

Estimators only ever see (X_train, y_train, X_test). X is built from a frozen list of Phase 3
feature columns; identifiers, timestamps, provenance and label columns are refused. All
preprocessing is inside the fitted pipeline, so imputation and scaling statistics come from the
fold's training rows only. Parameters are fixed here, before any evaluation, and never tuned on
2021-2025 outcomes.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from lineseek.contracts import PROP_MARKETS
from lineseek.contracts.artifacts import ARTIFACT_SCHEMAS, LABEL_ONLY_COLUMNS
from lineseek.features.player_game import FEATURE_COLUMNS

MODEL_SET_VERSION = "1.0.0"
MODELS = ("ridge", "lightgbm")
SEED = 20261002
# Every Phase 3 feature column, in contract order (numeric and boolean only).
FEATURES: tuple[str, ...] = tuple(FEATURE_COLUMNS)
# Diagnostic ablation (LightGBM only): the player's own history of the row's market.
HISTORY_ONLY_FEATURES: tuple[str, ...] = (
    "prior_games_n", "market_prev", "market_roll3_mean", "market_roll3_n", "market_roll5_mean",
    "market_roll5_n", "market_roll8_mean", "market_roll8_n", "market_season_mean",
    "market_season_median", "market_season_std", "market_season_games")
# Never model inputs: keys, timestamps, provenance, labels, cohorts, folds.
FORBIDDEN = frozenset({*ARTIFACT_SCHEMAS["features"], *LABEL_ONLY_COLUMNS, "season", "week",
                       "player_name", "player_display_name", "admission_path",
                       "outcome_admitted", "qb_role", "fold_id", "y", "baseline", "model"})
COUNT_MARKETS = frozenset(k for k, m in PROP_MARKETS.items() if m.nonnegative)

RIDGE_PARAMS: dict[str, Any] = {"alpha": 1.0}
IMPUTER_PARAMS: dict[str, Any] = {"strategy": "median", "keep_empty_features": True}
# Conservative, fixed before evaluation. L1 objective because MAE is the primary metric;
# deterministic + force_row_wise + one thread + fixed seed for reproducible CPU training; no
# bagging or feature subsampling; fixed rounds (no early stopping on any evaluation data).
LGBM_PARAMS: dict[str, Any] = {
    "objective": "l1",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "reg_lambda": 1.0,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "max_bin": 255,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "random_state": SEED,
    "verbose": -1,
}


class ModelInputError(ValueError):
    pass


def feature_matrix(df: pl.DataFrame, features: tuple[str, ...] = FEATURES) -> np.ndarray:
    """float64 matrix of exactly `features` (nulls -> NaN); refuses any forbidden column."""
    bad = sorted(set(features) & FORBIDDEN)
    unknown = sorted(set(features) - set(FEATURE_COLUMNS))
    if bad or unknown:
        raise ModelInputError(f"not model features: forbidden={bad} unknown={unknown}")
    return df.select(pl.col(c).cast(pl.Float64) for c in features).to_numpy()


def make_ridge() -> Pipeline:
    """Median imputation + fixed-width missing indicators -> standardisation -> Ridge."""
    return Pipeline([
        ("features", FeatureUnion([
            ("impute", SimpleImputer(**IMPUTER_PARAMS)),
            ("missing", MissingIndicator(features="all")),
        ])),
        ("scale", StandardScaler()),
        ("ridge", Ridge(**RIDGE_PARAMS)),
    ])


def make_lightgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**LGBM_PARAMS)


def make_model(name: str):
    if name == "ridge":
        return make_ridge()
    if name in ("lightgbm", "lightgbm_history_only"):
        return make_lightgbm()
    raise ModelInputError(f"unknown model {name!r}")


def fit_predict(name: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    """Fit on the training rows only and predict the test rows; returns (raw, fitted model)."""
    if len(x_train) != len(y_train) or np.isnan(y_train).any():
        raise ModelInputError("training targets must be complete and aligned")
    model = make_model(name).fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64), model


def clip(market: str, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count markets cannot be negative: prediction = max(0, raw). Yards are never clipped."""
    if market in COUNT_MARKETS:
        return np.maximum(raw, 0.0), raw < 0
    return raw.copy(), np.zeros(len(raw), dtype=bool)


def ridge_coefficients(model: Pipeline, features: tuple[str, ...]) -> dict[str, float]:
    """Standardised coefficients: imputed features, then their missing indicators."""
    coef = model.named_steps["ridge"].coef_
    names = [*features, *(f"missing:{f}" for f in features)]
    return dict(zip(names, coef.tolist(), strict=True))


def lightgbm_gain(model: lgb.LGBMRegressor, features: tuple[str, ...]) -> dict[str, float]:
    gain = model.booster_.feature_importance(importance_type="gain")
    return dict(zip(features, gain.tolist(), strict=True))
