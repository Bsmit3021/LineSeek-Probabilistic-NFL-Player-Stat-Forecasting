"""Phase 4C.1b threshold-conditioned CDF (docs/adr/0008): P(Y <= k | X, k, participated).

One LightGBM binary classifier per market and fold replaces Phase 4C.1's one classifier per
threshold. Each training row is repeated once per frozen grid threshold k with one extra
numeric input, `threshold` = k, and target 1{y <= k}. Every other parameter equals the Phase
4C.1 / Phase 4C zero-model configuration (Phase 4B LightGBM, objective "binary"). A monotone
constraint of +1 on `threshold` (all 42 Phase 3 features unconstrained) makes every tree, hence
the raw score and the predicted probability, non-decreasing in k for fixed X. There is no
post-hoc monotone repair: a decrease larger than floating-point noise (FP_TOL) is an error.
"""

import lightgbm as lgb
import numpy as np

from lineseek.models.cdf import CDF_PARAMS, CDFError
from lineseek.models.learned import FEATURES

TC_CDF_SET_VERSION = "2.0.0"
TC_FEATURES: tuple[str, ...] = (*FEATURES, "threshold")  # the only new input is the last one
TC_PARAMS = {**CDF_PARAMS, "monotone_constraints": [0] * len(FEATURES) + [1],
             "monotone_constraints_method": "basic"}
FP_TOL = 1e-12  # largest decrease across thresholds accepted as floating-point noise


def stack(x: np.ndarray, thresholds: list[int]) -> np.ndarray:
    """(n * T, F + 1): each row of x repeated for every threshold (row-major: row i holds
    thresholds in ascending order at i*T .. i*T + T - 1), threshold appended as the last column."""
    t = np.asarray(thresholds, dtype=np.float64)
    out = np.empty((len(x) * len(t), x.shape[1] + 1))
    out[:, :-1] = np.repeat(x, len(t), axis=0)
    out[:, -1] = np.tile(t, len(x))
    return out


def fit_predict_tc(thresholds: list[int], x_train: np.ndarray, y_train: np.ndarray,
                   x_test: np.ndarray) -> np.ndarray:
    """(n_test, T) P(Y <= k) from one threshold-conditioned, monotone-constrained classifier."""
    if len(x_train) != len(y_train) or np.isnan(y_train).any():
        raise CDFError("training targets must be complete and aligned")
    if x_train.shape[1] != len(FEATURES) or list(thresholds) != sorted(set(thresholds)):
        raise CDFError("inputs must be the frozen features and an ascending threshold grid")
    t = np.asarray(thresholds, dtype=np.float64)
    z = (y_train[:, None] <= t[None, :]).reshape(-1).astype(int)
    if z.min() == z.max():  # one class only: the probability is that class
        return np.full((len(x_test), len(t)), float(z[0]))
    model = lgb.LGBMClassifier(**TC_PARAMS).fit(stack(x_train, thresholds), z)
    p = model.predict_proba(stack(x_test, thresholds))[:, list(model.classes_).index(1)]
    return p.reshape(len(x_test), len(t))


def check_monotone(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Record decreases across ascending thresholds. Decreases above FP_TOL are an error (the
    constraint failed); smaller ones are floating-point noise, removed by a running maximum."""
    d = np.diff(raw, axis=1)
    worst = float(-d.min()) if d.size else 0.0
    if worst > FP_TOL:
        raise CDFError(f"monotone constraint violated by {worst:.3g} (> {FP_TOL})")
    final = np.maximum.accumulate(raw, axis=1)
    return {"final": final, "violations": (d < 0).sum(axis=1), "changed": final != raw,
            "max_decrease": max(worst, 0.0)}
