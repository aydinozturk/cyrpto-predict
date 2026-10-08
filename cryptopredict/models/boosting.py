"""LightGBM forecasters: a return regressor and a direction classifier.

Both tolerate NaN in X, are deterministic for a fixed ``random_state`` and
``n_jobs``, and pick the number of trees by early stopping on the last
``validation_fraction`` of the training window (chronological, no shuffle), so
no test data is ever touched. With ``refit=True`` the model is then refit on the
whole training window with that number of trees.

Requires the optional ``lightgbm`` dependency (``pip install cryptopredict[ml]``).
"""

from __future__ import annotations

import inspect
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from cryptopredict.models.baseline import BaselineForecaster

SEED = 42
# Early stopping is skipped if either part of the split would be smaller than this.
MIN_ES_ROWS = 30
# lightgbm >= 4.7 deprecates ``eval_set`` in favour of ``eval_X`` / ``eval_y``.
_EVAL_X = "eval_X" in inspect.signature(lgb.LGBMModel.fit).parameters

_TREE_PARAMS = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "n_jobs",
    "random_state",
)


class _LGBMForecaster(BaselineForecaster):
    """Shared plumbing: feature alignment, early stopping and refit."""

    def _estimator_class(self) -> type:
        raise NotImplementedError

    def _extra_params(self) -> dict[str, Any]:
        return {}

    def _make_estimator(self, n_estimators: int | None = None):
        params = {p: getattr(self, p) for p in _TREE_PARAMS}
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        return self._estimator_class()(
            subsample_freq=1 if self.subsample < 1.0 else 0,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            **params,
            **self._extra_params(),
        )

    def _values(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: X is missing training feature(s) {missing}")
        values = X[self.feature_names_].to_numpy(dtype="float64")
        return np.where(np.isinf(values), np.nan, values)

    def _fit_estimator(self, values: np.ndarray, target: np.ndarray):
        """Fit with early stopping on the chronological tail; sets ``best_iteration_``."""
        n_val = int(len(target) * self.validation_fraction)
        n_fit = len(target) - n_val
        use_es = (
            self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
            and min(n_val, n_fit) >= MIN_ES_ROWS
            and self._can_validate(target[:n_fit], target[n_fit:])
        )
        if not use_es:
            self.best_iteration_ = int(self.n_estimators)
            return self._make_estimator().fit(values, target)

        val_x, val_y = values[n_fit:], target[n_fit:]
        eval_kw = {"eval_X": (val_x,), "eval_y": (val_y,)} if _EVAL_X else {"eval_set": [(val_x, val_y)]}
        model = self._make_estimator().fit(
            values[:n_fit],
            target[:n_fit],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            **eval_kw,
        )
        self.best_iteration_ = int(model.best_iteration_ or self.n_estimators)
        if not self.refit:
            return model
        return self._make_estimator(n_estimators=self.best_iteration_).fit(values, target)

    def _can_validate(self, fit_target: np.ndarray, val_target: np.ndarray) -> bool:
        return True


class LGBMForecaster(_LGBMForecaster):
    """LightGBM regression of the h-step log return.

    ``objective`` is ``"l2"`` or ``"huber"``. Log returns are small numbers, so
    LightGBM's default Huber delta (0.9) would make Huber identical to L2; with
    ``huber_delta=None`` it is set to 1.345 robust standard deviations (MAD) of
    the training target instead.
    """

    name = "lgbm"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        max_depth: int = -1,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        objective: str = "l2",
        huber_delta: float | None = None,
        early_stopping_rounds: int | None = 50,
        validation_fraction: float = 0.15,
        refit: bool = True,
        n_jobs: int = 1,
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.huber_delta = huber_delta
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _estimator_class(self) -> type:
        return lgb.LGBMRegressor

    def _extra_params(self) -> dict[str, Any]:
        if self.objective == "l2":
            return {"objective": "regression"}
        if self.objective == "huber":
            return {"objective": "huber", "alpha": self.huber_delta_}
        raise ValueError(f"objective must be 'l2' or 'huber', got {self.objective!r}")

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if self.huber_delta is not None:
            self.huber_delta_ = float(self.huber_delta)
        else:
            mad = float(np.median(np.abs(y - np.median(y)))) * 1.4826
            self.huber_delta_ = 1.345 * mad if mad > 0 else 1.0
        self.model_ = self._fit_estimator(self._values(X), y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(self._values(X)), dtype="float64").ravel()


class LGBMDirectionForecaster(_LGBMForecaster):
    """LightGBM classifier of ``y > 0``, turned into an expected log return.

    ``predict`` returns ``p * up_mean_ + (1 - p) * down_mean_`` where ``p`` is
    ``predict_proba(X)`` and the conditional means ``E[y | y > 0]`` and
    ``E[y | y <= 0]`` come from the training target (no model output involved,
    so they are not overfit). The prediction is positive exactly when
    ``p > breakeven_proba_``. With ``magnitude="symmetric"`` both sides use
    ``E[|y|]`` instead, so the breakeven is exactly 0.5.
    """

    name = "lgbm_cls"

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        max_depth: int = -1,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        magnitude: str = "conditional",
        early_stopping_rounds: int | None = 50,
        validation_fraction: float = 0.15,
        refit: bool = True,
        n_jobs: int = 1,
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.magnitude = magnitude
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _estimator_class(self) -> type:
        return lgb.LGBMClassifier

    def _extra_params(self) -> dict[str, Any]:
        return {"objective": "binary"}

    def _can_validate(self, fit_target: np.ndarray, val_target: np.ndarray) -> bool:
        return len(np.unique(fit_target)) == 2 and len(np.unique(val_target)) == 2

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if self.magnitude not in ("conditional", "symmetric"):
            raise ValueError(f"magnitude must be 'conditional' or 'symmetric', got {self.magnitude!r}")
        up = y > 0
        if self.magnitude == "symmetric":
            size = float(np.abs(y).mean())
            self.up_mean_, self.down_mean_ = size, -size
        else:
            self.up_mean_ = float(y[up].mean()) if up.any() else 0.0
            self.down_mean_ = float(y[~up].mean()) if (~up).any() else 0.0
        labels = up.astype("int64")
        if len(np.unique(labels)) < 2:
            # A single class cannot train a classifier: predict its frequency.
            self.model_ = None
            self.constant_proba_ = float(labels.mean())
            self.best_iteration_ = 0
            return
        self.model_ = self._fit_estimator(self._values(X), labels)

    @property
    def breakeven_proba_(self) -> float:
        """P(up) above which the expected return is positive."""
        spread = self.up_mean_ - self.down_mean_
        return -self.down_mean_ / spread if spread > 0 else 0.5

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(y > 0) for every row, as a 1-D array (not sklearn's 2-column layout)."""
        values = self._values(X)
        if self.model_ is None:
            return np.full(len(values), self.constant_proba_, dtype="float64")
        return np.asarray(self.model_.predict_proba(values)[:, 1], dtype="float64")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(X)
        return p * self.up_mean_ + (1.0 - p) * self.down_mean_
