"""Model combinations: a weighted average and a leak-free stacked ensemble.

Members are given by registry name, optionally with parameters:
``members=("ridge", ("gbm", {"max_iter": 100}), "lgbm")``. A member whose
optional dependency is not installed (e.g. ``lgbm`` without lightgbm) is
skipped with a warning.
"""

from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cryptopredict.evaluation.splits import walk_forward_splits
from cryptopredict.models.baseline import BaselineForecaster

MemberSpec = Union[str, tuple[str, dict[str, Any]]]
DEFAULT_MEMBERS: tuple[str, ...] = ("ridge", "gbm", "lgbm")


def _member_name(spec: MemberSpec) -> str:
    return spec if isinstance(spec, str) else spec[0]


def build_members(members: tuple[MemberSpec, ...]) -> list[tuple[int, str, Any]]:
    """``(position, name, unfitted model)`` for every member that can be built."""
    from cryptopredict.models.registry import get_model

    built = []
    for pos, spec in enumerate(members):
        name, params = (spec, {}) if isinstance(spec, str) else (spec[0], dict(spec[1]))
        try:
            built.append((pos, name, get_model(name, **params)))
        except (ValueError, ImportError) as exc:
            warnings.warn(f"ensemble member {name!r} skipped: {exc}", stacklevel=3)
    if not built:
        raise ValueError(f"no usable ensemble member among {[_member_name(m) for m in members]}")
    return built


class _MemberForecaster(BaselineForecaster):
    def _check_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: X is missing training feature(s) {missing}")
        return X[self.feature_names_]

    def _member_predictions(self, X: pd.DataFrame) -> np.ndarray:
        X = self._check_columns(X)
        cols = [np.asarray(m.predict(X), dtype="float64").ravel() for m in self.models_]
        return np.column_stack(cols)


class EnsembleForecaster(_MemberForecaster):
    """Weighted average of member forecasts.

    ``weights`` (one per entry of ``members``, default equal) are renormalized
    over the members that could actually be built.
    """

    name = "ensemble"

    def __init__(self, members: tuple[MemberSpec, ...] = DEFAULT_MEMBERS, weights: tuple[float, ...] | None = None):
        self.members = members
        self.weights = weights

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if self.weights is not None and len(self.weights) != len(self.members):
            raise ValueError(f"got {len(self.weights)} weights for {len(self.members)} members")
        built = build_members(self.members)
        w = np.ones(len(built)) if self.weights is None else np.array([self.weights[p] for p, _, _ in built], float)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be non-negative with a positive sum, got {w.tolist()}")
        self.member_names_ = [name for _, name, _ in built]
        self.weights_ = w / w.sum()
        self.models_ = [model.fit(X, y) for _, _, model in built]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._member_predictions(X) @ self.weights_


class StackingForecaster(_MemberForecaster):
    """Ridge meta-model with non-negative weights over member forecasts.

    The meta-model is trained on chronological out-of-fold member predictions
    inside the training window (``walk_forward_splits(n_splits=inner_splits,
    gap=gap)``), so it never sees a forecast made with future data; the members
    are then refit on the whole training window. Member forecasts are divided by
    their RMS before the meta fit so that ``alpha`` does not depend on the tiny
    scale of log returns. No intercept by default: a constant drift term usually
    hurts against the zero-return baseline.
    """

    name = "stack"

    def __init__(
        self,
        members: tuple[MemberSpec, ...] = DEFAULT_MEMBERS,
        inner_splits: int = 3,
        gap: int = 0,
        alpha: float = 1.0,
        positive: bool = True,
        fit_intercept: bool = False,
    ):
        self.members = members
        self.inner_splits = inner_splits
        self.gap = gap
        self.alpha = alpha
        self.positive = positive
        self.fit_intercept = fit_intercept

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        built = build_members(self.members)
        self.member_names_ = [name for _, name, _ in built]
        specs = [self.members[p] for p, _, _ in built]

        oof_rows, oof_preds = [], []
        for train_idx, test_idx in walk_forward_splits(len(X), self.inner_splits, gap=self.gap):
            fold_models = [m for _, _, m in build_members(tuple(specs))]
            cols = [
                np.asarray(m.fit(X.iloc[train_idx], y[train_idx]).predict(X.iloc[test_idx]), dtype="float64").ravel()
                for m in fold_models
            ]
            oof_rows.append(test_idx)
            oof_preds.append(np.column_stack(cols))
        rows = np.concatenate(oof_rows)
        Z = np.vstack(oof_preds)
        self.n_oof_ = int(rows.size)

        scale = np.sqrt(np.mean(Z**2, axis=0))
        scale[scale == 0] = 1.0
        meta = Ridge(alpha=self.alpha, positive=self.positive, fit_intercept=self.fit_intercept)
        meta.fit(Z / scale, y[rows])
        self.coef_ = np.asarray(meta.coef_, dtype="float64") / scale
        self.intercept_ = float(meta.intercept_) if self.fit_intercept else 0.0

        self.models_ = [model.fit(X, y) for _, _, model in built]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._member_predictions(X) @ self.coef_ + self.intercept_
