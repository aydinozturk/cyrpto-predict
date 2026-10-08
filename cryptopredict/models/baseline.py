"""Naive baselines. Every forecaster should beat these to be worth using."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from cryptopredict.core.types import LOG_RET_1


def check_xy(X: pd.DataFrame, y: pd.Series) -> None:
    """Shared fit-time input checks."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}")
    if len(y) == 0:
        raise ValueError("cannot fit on an empty training set")
    if pd.isna(np.asarray(y, dtype="float64")).any():
        raise ValueError("y contains NaN; drop rows without a target before fitting")


class BaselineForecaster(RegressorMixin, BaseEstimator):
    """Common plumbing: remembers training feature columns; sklearn ``clone`` works."""

    name = "baseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineForecaster":
        check_xy(X, y)
        self.feature_names_ = list(X.columns)
        self._fit(X, np.asarray(y, dtype="float64"))
        return self

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        pass


class ZeroReturn(BaselineForecaster):
    """Random walk: the next log return is predicted to be 0 (price unchanged)."""

    name = "zero"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype="float64")


class MeanReturn(BaselineForecaster):
    """Constant drift: the mean training target."""

    name = "mean"

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.mean_ = float(y.mean())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype="float64")


class LastReturn(BaselineForecaster):
    """Momentum: the next return equals the last observed one, ``X['log_ret_1']``.

    For a horizon ``h > 1`` this predicts the 1-bar return, not ``h`` times it.
    """

    name = "last"

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if LOG_RET_1 not in X.columns:
            raise ValueError(f"LastReturn needs feature column {LOG_RET_1!r}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[LOG_RET_1].to_numpy(dtype="float64")


class MovingAverageReturn(BaselineForecaster):
    """Recent drift.

    If ``column`` is given (e.g. a rolling-mean-return feature), the prediction is
    that column of ``X`` row by row. Otherwise the prediction is the constant mean
    of the last ``window`` training targets, i.e. the drift at the end of the
    training window.
    """

    name = "ma"

    def __init__(self, window: int = 24, column: str | None = None):
        self.window = window
        self.column = column

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.column is not None and self.column not in X.columns:
            raise ValueError(f"MovingAverageReturn column {self.column!r} not in X")
        self.mean_ = float(y[-self.window :].mean())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.column is not None:
            return X[self.column].to_numpy(dtype="float64")
        return np.full(len(X), self.mean_, dtype="float64")
