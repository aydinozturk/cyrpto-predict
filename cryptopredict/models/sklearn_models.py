"""Learned forecasters wrapping scikit-learn pipelines.

Scaling lives inside the pipeline, so in walk-forward evaluation it is fit on the
training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cryptopredict.models.baseline import BaselineForecaster

SEED = 42


class _PipelineForecaster(BaselineForecaster):
    def _make_pipeline(self) -> Pipeline:
        raise NotImplementedError

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.pipeline_ = self._make_pipeline().fit(X.to_numpy(dtype="float64"), y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"{self.name}: X is missing training feature(s) {missing}")
        values = X[self.feature_names_].to_numpy(dtype="float64")
        return np.asarray(self.pipeline_.predict(values), dtype="float64").ravel()


class RidgeForecaster(_PipelineForecaster):
    """StandardScaler + Ridge regression. X must not contain NaN."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _make_pipeline(self) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])


class GBMForecaster(_PipelineForecaster):
    """Histogram gradient boosting (deterministic, tolerates NaN in X)."""

    name = "gbm"

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = SEED,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def _make_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "gbm",
                    HistGradientBoostingRegressor(
                        max_iter=self.max_iter,
                        learning_rate=self.learning_rate,
                        max_leaf_nodes=self.max_leaf_nodes,
                        min_samples_leaf=self.min_samples_leaf,
                        l2_regularization=self.l2_regularization,
                        early_stopping=False,
                        random_state=self.random_state,
                    ),
                )
            ]
        )
