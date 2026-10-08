"""Walk-forward evaluation: refit a fresh model on every fold, score out of sample."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import regression_report
from .splits import walk_forward_splits


@dataclass
class WalkForwardResult:
    """Outcome of :func:`walk_forward_evaluate`.

    folds:       one row per fold: train/test bounds plus the fold's metrics.
    overall:     metrics over all out-of-sample predictions concatenated.
    predictions: index = test rows of ``X``; columns ``y_true``, ``y_pred``, ``fold``.
    """

    folds: pd.DataFrame
    overall: dict[str, float]
    predictions: pd.DataFrame


def _new_model(model_factory: Any):
    # An unfitted sklearn-style estimator is accepted too: clone it per fold.
    if hasattr(model_factory, "fit") and not callable(model_factory):
        from sklearn.base import clone

        return clone(model_factory)
    return model_factory()


def walk_forward_evaluate(
    model_factory: Callable[[], Any] | Any,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    *,
    n_splits: int = 5,
    train_size: int | None = None,
    test_size: int | None = None,
    expanding: bool = True,
    gap: int = 0,
    threshold: float = 0.0,
) -> WalkForwardResult:
    """Evaluate a forecaster with time-ordered walk-forward validation.

    ``model_factory`` is called once per fold and must return a new, unfitted
    ``Forecaster`` (``fit(X, y)`` / ``predict(X)``); each model only ever sees its
    fold's training rows. ``splits`` overrides the split parameters, which are
    otherwise passed to :func:`walk_forward_splits`. ``X`` and ``y`` must share
    the same index and contain no NaN (drop the trailing rows without a target
    beforehand).
    """
    if len(X) != len(y):
        raise ValueError(f"length mismatch: X={len(X)}, y={len(y)}")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    if y.isna().any() or X.isna().any().any():
        raise ValueError("X and y must not contain NaN")
    if splits is None:
        splits = walk_forward_splits(
            len(X), n_splits, train_size=train_size, test_size=test_size,
            expanding=expanding, gap=gap,
        )

    rows: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        if train_idx.size == 0 or test_idx.size == 0:
            raise ValueError(f"fold {fold}: empty train or test set")
        if train_idx.max() >= test_idx.min():
            raise ValueError(f"fold {fold}: training data must precede test data")

        model = _new_model(model_factory)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(X.iloc[test_idx]), dtype="float64").ravel()
        if pred.size != test_idx.size:
            raise ValueError(f"fold {fold}: predict returned {pred.size} values for {test_idx.size} rows")

        y_test = y.iloc[test_idx]
        rows.append({
            "fold": fold,
            "train_start": X.index[train_idx[0]],
            "train_end": X.index[train_idx[-1]],
            "test_start": X.index[test_idx[0]],
            "test_end": X.index[test_idx[-1]],
            "n_train": int(train_idx.size),
            **regression_report(y_test.to_numpy(), pred, threshold),
        })
        parts.append(pd.DataFrame(
            {"y_true": y_test.to_numpy(dtype="float64"), "y_pred": pred, "fold": fold},
            index=y_test.index,
        ))

    if not parts:
        raise ValueError("no folds to evaluate")
    predictions = pd.concat(parts)
    if predictions.index.has_duplicates:
        raise ValueError("test sets of different folds overlap")
    overall = regression_report(predictions["y_true"], predictions["y_pred"], threshold)
    return WalkForwardResult(
        folds=pd.DataFrame(rows).set_index("fold"),
        overall=overall,
        predictions=predictions,
    )
