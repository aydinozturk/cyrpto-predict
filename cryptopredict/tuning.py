"""Leak-free hyperparameter search with walk-forward validation.

:func:`tune` scores candidate parameters by an inner walk-forward over the data
it is given and nothing else; pass it a training window only.
:func:`nested_walk_forward` is the honest estimate of a tuned model: every outer
fold is tuned on its own training rows and scored once on its test rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler

from cryptopredict.evaluation.metrics import regression_report
from cryptopredict.evaluation.splits import walk_forward_splits
from cryptopredict.evaluation.walkforward import walk_forward_evaluate
from cryptopredict.models.registry import get_model

LOWER_IS_BETTER = {"mae", "rmse", "smape", "relative_mae"}
HIGHER_IS_BETTER = {"directional_accuracy"}

_LGBM_SPACE = {
    "learning_rate": [0.01, 0.03, 0.1],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [20, 50, 100, 200],
    "reg_lambda": [0.0, 1.0, 10.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
}

# Search spaces used when neither ``param_grid`` nor ``param_distributions`` is given.
DEFAULT_SPACES: dict[str, dict[str, list]] = {
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "gbm": {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_iter": [100, 200, 400],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [20, 50, 100],
        "l2_regularization": [0.0, 1.0, 10.0],
    },
    "lgbm": _LGBM_SPACE,
    "lgbm_cls": _LGBM_SPACE,
}


@dataclass
class TuneResult:
    """Outcome of :func:`tune`.

    best_params: the winning candidate (``{}`` = the model's defaults).
    best_score:  its inner walk-forward ``metric``.
    trials:      one row per evaluated candidate: ``params``, ``score``, ``seconds``, ``error``.
    timed_out:   ``True`` if ``time_budget_sec`` cut the search short.
    """

    best_params: dict[str, Any]
    best_score: float
    trials: pd.DataFrame
    timed_out: bool = False


@dataclass
class NestedResult:
    """Outcome of :func:`nested_walk_forward`, shaped like ``WalkForwardResult``.

    ``folds`` additionally holds each fold's chosen ``params``, its ``inner_score``
    and the number of trials; ``params`` lists the chosen parameters per fold.
    """

    folds: pd.DataFrame
    overall: dict[str, float]
    predictions: pd.DataFrame
    params: list[dict[str, Any]] = field(default_factory=list)


def _sign(metric: str) -> float:
    if metric in LOWER_IS_BETTER:
        return 1.0
    if metric in HIGHER_IS_BETTER:
        return -1.0
    raise ValueError(f"unknown metric {metric!r}; use one of {sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)}")


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def candidates(
    model_name: str,
    param_grid: dict | list[dict] | None = None,
    param_distributions: dict | None = None,
    n_trials: int | None = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Candidate parameter sets; the model's defaults (``{}``) always come first.

    ``param_grid`` is searched exhaustively in order (capped at ``n_trials``).
    ``param_distributions`` (lists and/or scipy distributions; default: the
    model's ``DEFAULT_SPACES`` entry) is sampled with ``seed``; when it only holds
    lists, distinct grid points are drawn without replacement.
    """
    if param_grid is not None and param_distributions is not None:
        raise ValueError("pass either param_grid or param_distributions, not both")
    if n_trials is not None and n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    limit = None if n_trials is None else n_trials - 1

    if param_grid is not None:
        found = list(ParameterGrid(param_grid))
    else:
        space = param_distributions if param_distributions is not None else DEFAULT_SPACES.get(model_name, {})
        if not space:
            found = []
        elif all(isinstance(v, (list, tuple)) for v in space.values()):
            grid = list(ParameterGrid(space))
            order = np.random.default_rng(seed).permutation(len(grid))
            found = [grid[i] for i in order]
        else:
            found = list(ParameterSampler(space, n_iter=20 if limit is None else max(limit, 1), random_state=seed))

    out: list[dict[str, Any]] = [{}]
    for params in found:
        params = {k: _plain(v) for k, v in params.items()}
        if params and params not in out:
            out.append(params)
    return out if n_trials is None else out[:n_trials]


def tune(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict | list[dict] | None = None,
    param_distributions: dict | None = None,
    n_trials: int | None = 20,
    inner_splits: int = 3,
    gap: int = 0,
    metric: str = "mae",
    seed: int = 42,
    time_budget_sec: float | None = None,
    base_params: dict[str, Any] | None = None,
) -> TuneResult:
    """Pick the parameters of ``model_name`` with the best inner walk-forward ``metric``.

    Only ``X`` / ``y`` are used, so pass a training window. ``base_params`` are
    fixed and merged under every candidate. The first candidate (the defaults)
    always runs; after that, the search stops as soon as ``time_budget_sec`` is
    used up and returns the best result so far. A candidate that raises is
    recorded with its error and a score of ``inf``.
    """
    sign = _sign(metric)
    base = dict(base_params or {})
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    timed_out = False
    for i, params in enumerate(candidates(model_name, param_grid, param_distributions, n_trials, seed)):
        if i > 0 and time_budget_sec is not None and time.monotonic() - start >= time_budget_sec:
            timed_out = True
            break
        t0 = time.monotonic()
        kwargs = {**base, **params}
        try:
            res = walk_forward_evaluate(
                lambda: get_model(model_name, **kwargs), X, y, n_splits=inner_splits, gap=gap
            )
            score, error = float(res.overall[metric]), None
        except Exception as exc:  # noqa: BLE001 - a bad candidate must not end the search
            score, error = float("nan"), f"{type(exc).__name__}: {exc}"
        rows.append({"params": params, "score": score, "seconds": time.monotonic() - t0, "error": error})

    trials = pd.DataFrame(rows)
    losses = np.array([sign * s if np.isfinite(s) else np.inf for s in trials["score"]])
    if not np.isfinite(losses).any():
        raise ValueError(f"every candidate failed for {model_name!r}: {trials['error'].iloc[0]}")
    best = int(np.argmin(losses))  # first best wins ties, i.e. the defaults
    return TuneResult(
        best_params={**base, **trials["params"].iloc[best]},
        best_score=float(trials["score"].iloc[best]),
        trials=trials,
        timed_out=timed_out,
    )


def nested_walk_forward(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    *,
    param_grid: dict | list[dict] | None = None,
    param_distributions: dict | None = None,
    n_trials: int | None = 20,
    metric: str = "mae",
    seed: int = 42,
    time_budget_sec: float | None = None,
    base_params: dict[str, Any] | None = None,
    gap: int = 0,
    train_size: int | None = None,
    test_size: int | None = None,
    expanding: bool = True,
    threshold: float = 0.0,
) -> NestedResult:
    """Outer walk-forward where each fold is tuned with :func:`tune` on its training rows only.

    ``time_budget_sec`` applies to each fold's search. ``gap`` is used for both
    the outer and the inner splits.
    """
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    splits = walk_forward_splits(
        len(X), outer_splits, train_size=train_size, test_size=test_size, expanding=expanding, gap=gap
    )
    rows: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    chosen: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        result = tune(
            model_name, X_train, y_train,
            param_grid=param_grid, param_distributions=param_distributions, n_trials=n_trials,
            inner_splits=inner_splits, gap=gap, metric=metric, seed=seed,
            time_budget_sec=time_budget_sec, base_params=base_params,
        )
        model = get_model(model_name, **result.best_params).fit(X_train, y_train)
        pred = np.asarray(model.predict(X.iloc[test_idx]), dtype="float64").ravel()
        y_test = y.iloc[test_idx]
        chosen.append(result.best_params)
        rows.append({
            "fold": fold,
            "train_start": X.index[train_idx[0]],
            "train_end": X.index[train_idx[-1]],
            "test_start": X.index[test_idx[0]],
            "test_end": X.index[test_idx[-1]],
            "n_train": int(train_idx.size),
            "params": result.best_params,
            "inner_score": result.best_score,
            "n_trials": len(result.trials),
            "timed_out": result.timed_out,
            **regression_report(y_test.to_numpy(), pred, threshold),
        })
        parts.append(pd.DataFrame(
            {"y_true": y_test.to_numpy(dtype="float64"), "y_pred": pred, "fold": fold}, index=y_test.index
        ))

    predictions = pd.concat(parts)
    return NestedResult(
        folds=pd.DataFrame(rows).set_index("fold"),
        overall=regression_report(predictions["y_true"], predictions["y_pred"], threshold),
        predictions=predictions,
        params=chosen,
    )
