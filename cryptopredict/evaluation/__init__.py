"""Evaluation: walk-forward validation, metrics and backtesting."""

from .backtest import BacktestResult, backtest
from .metrics import (
    directional_accuracy,
    mae,
    regression_report,
    relative_mae,
    rmse,
    smape,
)
from .splits import walk_forward_splits
from .stats import (
    DieboldMarianoResult,
    diebold_mariano,
    diebold_mariano_test,
    dm_vs_zero,
    long_run_variance,
)
from .walkforward import WalkForwardResult, walk_forward_evaluate

__all__ = [
    "BacktestResult",
    "DieboldMarianoResult",
    "WalkForwardResult",
    "backtest",
    "directional_accuracy",
    "diebold_mariano",
    "diebold_mariano_test",
    "dm_vs_zero",
    "long_run_variance",
    "mae",
    "regression_report",
    "relative_mae",
    "rmse",
    "smape",
    "walk_forward_evaluate",
    "walk_forward_splits",
]
