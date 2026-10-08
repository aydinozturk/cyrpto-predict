"""Shared contract every subpackage relies on.

OHLCV frame
    index: ``pd.DatetimeIndex`` named ``"open_time"``, tz=UTC, strictly increasing
    (sorted, no duplicates); columns ``OHLCV_COLUMNS`` as float64, no NaN.

Target
    ``y[t] = log(close[t + h] / close[t])`` with default ``h = DEFAULT_HORIZON``,
    stored in column ``TARGET_COLUMN``. Features may only use information
    available at bar ``t`` or earlier. ``"log_ret_1"`` (``log(close[t] / close[t-1])``)
    is always present in the feature set.

Forecaster
    ``fit(X, y) -> self`` and ``predict(X) -> 1-D np.ndarray`` of length ``len(X)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

INDEX_NAME = "open_time"
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "target"
LOG_RET_1 = "log_ret_1"
DEFAULT_HORIZON = 1


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Check ``df`` against the OHLCV contract and return a normalized copy.

    The returned frame contains exactly ``OHLCV_COLUMNS`` (in that order, extra
    columns dropped) cast to float64. Raises ``ValueError`` on any violation.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"expected a pandas DataFrame, got {type(df).__name__}")

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"index must be a DatetimeIndex, got {type(idx).__name__}")
    if idx.name != INDEX_NAME:
        raise ValueError(f"index must be named {INDEX_NAME!r}, got {idx.name!r}")
    if idx.tz is None or str(idx.tz) != "UTC":
        raise ValueError(f"index must be tz-aware UTC, got tz={idx.tz!r}")
    if idx.hasnans:
        raise ValueError("index contains NaT values")
    if idx.has_duplicates:
        dupes = idx[idx.duplicated()].unique()
        raise ValueError(f"index has {len(dupes)} duplicated timestamp(s), e.g. {dupes[0]}")
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be sorted in increasing order")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV column(s): {missing}")

    out = df[OHLCV_COLUMNS].copy()
    for col in OHLCV_COLUMNS:
        if not pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            raise ValueError(f"column {col!r} must be numeric, got dtype {out[col].dtype}")
    out = out.astype("float64")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"NaN values in column(s): {bad}")
    return out


@runtime_checkable
class Forecaster(Protocol):
    """Regression model predicting the h-step-ahead log return."""

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
