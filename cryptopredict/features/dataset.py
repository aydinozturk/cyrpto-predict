"""Target creation and feature/target alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptopredict.core.types import DEFAULT_HORIZON, TARGET_COLUMN, validate_ohlcv

from .indicators import FeatureConfigLike, build_features


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")


def forward_log_return(
    close: pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> pd.Series:
    """Return ``log(close[t + horizon] / close[t])`` on ``close``'s index."""
    _validate_horizon(horizon)
    if not isinstance(close, pd.Series):
        raise ValueError(f"close must be a pandas Series, got {type(close).__name__}")
    if not pd.api.types.is_numeric_dtype(close) or pd.api.types.is_bool_dtype(close):
        raise ValueError("close must be numeric")
    values = close.astype("float64")
    if values.isna().any():
        raise ValueError("close must not contain NaN")
    if (values <= 0.0).any():
        raise ValueError("close prices must be strictly positive")
    return np.log(values.shift(-horizon) / values).rename(TARGET_COLUMN)


def make_target(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.Series:
    """Create the forward log-return target defined by the shared contract."""
    ohlcv = validate_ohlcv(df)
    return forward_log_return(ohlcv["close"], horizon=horizon)


def make_dataset(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    config: FeatureConfigLike = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned, finite ``(X, y)`` ready for estimator input.

    Both indicator warm-up rows and the final ``horizon`` rows without a known
    target are removed.  No imputation is performed.
    """
    X_all = build_features(df, config=config)
    y_all = make_target(df, horizon=horizon)
    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid].copy()
    y = y_all.loc[valid].copy()
    if not X.index.equals(y.index):
        raise AssertionError("feature and target alignment failed")
    return X, y


def latest_features(
    df: pd.DataFrame,
    config: FeatureConfigLike = None,
) -> pd.DataFrame:
    """Return the most recent complete feature row for live prediction."""
    complete = build_features(df, config=config).dropna(axis=0, how="any")
    if complete.empty:
        raise ValueError("not enough history to build a complete feature row")
    return complete.tail(1).copy()
