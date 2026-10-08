"""Target creation and feature/target alignment.

Targets never span a candle gap: ``y[t]`` is only defined when bars ``t`` and
``t + h`` belong to the same contiguous segment (see :func:`segment_ids`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cryptopredict.core.types import DEFAULT_HORIZON, TARGET_COLUMN, validate_ohlcv

from .indicators import BarLike, FeatureConfigLike, build_features, segment_ids


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")


def forward_log_return(
    close: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    bar: BarLike = None,
) -> pd.Series:
    """Return ``log(close[t + horizon] / close[t])`` on ``close``'s index.

    With a ``DatetimeIndex``, values whose ``horizon`` bars span a candle gap
    (a step longer than ``bar``, inferred by default) are ``NaN``.
    """
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
    result = np.log(values.shift(-horizon) / values)
    if isinstance(values.index, pd.DatetimeIndex):
        segments = pd.Series(segment_ids(values.index, bar), index=values.index)
        result = result.where(segments.shift(-horizon) == segments)
    elif bar is not None:
        raise ValueError("bar requires close to have a DatetimeIndex")
    return result.rename(TARGET_COLUMN)


def make_target(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    bar: BarLike = None,
) -> pd.Series:
    """Create the forward log-return target defined by the shared contract."""
    ohlcv = validate_ohlcv(df)
    return forward_log_return(ohlcv["close"], horizon=horizon, bar=bar)


def make_dataset(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    config: FeatureConfigLike = None,
    bar: BarLike = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned, finite ``(X, y)`` ready for estimator input.

    Indicator warm-up rows (after the series start and after every candle gap)
    and rows without a known target (the final ``horizon`` rows and those whose
    target spans a gap) are removed.  No imputation is performed.
    """
    X_all = build_features(df, config=config, bar=bar)
    y_all = make_target(df, horizon=horizon, bar=bar)
    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid].copy()
    y = y_all.loc[valid].copy()
    if not X.index.equals(y.index):
        raise AssertionError("feature and target alignment failed")
    return X, y


def latest_features(
    df: pd.DataFrame,
    config: FeatureConfigLike = None,
    bar: BarLike = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the feature row of the final bar of ``df`` for live prediction.

    With ``columns`` (e.g. the features a saved model was trained on) only those
    are returned and need to be complete; other columns may still be warming up.

    Raises ``ValueError`` if that row is incomplete, e.g. because the final bar
    follows a candle gap too closely: an older row would silently describe a
    stale market state.
    """
    features = build_features(df, config=config, bar=bar)
    if columns is not None:
        columns = list(columns)
        missing = [c for c in columns if c not in features.columns]
        if missing:
            raise ValueError(f"data lacks feature column(s): {missing}")
        features = features[columns]
    if features.empty:
        raise ValueError("not enough history to build a complete feature row")
    latest = features.tail(1)
    if latest.isna().any(axis=None):
        segments = segment_ids(features.index, bar)
        if segments[-1] > 0:
            since_gap = int((segments == segments[-1]).sum())
            gap_end = features.index[len(features) - since_gap]
            raise ValueError(
                f"the final bar is only {since_gap} bar(s) after a candle gap ending at "
                f"{gap_end.isoformat()}; more history after the gap is needed to build "
                "a complete feature row"
            )
        raise ValueError("not enough history to build a complete feature row")
    return latest.copy()
