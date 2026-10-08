"""Causal technical indicators used by the forecasting models.

Every value at timestamp ``t`` is computed from bars at or before ``t``.  The
module intentionally leaves warm-up rows as ``NaN``; dataset construction is
responsible for dropping rows that are not ready for model input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from cryptopredict.core.types import LOG_RET_1, validate_ohlcv


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Serializable configuration for :func:`build_features`."""

    return_lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    return_mean_windows: tuple[int, ...] = (24,)
    sma_windows: tuple[int, ...] = (6, 12, 24)
    ema_windows: tuple[int, ...] = (6, 12, 24)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    volatility_window: int = 24
    volume_zscore_window: int = 24
    add_time_features: bool = False

    def __post_init__(self) -> None:
        sequence_fields = (
            "return_lags",
            "return_mean_windows",
            "sma_windows",
            "ema_windows",
        )
        for field_name in sequence_fields:
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise ValueError(f"{field_name} must be a tuple of positive integers")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
            for value in values:
                _require_positive_int(field_name, value)
        if 1 not in self.return_lags:
            raise ValueError("return_lags must contain 1 so log_ret_1 is available")

        for field_name in (
            "rsi_window",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "bollinger_window",
            "volatility_window",
            "volume_zscore_window",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be smaller than macd_slow")
        if not math.isfinite(self.bollinger_std) or self.bollinger_std <= 0:
            raise ValueError("bollinger_std must be a positive finite number")
        if not isinstance(self.add_time_features, bool):
            raise ValueError("add_time_features must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation suitable for model metadata."""
        data = asdict(self)
        for key in (
            "return_lags",
            "return_mean_windows",
            "sma_windows",
            "ema_windows",
        ):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FeatureConfig":
        """Build a config from a full or partial mapping.

        Missing keys keep their defaults, which makes small CLI overrides easy.
        Unknown keys are rejected to catch misspellings early.
        """
        if not isinstance(values, Mapping):
            raise ValueError("feature config must be a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown feature config key(s): {unknown}")
        data = dict(values)
        for key in (
            "return_lags",
            "return_mean_windows",
            "sma_windows",
            "ema_windows",
        ):
            if key in data:
                try:
                    data[key] = tuple(data[key])
                except TypeError as exc:
                    raise ValueError(f"{key} must be an iterable of positive integers") from exc
        return cls(**data)


FeatureConfigLike = FeatureConfig | Mapping[str, Any] | None


def resolve_config(config: FeatureConfigLike = None) -> FeatureConfig:
    """Normalize ``None``, a config object, or a partial mapping."""
    if config is None:
        return FeatureConfig()
    if isinstance(config, FeatureConfig):
        return config
    if isinstance(config, Mapping):
        return FeatureConfig.from_dict(config)
    raise ValueError(
        "config must be None, FeatureConfig, or a mapping; "
        f"got {type(config).__name__}"
    )


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must contain/be a positive integer, got {value!r}")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.mask(denominator == 0.0)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    relative_strength = _safe_ratio(avg_gain, avg_loss)
    result = 100.0 - 100.0 / (1.0 + relative_strength)
    result = result.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    result = result.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return result


def build_features(
    df: pd.DataFrame,
    config: FeatureConfigLike = None,
) -> pd.DataFrame:
    """Build causal model features with the same index as ``df``.

    Warm-up rows contain ``NaN`` by design.  Use
    :func:`cryptopredict.features.make_dataset` to align and remove incomplete
    rows for training, or :func:`cryptopredict.features.latest_features` for a
    live prediction row.
    """
    cfg = resolve_config(config)
    ohlcv = validate_ohlcv(df)
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    if (close <= 0.0).any():
        raise ValueError("close prices must be strictly positive")

    features = pd.DataFrame(index=ohlcv.index)
    log_close = np.log(close)
    for lag in cfg.return_lags:
        features[f"log_ret_{lag}"] = log_close.diff(lag)
    # The shared model contract requires this exact column.
    if LOG_RET_1 not in features:
        raise AssertionError("FeatureConfig validation failed to preserve log_ret_1")

    one_bar_return = features[LOG_RET_1]
    for window in cfg.return_mean_windows:
        features[f"ret_mean_{window}"] = one_bar_return.rolling(
            window=window, min_periods=window
        ).mean()

    for window in cfg.sma_windows:
        average = close.rolling(window=window, min_periods=window).mean()
        features[f"close_sma_{window}_ratio"] = _safe_ratio(close, average) - 1.0

    for window in cfg.ema_windows:
        average = close.ewm(span=window, adjust=False, min_periods=window).mean()
        features[f"close_ema_{window}_ratio"] = _safe_ratio(close, average) - 1.0

    features[f"rsi_{cfg.rsi_window}"] = _rsi(close, cfg.rsi_window)

    fast = close.ewm(
        span=cfg.macd_fast, adjust=False, min_periods=cfg.macd_fast
    ).mean()
    slow = close.ewm(
        span=cfg.macd_slow, adjust=False, min_periods=cfg.macd_slow
    ).mean()
    macd = fast - slow
    features["macd"] = macd
    features["macd_signal"] = macd.ewm(
        span=cfg.macd_signal, adjust=False, min_periods=cfg.macd_signal
    ).mean()

    middle = close.rolling(
        window=cfg.bollinger_window, min_periods=cfg.bollinger_window
    ).mean()
    deviation = close.rolling(
        window=cfg.bollinger_window, min_periods=cfg.bollinger_window
    ).std(ddof=0)
    upper = middle + cfg.bollinger_std * deviation
    lower = middle - cfg.bollinger_std * deviation
    band_range = upper - lower
    features[f"bollinger_pct_b_{cfg.bollinger_window}"] = _safe_ratio(
        close - lower, band_range
    )
    features[f"bollinger_width_{cfg.bollinger_window}"] = _safe_ratio(
        band_range, middle
    )

    features[f"volatility_{cfg.volatility_window}"] = one_bar_return.rolling(
        window=cfg.volatility_window, min_periods=cfg.volatility_window
    ).std(ddof=0)

    volume_mean = volume.rolling(
        window=cfg.volume_zscore_window, min_periods=cfg.volume_zscore_window
    ).mean()
    volume_std = volume.rolling(
        window=cfg.volume_zscore_window, min_periods=cfg.volume_zscore_window
    ).std(ddof=0)
    features[f"volume_zscore_{cfg.volume_zscore_window}"] = _safe_ratio(
        volume - volume_mean, volume_std
    )

    if cfg.add_time_features:
        hour = ohlcv.index.hour.to_numpy(dtype="float64")
        day_of_week = ohlcv.index.dayofweek.to_numpy(dtype="float64")
        features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        features["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        features["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    # Degenerate flat-price/flat-volume windows can have zero denominators.
    # Keep those rows incomplete instead of leaking infinities into estimators.
    return features.replace([np.inf, -np.inf], np.nan).astype("float64")
