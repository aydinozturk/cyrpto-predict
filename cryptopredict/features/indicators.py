"""Causal technical indicators used by the forecasting models.

Every value at timestamp ``t`` is computed from bars at or before ``t``.  The
module intentionally leaves warm-up rows as ``NaN``; dataset construction is
responsible for dropping rows that are not ready for model input.

Missing candles (exchange outages) split the series into contiguous segments.
Indicators are computed per segment, so no window or EMA state spans a gap and
the rows right after a gap get the same warm-up ``NaN`` as the series start.

Higher-timeframe (HTF) features resample each segment into longer candles
(e.g. 4h, 1d aligned to UTC midnight) and only use *complete, closed* ones: bar
``t`` sees the latest HTF candle whose close (``open_time + htf``) is at or
before its own close (``t + bar``).  A partially formed HTF candle never leaks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
import re
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
    atr_window: int = 14
    stoch_window: int = 14
    stoch_smooth: int = 3
    obv_window: int = 24
    range_vol_window: int = 24
    add_candle_features: bool = True
    moment_window: int = 72
    # Pandas offset aliases that divide a UTC day, e.g. "4h", "1d".  An HTF is
    # skipped unless ``bar < htf <= htf_max_ratio * bar``; the ratio caps warm-up.
    htf_intervals: tuple[str, ...] = ("4h", "1d")
    htf_window: int = 14
    htf_max_ratio: int = 24

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
            "atr_window",
            "stoch_window",
            "stoch_smooth",
            "obv_window",
            "range_vol_window",
            "moment_window",
            "htf_window",
            "htf_max_ratio",
        ):
            _require_positive_int(field_name, getattr(self, field_name))
        if self.moment_window < 4:
            raise ValueError("moment_window must be at least 4 (kurtosis needs 4 values)")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be smaller than macd_slow")
        if not math.isfinite(self.bollinger_std) or self.bollinger_std <= 0:
            raise ValueError("bollinger_std must be a positive finite number")
        for field_name in ("add_time_features", "add_candle_features"):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        if not isinstance(self.htf_intervals, tuple):
            raise ValueError("htf_intervals must be a tuple of interval strings")
        lengths = [_htf_length(value) for value in self.htf_intervals]
        if len(lengths) != len(set(lengths)):
            raise ValueError("htf_intervals must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation suitable for model metadata."""
        data = asdict(self)
        for key in (
            "return_lags",
            "return_mean_windows",
            "sma_windows",
            "ema_windows",
            "htf_intervals",
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
            "htf_intervals",
        ):
            if key in data:
                if isinstance(data[key], str):
                    raise ValueError(f"{key} must be a list, got the string {data[key]!r}")
                try:
                    data[key] = tuple(data[key])
                except TypeError as exc:
                    raise ValueError(f"{key} must be an iterable") from exc
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


_DAY = pd.Timedelta(days=1)
# Binance-style interval names: minutes, hours, days.
_HTF_PATTERN = re.compile(r"(\d+)([mhd])")
_HTF_UNITS = {"m": pd.Timedelta(minutes=1), "h": pd.Timedelta(hours=1), "d": _DAY}


def _htf_length(value: Any) -> pd.Timedelta:
    """Validate one ``htf_intervals`` entry and return its length."""
    if not isinstance(value, str):
        raise ValueError(f"htf_intervals must contain interval strings, got {value!r}")
    match = _HTF_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"htf_intervals entry {value!r} is not an interval like '4h' or '1d'")
    length = int(match[1]) * _HTF_UNITS[match[2]]
    if length <= pd.Timedelta(0) or _DAY % length != pd.Timedelta(0):
        # Flooring to multiples of a day divisor keeps candles aligned to UTC midnight.
        raise ValueError(f"htf_intervals entry {value!r} must evenly divide one day")
    return length


BarLike = pd.Timedelta | str | None


def infer_bar(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Bar length of ``index``: its most common step (the smallest one on ties).

    Returns ``None`` when there are fewer than two timestamps.
    """
    if len(index) < 2:
        return None
    counts = pd.Series(index[1:] - index[:-1]).value_counts()
    return counts[counts == counts.max()].index.min()


def _resolve_bar(index: pd.DatetimeIndex, bar: BarLike) -> pd.Timedelta | None:
    if bar is None:
        return infer_bar(index)
    try:
        value = pd.Timedelta(bar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bar must be a positive Timedelta, got {bar!r}") from exc
    if pd.isna(value) or value <= pd.Timedelta(0):
        raise ValueError(f"bar must be a positive Timedelta, got {bar!r}")
    return value


def segment_ids(index: pd.DatetimeIndex, bar: BarLike = None) -> np.ndarray:
    """Number the contiguous runs of ``index``: a step longer than ``bar`` starts a new one.

    ``bar`` defaults to :func:`infer_bar`. Gap-free data is a single segment (all 0).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"index must be a DatetimeIndex, got {type(index).__name__}")
    step = _resolve_bar(index, bar)
    if step is None:
        return np.zeros(len(index), dtype="int64")
    breaks = np.asarray(index[1:] - index[:-1] > step)
    return np.concatenate([[0], np.cumsum(breaks)]).astype("int64")


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
    bar: BarLike = None,
) -> pd.DataFrame:
    """Build causal model features with the same index as ``df``.

    Warm-up rows contain ``NaN`` by design, at the start of the series and after
    every candle gap (a step longer than ``bar``, inferred from the index by
    default).  Use :func:`cryptopredict.features.make_dataset` to align and
    remove incomplete rows for training, or
    :func:`cryptopredict.features.latest_features` for a live prediction row.
    """
    cfg = resolve_config(config)
    ohlcv = validate_ohlcv(df)
    if (ohlcv["close"] <= 0.0).any():
        raise ValueError("close prices must be strictly positive")

    step = _resolve_bar(ohlcv.index, bar)
    segments = segment_ids(ohlcv.index, step)
    if len(segments) == 0 or segments[-1] == 0:
        return _segment_features(ohlcv, cfg, step)
    parts = [
        _segment_features(part, cfg, step) for _, part in ohlcv.groupby(segments, sort=False)
    ]
    return pd.concat(parts)


def active_htf_intervals(config: FeatureConfigLike, bar: BarLike) -> list[tuple[str, pd.Timedelta]]:
    """``(name, length)`` of the HTFs that :func:`build_features` uses for bars of length ``bar``.

    An HTF is used when it is a whole multiple of ``bar``, longer than it and at
    most ``htf_max_ratio`` bars long.  Without a known bar no HTF is used.
    """
    cfg = resolve_config(config)
    if bar is None:
        return []
    step = _resolve_bar(pd.DatetimeIndex([]), bar)
    active = []
    for name in cfg.htf_intervals:
        length = _htf_length(name)
        if length > step and length % step == pd.Timedelta(0) and length <= cfg.htf_max_ratio * step:
            active.append((name, length))
    return active


def required_history(config: FeatureConfigLike = None, bar: BarLike = "1h") -> int:
    """Upper bound on the gap-free bars needed before the final feature row is complete.

    Use it to size the history fetched for a live prediction.
    """
    cfg = resolve_config(config)
    needed = [
        max(cfg.return_lags) + 1,
        max(cfg.return_mean_windows, default=0) + 1,
        max(cfg.sma_windows + cfg.ema_windows, default=0),
        cfg.rsi_window + 1,
        cfg.macd_slow + cfg.macd_signal - 1,
        cfg.bollinger_window,
        cfg.volatility_window + 1,
        cfg.volume_zscore_window,
        cfg.atr_window + 1,
        cfg.stoch_window + cfg.stoch_smooth - 1,
        cfg.obv_window,
        cfg.range_vol_window,
        cfg.moment_window + 1,
    ]
    htf_candles = max(4, cfg.htf_window + 1)
    step = _resolve_bar(pd.DatetimeIndex([]), bar)
    for _, length in active_htf_intervals(cfg, step):
        # A partly covered first candle is unusable, hence one candle extra.
        needed.append((htf_candles + 1) * int(length / step))
    return max(needed)


def _segment_features(
    ohlcv: pd.DataFrame, cfg: FeatureConfig, bar: pd.Timedelta | None = None
) -> pd.DataFrame:
    """Features of a validated, gap-free OHLCV frame with bars of length ``bar``."""
    if bar is None:
        bar = infer_bar(ohlcv.index)
    close = ohlcv["close"]
    volume = ohlcv["volume"]
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
    # A flat window has a zero-width band: the close sits in its middle.
    features[f"bollinger_pct_b_{cfg.bollinger_window}"] = _safe_ratio(
        close - lower, band_range
    ).mask(band_range == 0.0, 0.5)
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
    # Constant volume deviates by zero standard deviations.
    features[f"volume_zscore_{cfg.volume_zscore_window}"] = _safe_ratio(
        volume - volume_mean, volume_std
    ).mask(volume_std == 0.0, 0.0)

    if cfg.add_time_features:
        hour = ohlcv.index.hour.to_numpy(dtype="float64")
        day_of_week = ohlcv.index.dayofweek.to_numpy(dtype="float64")
        features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        features["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        features["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    _add_range_features(features, ohlcv, cfg)
    for name, length in active_htf_intervals(cfg, bar):
        _add_htf_features(features, ohlcv, cfg, bar, name, length)

    # Degenerate flat-price/flat-volume windows can have zero denominators.
    # Keep those rows incomplete instead of leaking infinities into estimators.
    return features.replace([np.inf, -np.inf], np.nan).astype("float64")


def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window=window, min_periods=window).mean()
    std = values.rolling(window=window, min_periods=window).std(ddof=0)
    # A constant window deviates by zero standard deviations.
    return _safe_ratio(values - mean, std).mask(std == 0.0, 0.0)


def _add_range_features(features: pd.DataFrame, ohlcv: pd.DataFrame, cfg: FeatureConfig) -> None:
    """ATR, stochastic, OBV, range volatility, candle shape and return moments."""
    open_, high, low, close, volume = (ohlcv[c] for c in ("open", "high", "low", "close", "volume"))
    previous_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    atr = true_range.ewm(alpha=1.0 / cfg.atr_window, adjust=False, min_periods=cfg.atr_window).mean()
    features[f"atr_{cfg.atr_window}"] = _safe_ratio(atr, close)

    lowest = low.rolling(window=cfg.stoch_window, min_periods=cfg.stoch_window).min()
    highest = high.rolling(window=cfg.stoch_window, min_periods=cfg.stoch_window).max()
    span = highest - lowest
    # A flat window puts the close in the middle of its (empty) range, like RSI's 50.
    stoch_k = (100.0 * _safe_ratio(close - lowest, span)).mask(span == 0.0, 50.0)
    features[f"stoch_k_{cfg.stoch_window}"] = stoch_k
    features[f"stoch_d_{cfg.stoch_window}"] = stoch_k.rolling(
        window=cfg.stoch_smooth, min_periods=cfg.stoch_smooth
    ).mean()

    # The first bar has no previous close: it counts as unchanged.
    signed_volume = np.sign(close.diff()).fillna(0.0) * volume
    obv = signed_volume.cumsum()
    features[f"obv_zscore_{cfg.obv_window}"] = _rolling_zscore(obv, cfg.obv_window)
    signed_sum = signed_volume.rolling(window=cfg.obv_window, min_periods=cfg.obv_window).sum()
    volume_sum = volume.rolling(window=cfg.obv_window, min_periods=cfg.obv_window).sum()
    features[f"signed_volume_ratio_{cfg.obv_window}"] = _safe_ratio(
        signed_sum, volume_sum
    ).mask(volume_sum == 0.0, 0.0)

    log_range = np.log(high / low)
    log_body = np.log(close / open_)
    window = cfg.range_vol_window
    parkinson = (log_range**2).rolling(window=window, min_periods=window).mean() / (4.0 * math.log(2.0))
    garman_klass = (
        (0.5 * log_range**2 - (2.0 * math.log(2.0) - 1.0) * log_body**2)
        .rolling(window=window, min_periods=window)
        .mean()
    )
    features[f"parkinson_vol_{window}"] = np.sqrt(parkinson)
    # The Garman-Klass variance estimate can dip below zero on tiny ranges.
    features[f"garman_klass_vol_{window}"] = np.sqrt(garman_klass.clip(lower=0.0))

    if cfg.add_candle_features:
        bar_range = high - low
        flat = bar_range == 0.0
        features["candle_body"] = _safe_ratio(close - open_, bar_range).mask(flat, 0.0)
        features["candle_upper_wick"] = _safe_ratio(
            high - np.maximum(open_, close), bar_range
        ).mask(flat, 0.0)
        features["candle_lower_wick"] = _safe_ratio(
            np.minimum(open_, close) - low, bar_range
        ).mask(flat, 0.0)
        features["candle_log_range"] = log_range

    returns = features[LOG_RET_1]
    window = cfg.moment_window
    rolling = returns.rolling(window=window, min_periods=window)
    constant = rolling.std(ddof=0) == 0.0
    # Pandas reports kurtosis -3 for a constant window; treat it as no shape.
    features[f"ret_skew_{window}"] = rolling.skew().mask(constant, 0.0)
    features[f"ret_kurt_{window}"] = rolling.kurt().mask(constant, 0.0)


def _add_htf_features(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: FeatureConfig,
    bar: pd.Timedelta,
    name: str,
    length: pd.Timedelta,
) -> None:
    """Features of the closed ``length`` candles of a gap-free segment, as-of joined."""
    close = ohlcv["close"]
    bars_per_candle = int(length / bar)
    candle_start = ohlcv.index.floor(length)
    grouped = close.groupby(candle_start)
    candles = pd.DataFrame({"close": grouped.last(), "bars": grouped.size()})
    # Only fully covered candles: the segment's first/last candle may be partial.
    candles = candles[candles["bars"] == bars_per_candle]

    htf_close = candles["close"]
    log_close = np.log(htf_close)
    htf = pd.DataFrame(index=candles.index)
    htf["log_ret_1"] = log_close.diff(1)
    htf["log_ret_3"] = log_close.diff(3)
    htf["rsi"] = _rsi(htf_close, cfg.htf_window)
    htf["ema"] = htf_close.ewm(span=cfg.htf_window, adjust=False, min_periods=cfg.htf_window).mean()
    # A candle closes together with its last bar, whose open_time is start + length - bar.
    htf.index = htf.index + length - bar
    aligned = htf.reindex(ohlcv.index, method="ffill")

    prefix = f"htf_{name}"
    features[f"{prefix}_log_ret_1"] = aligned["log_ret_1"]
    features[f"{prefix}_log_ret_3"] = aligned["log_ret_3"]
    features[f"{prefix}_rsi_{cfg.htf_window}"] = aligned["rsi"]
    features[f"{prefix}_close_ema_{cfg.htf_window}_ratio"] = _safe_ratio(close, aligned["ema"]) - 1.0
