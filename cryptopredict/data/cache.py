"""Incremental on-disk OHLCV cache."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from cryptopredict.core.types import INDEX_NAME, OHLCV_COLUMNS, validate_ohlcv
from cryptopredict.data.binance import (
    DEFAULT_BASE_URL,
    GapMode,
    TimeLike,
    _handle_gaps,
    fetch_klines,
    interval_to_milliseconds,
    timestamp_to_milliseconds,
)
from cryptopredict.data.loader import load_csv, save_csv

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")


def _empty_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME)
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=index, dtype="float64")


def _cache_path(cache_dir: str | Path, symbol: str, interval: str) -> Path:
    normalized_symbol = symbol.strip().upper()
    if not _SAFE_COMPONENT.fullmatch(normalized_symbol):
        raise ValueError(f"unsafe symbol for cache filename: {symbol!r}")
    if not _SAFE_COMPONENT.fullmatch(interval):
        raise ValueError(f"unsafe interval for cache filename: {interval!r}")
    return Path(cache_dir) / f"{normalized_symbol}_{interval}.csv"


def _as_timestamp(milliseconds: int) -> pd.Timestamp:
    return pd.Timestamp(milliseconds, unit="ms", tz="UTC")


def get_ohlcv(
    symbol: str,
    interval: str,
    start: TimeLike | None = None,
    end: TimeLike | None = None,
    cache_dir: str | Path = ".cache/cryptopredict",
    *,
    base_url: str = DEFAULT_BASE_URL,
    session: Any | None = None,
    on_gap: GapMode = "warn",
) -> pd.DataFrame:
    """Load cached candles and fetch only ranges missing on either side.

    The complete merged series remains in the cache, while the returned frame is
    sliced to the requested inclusive ``start``/``end`` bounds.
    """
    interval_ms = interval_to_milliseconds(interval)
    start_ms = timestamp_to_milliseconds(start)
    end_ms = timestamp_to_milliseconds(end)
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("start must be earlier than or equal to end")

    path = _cache_path(cache_dir, symbol, interval)
    cached = load_csv(path) if path.exists() else _empty_frame()
    additions: list[pd.DataFrame] = []

    def fetch(missing_start: int | None, missing_end: int | None) -> None:
        frame = fetch_klines(
            symbol,
            interval,
            start=missing_start,
            end=missing_end,
            base_url=base_url,
            session=session,
            on_gap="ignore",
        )
        if not frame.empty:
            additions.append(frame)

    if cached.empty:
        fetch(start_ms, end_ms)
    else:
        first_ms = int(cached.index[0].value // 1_000_000)
        last_ms = int(cached.index[-1].value // 1_000_000)
        if start_ms is not None and start_ms < first_ms:
            fetch(start_ms, first_ms - 1)
        if end_ms is None or end_ms >= last_ms + interval_ms:
            fetch(last_ms + interval_ms, end_ms)

    pieces = [cached, *additions]
    merged = pd.concat(pieces).sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    merged = validate_ohlcv(merged)
    _handle_gaps(merged, interval, on_gap)
    save_csv(merged, path)

    result = merged
    if start_ms is not None:
        result = result.loc[result.index >= _as_timestamp(start_ms)]
    if end_ms is not None:
        result = result.loc[result.index <= _as_timestamp(end_ms)]
    return validate_ohlcv(result)
