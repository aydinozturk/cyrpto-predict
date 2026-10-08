"""Binance public kline client.

Only public REST endpoints are used; no API key is required. Returned frames
always satisfy the shared UTC OHLCV contract.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import requests

from cryptopredict.core.types import INDEX_NAME, OHLCV_COLUMNS, validate_ohlcv

DEFAULT_BASE_URL = os.getenv("CRYPTOPREDICT_BINANCE_URL", "https://api.binance.com")
KLINES_PATH = "/api/v3/klines"
MAX_PAGE_SIZE = 1000
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 3
BACKOFF_SECONDS = 0.25
PAGE_DELAY_SECONDS = 0.05
RETRYABLE_STATUS_CODES = {418, 429, 500, 502, 503, 504}

_INTERVAL_MILLISECONDS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}

TimeLike = str | int | float | datetime | pd.Timestamp
GapMode = Literal["warn", "raise", "ignore"]


def interval_to_milliseconds(interval: str) -> int:
    """Return the duration of a fixed Binance interval in milliseconds."""
    try:
        return _INTERVAL_MILLISECONDS[interval]
    except KeyError as exc:
        supported = ", ".join(_INTERVAL_MILLISECONDS)
        raise ValueError(
            f"unsupported or non-fixed Binance interval {interval!r}; "
            f"supported: {supported}"
        ) from exc


def timestamp_to_milliseconds(value: TimeLike | None) -> int | None:
    """Normalize a timestamp-like value to Unix milliseconds.

    Integer and floating-point inputs are interpreted as Unix milliseconds.
    Naive datetime-like values are interpreted as UTC.
    """
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        return int(value)
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp cannot be NaT")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.value // 1_000_000)


def _retry_delay(response: Any, attempt: int) -> float:
    retry_after = getattr(response, "headers", {}).get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    return BACKOFF_SECONDS * (2**attempt)


def _request_page(session: Any, url: str, params: dict[str, Any]) -> list[list[Any]]:
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(BACKOFF_SECONDS * (2**attempt))
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            if attempt == MAX_RETRIES:
                response.raise_for_status()
            time.sleep(_retry_delay(response, attempt))
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"unexpected Binance response: {payload!r}")
        return payload
    raise RuntimeError("Binance request failed without an error")  # pragma: no cover


def find_gaps(
    df: pd.DataFrame, interval: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return pairs surrounding missing candles.

    A gap must span an exact multiple of the requested interval. A non-multiple
    indicates malformed/misaligned data and always raises rather than being
    treated as an exchange maintenance gap.
    """
    if len(df) < 2:
        return []
    expected = interval_to_milliseconds(interval)
    timestamps = df.index
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for previous, current in zip(timestamps[:-1], timestamps[1:], strict=True):
        actual = int((current.value - previous.value) // 1_000_000)
        if actual <= 0 or actual % expected != 0:
            raise ValueError(
                f"irregular {interval} candle interval between "
                f"{previous.isoformat()} and {current.isoformat()}: "
                f"expected a positive multiple of {expected} ms, got {actual} ms"
            )
        if actual > expected:
            gaps.append((previous, current))
    return gaps


def _gap_message(
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]], interval: str
) -> str:
    previous, current = gaps[0]
    return (
        f"found {len(gaps)} missing-candle gap(s) for {interval}; "
        f"first gap is between {previous.isoformat()} and {current.isoformat()}"
    )


def validate_interval_continuity(df: pd.DataFrame, interval: str) -> None:
    """Raise when a frame contains missing or irregular fixed-interval bars."""
    gaps = find_gaps(df, interval)
    if gaps:
        previous, current = gaps[0]
        actual = int((current.value - previous.value) // 1_000_000)
        expected = interval_to_milliseconds(interval)
        raise ValueError(
            f"{_gap_message(gaps, interval)}; expected {expected} ms, got {actual} ms"
        )


def _handle_gaps(df: pd.DataFrame, interval: str, on_gap: GapMode) -> None:
    if on_gap not in {"warn", "raise", "ignore"}:
        raise ValueError("on_gap must be one of: 'warn', 'raise', 'ignore'")
    gaps = find_gaps(df, interval)
    if not gaps or on_gap == "ignore":
        return
    if on_gap == "raise":
        validate_interval_continuity(df, interval)
        return  # pragma: no cover - validation always raises when gaps exist
    warnings.warn(_gap_message(gaps, interval), RuntimeWarning, stacklevel=2)


def _rows_to_frame(rows: list[list[Any]], *, now_ms: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            raise ValueError(f"malformed Binance kline row: {row!r}")
        if int(row[6]) > now_ms:
            continue
        records.append(
            {
                INDEX_NAME: int(row[0]),
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5],
            }
        )

    if not records:
        index = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME)
        return validate_ohlcv(pd.DataFrame(columns=OHLCV_COLUMNS, index=index, dtype="float64"))

    frame = pd.DataFrame.from_records(records)
    frame[INDEX_NAME] = pd.to_datetime(frame[INDEX_NAME], unit="ms", utc=True)
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = (
        frame.set_index(INDEX_NAME)
        .sort_index()
        .loc[lambda value: ~value.index.duplicated(keep="last")]
    )
    return validate_ohlcv(frame)


def fetch_klines(
    symbol: str,
    interval: str,
    start: TimeLike | None = None,
    end: TimeLike | None = None,
    limit: int | None = None,
    base_url: str = DEFAULT_BASE_URL,
    session: Any | None = None,
    on_gap: GapMode = "warn",
) -> pd.DataFrame:
    """Fetch closed Binance candles, paging forward in batches of at most 1000.

    ``limit`` is a total result cap. With neither a start nor end bound, Binance's
    latest page is returned. Duplicate candles are resolved in favor of the last
    response. Exchange maintenance gaps warn by default; ``on_gap`` may instead
    raise or ignore them. Misaligned intervals always raise.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be non-empty")
    interval_ms = interval_to_milliseconds(interval)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")
    if limit is not None:
        limit = int(limit)

    start_ms = timestamp_to_milliseconds(start)
    end_ms = timestamp_to_milliseconds(end)
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("start must be earlier than or equal to end")

    http = session if session is not None else requests.Session()
    owns_session = session is None
    cursor = start_ms
    remaining = limit
    rows: list[list[Any]] = []
    one_unbounded_page = start_ms is None and end_ms is None
    url = f"{base_url.rstrip('/')}{KLINES_PATH}"

    try:
        while True:
            page_size = min(MAX_PAGE_SIZE, remaining) if remaining is not None else MAX_PAGE_SIZE
            params: dict[str, Any] = {
                "symbol": symbol.strip().upper(),
                "interval": interval,
                "limit": page_size,
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms

            page = _request_page(http, url, params)
            if not page:
                break
            rows.extend(page)

            if remaining is not None:
                remaining -= len(page)
                if remaining <= 0:
                    break
            if one_unbounded_page or len(page) < page_size:
                break

            last_open_ms = int(page[-1][0])
            next_cursor = last_open_ms + interval_ms
            if cursor is not None and next_cursor <= cursor:
                raise ValueError("Binance pagination did not advance")
            if end_ms is not None and next_cursor > end_ms:
                break
            cursor = next_cursor
            time.sleep(PAGE_DELAY_SECONDS)
    finally:
        if owns_session:
            http.close()

    frame = _rows_to_frame(rows, now_ms=int(pd.Timestamp.now(tz="UTC").value // 1_000_000))
    if limit is not None:
        frame = frame.iloc[:limit]
    _handle_gaps(frame, interval, on_gap)
    return frame
