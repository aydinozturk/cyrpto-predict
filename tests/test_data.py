from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
import pytest
import requests

from cryptopredict.core.types import INDEX_NAME, OHLCV_COLUMNS
from cryptopredict.data import get_ohlcv, load_csv, save_csv
from cryptopredict.data.binance import fetch_klines

HOUR_MS = 60 * 60 * 1000


def _kline(open_ms: int, *, close_ms: int | None = None) -> list[Any]:
    return [
        open_ms,
        "100.0",
        "105.0",
        "95.0",
        "102.0",
        "12.5",
        close_ms if close_ms is not None else open_ms + HOUR_MS - 1,
        "0",
        1,
        "0",
        "0",
        "0",
    ]


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_fetch_klines_paginates_and_deduplicates(monkeypatch):
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    first = [_kline(start + offset * HOUR_MS) for offset in range(1000)]
    second = [_kline(start + 999 * HOUR_MS), _kline(start + 1000 * HOUR_MS)]
    session = FakeSession([FakeResponse(first), FakeResponse(second)])
    monkeypatch.setattr("cryptopredict.data.binance.time.sleep", lambda _: None)

    result = fetch_klines(
        "btcusdt",
        "1h",
        start=start,
        end=start + 1000 * HOUR_MS,
        session=session,
    )

    assert len(session.calls) == 2
    assert session.calls[0][1]["params"]["limit"] == 1000
    assert session.calls[1][1]["params"]["startTime"] == start + 1000 * HOUR_MS
    assert len(result) == 1001
    assert result.index.is_unique
    assert list(result.columns) == OHLCV_COLUMNS
    assert (result.dtypes == "float64").all()


def test_fetch_klines_retries_and_drops_unclosed_candle(monkeypatch):
    now_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
    closed_open = now_ms - 2 * HOUR_MS
    page = [
        _kline(closed_open, close_ms=now_ms - 1),
        _kline(closed_open + HOUR_MS, close_ms=now_ms + HOUR_MS),
    ]
    session = FakeSession(
        [FakeResponse({"code": -1003}, status_code=429, headers={"Retry-After": "0"}), FakeResponse(page)]
    )
    sleeps = []
    monkeypatch.setattr("cryptopredict.data.binance.time.sleep", sleeps.append)

    result = fetch_klines("BTCUSDT", "1h", session=session)

    assert len(session.calls) == 2
    assert sleeps == [0.0]
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp(closed_open, unit="ms", tz="UTC")


def test_fetch_klines_does_not_retry_non_retryable_client_error(monkeypatch):
    session = FakeSession([FakeResponse({"code": -1121}, status_code=400)])
    sleeps = []
    monkeypatch.setattr("cryptopredict.data.binance.time.sleep", sleeps.append)

    with pytest.raises(requests.HTTPError, match="400"):
        fetch_klines("NOT_A_SYMBOL", "1h", session=session)

    assert len(session.calls) == 1
    assert sleeps == []


def test_fetch_klines_warns_and_returns_frame_with_missing_candle():
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    session = FakeSession([FakeResponse([_kline(start), _kline(start + 2 * HOUR_MS)])])

    with pytest.warns(RuntimeWarning, match="1 missing-candle gap"):
        result = fetch_klines("BTCUSDT", "1h", start=start, session=session)

    assert len(result) == 2


def test_fetch_klines_can_raise_or_ignore_missing_candle():
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    page = [_kline(start), _kline(start + 2 * HOUR_MS)]

    with pytest.raises(ValueError, match="1 missing-candle gap"):
        fetch_klines(
            "BTCUSDT",
            "1h",
            start=start,
            session=FakeSession([FakeResponse(page)]),
            on_gap="raise",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fetch_klines(
            "BTCUSDT",
            "1h",
            start=start,
            session=FakeSession([FakeResponse(page)]),
            on_gap="ignore",
        )
    assert caught == []
    assert len(result) == 2


def test_fetch_klines_always_rejects_irregular_interval():
    start = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
    page = [_kline(start), _kline(start + HOUR_MS + HOUR_MS // 2)]

    with pytest.raises(ValueError, match="irregular 1h candle interval"):
        fetch_klines(
            "BTCUSDT",
            "1h",
            start=start,
            session=FakeSession([FakeResponse(page)]),
            on_gap="ignore",
        )


def test_csv_round_trip_preserves_contract(tmp_path, ohlcv):
    path = tmp_path / "nested" / "sample.csv"

    assert save_csv(ohlcv.iloc[:10], path) == path
    restored = load_csv(path)

    pd.testing.assert_frame_equal(restored, ohlcv.iloc[:10], check_freq=False)
    assert restored.index.name == INDEX_NAME
    assert str(restored.index.tz) == "UTC"


def test_get_ohlcv_fetches_and_merges_missing_cache_tail(monkeypatch, tmp_path, ohlcv):
    path = tmp_path / "BTCUSDT_1h.csv"
    save_csv(ohlcv.iloc[:3], path)
    calls = []

    def fake_fetch(symbol, interval, **kwargs):
        calls.append((symbol, interval, kwargs))
        return ohlcv.iloc[3:5]

    monkeypatch.setattr("cryptopredict.data.cache.fetch_klines", fake_fetch)

    result = get_ohlcv(
        "BTCUSDT",
        "1h",
        start=ohlcv.index[0],
        end=ohlcv.index[4],
        cache_dir=tmp_path,
    )

    assert len(calls) == 1
    assert calls[0][2]["start"] == int(ohlcv.index[3].value // 1_000_000)
    assert calls[0][2]["end"] == int(ohlcv.index[4].value // 1_000_000)
    pd.testing.assert_frame_equal(result, ohlcv.iloc[:5], check_freq=False)
    pd.testing.assert_frame_equal(load_csv(path), ohlcv.iloc[:5], check_freq=False)


def test_get_ohlcv_uses_cache_when_range_is_covered(monkeypatch, tmp_path, ohlcv):
    save_csv(ohlcv.iloc[:5], tmp_path / "BTCUSDT_1h.csv")

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("covered cache range must not hit the network")

    monkeypatch.setattr("cryptopredict.data.cache.fetch_klines", unexpected_fetch)

    result = get_ohlcv(
        "BTCUSDT",
        "1h",
        start=ohlcv.index[1],
        end=ohlcv.index[3],
        cache_dir=tmp_path,
    )

    pd.testing.assert_frame_equal(result, ohlcv.iloc[1:4], check_freq=False)


def test_get_ohlcv_warns_but_uses_cache_with_exchange_gap(tmp_path, ohlcv):
    gapped = ohlcv.iloc[[0, 1, 3, 4]]
    save_csv(gapped, tmp_path / "BTCUSDT_1h.csv")

    with pytest.warns(RuntimeWarning, match="1 missing-candle gap"):
        result = get_ohlcv(
            "BTCUSDT",
            "1h",
            start=gapped.index[0],
            end=gapped.index[-1],
            cache_dir=tmp_path,
        )

    pd.testing.assert_frame_equal(result, gapped, check_freq=False)
