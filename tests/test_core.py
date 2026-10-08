import numpy as np
import pandas as pd
import pytest

from cryptopredict.core.types import (
    INDEX_NAME,
    OHLCV_COLUMNS,
    Forecaster,
    validate_ohlcv,
)


def _frame(n: int = 5, tz: str | None = "UTC") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz=tz, name=INDEX_NAME)
    data = {c: np.arange(n, dtype="float64") + 1 for c in OHLCV_COLUMNS}
    return pd.DataFrame(data, index=idx)


def test_fixture_matches_contract(ohlcv):
    assert len(ohlcv) == 600
    assert list(ohlcv.columns) == OHLCV_COLUMNS
    assert ohlcv.index.name == INDEX_NAME
    assert str(ohlcv.index.tz) == "UTC"
    assert ohlcv.index.is_monotonic_increasing and ohlcv.index.is_unique
    assert (ohlcv.dtypes == "float64").all()
    # hourly bars without gaps
    assert (ohlcv.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (ohlcv["high"] >= ohlcv[["open", "close", "low"]].max(axis=1)).all()
    assert (ohlcv["low"] <= ohlcv[["open", "close"]].min(axis=1)).all()


def test_valid_frame_is_normalized():
    df = _frame().astype({"volume": "int64"})
    df["extra"] = 1.0
    df = df[["close", "extra", "volume", "open", "high", "low"]]
    out = validate_ohlcv(df)
    assert list(out.columns) == OHLCV_COLUMNS
    assert (out.dtypes == "float64").all()
    assert out is not df


def test_naive_index_rejected():
    with pytest.raises(ValueError, match="UTC"):
        validate_ohlcv(_frame(tz=None))


def test_non_utc_index_rejected():
    with pytest.raises(ValueError, match="UTC"):
        validate_ohlcv(_frame(tz="Europe/Istanbul"))


def test_duplicate_index_rejected():
    df = _frame()
    df = pd.concat([df, df.iloc[[-1]]])
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(df)


def test_unsorted_index_rejected():
    with pytest.raises(ValueError, match="sorted"):
        validate_ohlcv(_frame().iloc[::-1])


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="volume"):
        validate_ohlcv(_frame().drop(columns="volume"))


def test_wrong_index_name_rejected():
    with pytest.raises(ValueError, match=INDEX_NAME):
        validate_ohlcv(_frame().rename_axis("time"))


def test_non_datetime_index_rejected():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv(_frame().reset_index(drop=True))


def test_nan_rejected():
    df = _frame()
    df.iloc[2, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_ohlcv(df)


def test_non_numeric_rejected():
    df = _frame()
    df["close"] = df["close"].astype(str)
    with pytest.raises(ValueError, match="numeric"):
        validate_ohlcv(df)


def test_forecaster_protocol_runtime_check():
    class Last:
        name = "last"

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    assert isinstance(Last(), Forecaster)
    assert not isinstance(object(), Forecaster)
