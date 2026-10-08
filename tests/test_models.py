import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from cryptopredict.core.types import LOG_RET_1, Forecaster
from cryptopredict.models import (
    GBMForecaster,
    LastReturn,
    MeanReturn,
    MovingAverageReturn,
    RidgeForecaster,
    ZeroReturn,
    available_models,
    get_model,
    load_model,
    save_model,
)

ALL_NAMES = ["zero", "mean", "last", "ma", "ridge", "gbm"]


def make_xy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic returns with a linear signal plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="open_time")
    X = pd.DataFrame(
        {
            LOG_RET_1: rng.normal(0, 0.01, n),
            "rsi_14": rng.uniform(0, 100, n),
            "vol_24": rng.uniform(0.001, 0.02, n),
        },
        index=idx,
    )
    y = 0.5 * X[LOG_RET_1] + 0.0002 * (X["rsi_14"] - 50) + rng.normal(0, 0.002, n)
    return X, y.rename("target")


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def test_registry():
    assert available_models() == ALL_NAMES
    assert isinstance(get_model("ridge", alpha=3.0), RidgeForecaster)
    assert get_model("ridge", alpha=3.0).alpha == 3.0
    assert get_model("ma", window=5).window == 5
    with pytest.raises(ValueError, match="unknown model"):
        get_model("lstm")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_predict_shape(name):
    X, y = make_xy()
    model = get_model(name)
    assert model.name == name
    assert isinstance(model, Forecaster)
    assert model.fit(X.iloc[:300], y.iloc[:300]) is model
    pred = model.predict(X.iloc[300:])
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (100,)
    assert pred.dtype == np.float64
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_clone_gives_unfitted_copy(name):
    model = get_model(name)
    copy = clone(model)
    assert type(copy) is type(model)
    assert copy.get_params() == model.get_params()


def test_baseline_values():
    X, y = make_xy()
    assert (ZeroReturn().fit(X, y).predict(X) == 0).all()
    assert np.allclose(MeanReturn().fit(X, y).predict(X), y.mean())
    assert np.allclose(LastReturn().fit(X, y).predict(X), X[LOG_RET_1])
    assert np.allclose(MovingAverageReturn(window=10).fit(X, y).predict(X.iloc[:3]), y.iloc[-10:].mean())
    assert np.allclose(MovingAverageReturn(column="vol_24").fit(X, y).predict(X), X["vol_24"])


def test_baseline_input_errors():
    X, y = make_xy()
    with pytest.raises(ValueError, match="log_ret_1"):
        LastReturn().fit(X.drop(columns=LOG_RET_1), y)
    with pytest.raises(ValueError, match="window"):
        MovingAverageReturn(window=0).fit(X, y)
    with pytest.raises(ValueError, match="not in X"):
        MovingAverageReturn(column="nope").fit(X, y)
    with pytest.raises(ValueError, match="length"):
        ZeroReturn().fit(X, y.iloc[:-1])
    y_nan = y.copy()
    y_nan.iloc[-1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        RidgeForecaster().fit(X, y_nan)


def test_ridge_beats_zero_on_linear_data():
    X, y = make_xy(n=600)
    train, test = slice(0, 450), slice(450, None)
    ridge = RidgeForecaster(alpha=1.0).fit(X.iloc[train], y.iloc[train])
    zero = ZeroReturn().fit(X.iloc[train], y.iloc[train])
    assert mae(ridge.predict(X.iloc[test]), y.iloc[test]) < 0.5 * mae(zero.predict(X.iloc[test]), y.iloc[test])


def test_gbm_learns_and_is_deterministic():
    X, y = make_xy(n=600)
    a = GBMForecaster().fit(X.iloc[:450], y.iloc[:450]).predict(X.iloc[450:])
    b = GBMForecaster().fit(X.iloc[:450], y.iloc[:450]).predict(X.iloc[450:])
    assert np.array_equal(a, b)
    assert mae(a, y.iloc[450:]) < mae(np.zeros_like(a), y.iloc[450:])


def test_predict_uses_training_column_order():
    X, y = make_xy()
    model = RidgeForecaster().fit(X, y)
    shuffled = X[["vol_24", LOG_RET_1, "rsi_14"]].assign(extra=1.0)
    assert np.allclose(model.predict(shuffled), model.predict(X))
    with pytest.raises(ValueError, match="missing"):
        model.predict(X.drop(columns="rsi_14"))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_save_load_roundtrip(tmp_path, name):
    X, y = make_xy()
    model = get_model(name).fit(X.iloc[:300], y.iloc[:300])
    path = tmp_path / "models" / f"{name}.joblib"
    meta = save_model(model, path, horizon=3, extra={"symbol": "BTCUSDT"})
    loaded, loaded_meta = load_model(path)
    assert loaded_meta == meta
    assert meta["name"] == name
    assert meta["feature_columns"] == list(X.columns)
    assert meta["horizon"] == 3
    assert meta["symbol"] == "BTCUSDT"
    assert np.array_equal(loaded.predict(X.iloc[300:]), model.predict(X.iloc[300:]))


def test_save_unfitted_needs_columns(tmp_path):
    with pytest.raises(ValueError, match="feature_columns"):
        save_model(ZeroReturn(), tmp_path / "m.joblib")
    meta = save_model(ZeroReturn(), tmp_path / "m.joblib", feature_columns=[LOG_RET_1])
    assert meta["feature_columns"] == [LOG_RET_1]


def test_load_rejects_foreign_file(tmp_path):
    import joblib

    path = tmp_path / "x.joblib"
    joblib.dump({"foo": 1}, path)
    with pytest.raises(ValueError, match="not a cryptopredict model"):
        load_model(path)


def test_models_on_real_fixture(ohlcv):
    """Smoke test on real bars with a minimal hand-made feature set."""
    close = ohlcv["close"]
    X = pd.DataFrame({LOG_RET_1: np.log(close / close.shift(1))}, index=ohlcv.index)
    X["ret_mean_24"] = X[LOG_RET_1].rolling(24).mean()
    y = np.log(close.shift(-1) / close).rename("target")
    data = X.join(y).dropna()
    X, y = data.drop(columns="target"), data["target"]
    for name in ALL_NAMES:
        pred = get_model(name).fit(X.iloc[:400], y.iloc[:400]).predict(X.iloc[400:])
        assert pred.shape == (len(X) - 400,) and np.isfinite(pred).all()
