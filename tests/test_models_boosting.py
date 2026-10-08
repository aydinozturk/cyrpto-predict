import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

pytest.importorskip("lightgbm")

from cryptopredict.core.types import LOG_RET_1, Forecaster  # noqa: E402
from cryptopredict.evaluation.walkforward import walk_forward_evaluate  # noqa: E402
from cryptopredict.models import ZeroReturn, load_model, save_model  # noqa: E402
from cryptopredict.models.boosting import LGBMDirectionForecaster, LGBMForecaster  # noqa: E402

MODELS = [LGBMForecaster, LGBMDirectionForecaster]


def make_ar(n: int = 2000, phi: float = 0.4, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """AR(1) returns plus a nonlinear feature signal; y[t] is the next return."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC", name="open_time")
    regime = rng.uniform(-1, 1, n)
    ret = np.zeros(n + 1)
    for t in range(1, n + 1):
        ret[t] = phi * ret[t - 1] + 0.004 * np.sign(regime[t - 1]) + rng.normal(0, 0.004)
    X = pd.DataFrame(
        {LOG_RET_1: ret[:-1], "regime": regime, "noise": rng.normal(0, 1, n)},
        index=idx,
    )
    return X, pd.Series(ret[1:], index=idx, name="target")


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


@pytest.mark.parametrize("cls", MODELS)
def test_beats_zero_on_signal(cls):
    X, y = make_ar()
    model = cls().fit(X.iloc[:1500], y.iloc[:1500])
    assert isinstance(model, Forecaster)
    pred = model.predict(X.iloc[1500:])
    assert pred.shape == (500,) and np.isfinite(pred).all()
    assert mae(y.iloc[1500:], pred) < 0.9 * mae(y.iloc[1500:], np.zeros(500))
    assert 1 <= model.best_iteration_ <= model.n_estimators


@pytest.mark.parametrize("cls", MODELS)
def test_deterministic(cls):
    X, y = make_ar(800)
    a = cls().fit(X.iloc[:600], y.iloc[:600]).predict(X.iloc[600:])
    b = cls().fit(X.iloc[:600], y.iloc[:600]).predict(X.iloc[600:])
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls", MODELS)
def test_clone_and_walk_forward(cls):
    X, y = make_ar(1200)
    model = cls(n_estimators=200, num_leaves=7)
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert not hasattr(copy, "model_")
    res = walk_forward_evaluate(model, X, y, n_splits=3)
    zero = walk_forward_evaluate(ZeroReturn(), X, y, n_splits=3)
    assert len(res.folds) == 3
    assert len(res.predictions) == len(zero.predictions)
    assert res.overall["mae"] < zero.overall["mae"]


@pytest.mark.parametrize("cls", MODELS)
def test_save_load_round_trip(cls, tmp_path):
    X, y = make_ar(600)
    model = cls(n_estimators=100).fit(X.iloc[:500], y.iloc[:500])
    meta = save_model(model, tmp_path / "m.joblib")
    assert meta["name"] == cls.name
    loaded, meta2 = load_model(tmp_path / "m.joblib")
    assert meta2["feature_columns"] == list(X.columns)
    np.testing.assert_array_equal(loaded.predict(X.iloc[500:]), model.predict(X.iloc[500:]))


@pytest.mark.parametrize("cls", MODELS)
def test_nan_tolerant_and_column_order(cls):
    X, y = make_ar(600)
    X = X.copy()
    X.iloc[::7, 1] = np.nan
    model = cls(n_estimators=100).fit(X.iloc[:500], y.iloc[:500])
    test = X.iloc[500:]
    pred = model.predict(test)
    assert np.isfinite(pred).all()
    np.testing.assert_array_equal(model.predict(test[test.columns[::-1]]), pred)
    with pytest.raises(ValueError, match="missing training feature"):
        model.predict(test.drop(columns=["regime"]))


def test_early_stopping_uses_only_training_tail(monkeypatch):
    X, y = make_ar(1000)
    seen = []
    orig = LGBMForecaster._fit_estimator

    def spy(self, values, target):
        seen.append(len(target))
        return orig(self, values, target)

    monkeypatch.setattr(LGBMForecaster, "_fit_estimator", spy)
    model = LGBMForecaster(early_stopping_rounds=20).fit(X.iloc[:800], y.iloc[:800])
    assert seen == [800]
    assert model.best_iteration_ < model.n_estimators
    assert model.model_.n_estimators == model.best_iteration_  # refit on all 800 rows
    no_refit = LGBMForecaster(early_stopping_rounds=20, refit=False).fit(X.iloc[:800], y.iloc[:800])
    assert no_refit.best_iteration_ == model.best_iteration_
    assert no_refit.model_.n_estimators == model.n_estimators


def test_no_early_stopping_on_tiny_or_disabled():
    X, y = make_ar(100)
    assert LGBMForecaster(n_estimators=30).fit(X, y).best_iteration_ == 30
    X, y = make_ar(600)
    model = LGBMForecaster(n_estimators=40, early_stopping_rounds=None).fit(X, y)
    assert model.best_iteration_ == 40


def test_huber_objective():
    X, y = make_ar(800)
    model = LGBMForecaster(objective="huber").fit(X.iloc[:600], y.iloc[:600])
    assert 0 < model.huber_delta_ < 0.1  # scaled to the returns, not LightGBM's 0.9
    assert np.isfinite(model.predict(X.iloc[600:])).all()
    assert LGBMForecaster(objective="huber", huber_delta=0.01).fit(X, y).huber_delta_ == 0.01
    with pytest.raises(ValueError, match="objective"):
        LGBMForecaster(objective="quantile").fit(X, y)


@pytest.mark.parametrize("magnitude", ["conditional", "symmetric"])
def test_direction_proba_and_sign(magnitude):
    X, y = make_ar()
    model = LGBMDirectionForecaster(magnitude=magnitude).fit(X.iloc[:1500], y.iloc[:1500])
    p = model.predict_proba(X.iloc[1500:])
    pred = model.predict(X.iloc[1500:])
    assert p.shape == (500,) and ((p >= 0) & (p <= 1)).all()
    assert model.down_mean_ < 0 < model.up_mean_
    np.testing.assert_allclose(pred, p * model.up_mean_ + (1 - p) * model.down_mean_)
    np.testing.assert_array_equal(pred > 0, p > model.breakeven_proba_)
    if magnitude == "symmetric":
        assert model.breakeven_proba_ == pytest.approx(0.5)
        np.testing.assert_array_equal(pred > 0, p > 0.5)
    # The classifier actually learned the direction of the regime signal.
    up = y.iloc[1500:].to_numpy() > 0
    assert np.mean((p > 0.5) == up) > 0.6


def test_direction_single_class():
    X, y = make_ar(300)
    model = LGBMDirectionForecaster().fit(X, y.abs() + 1e-4)
    np.testing.assert_array_equal(model.predict_proba(X), np.ones(len(X)))
    assert (model.predict(X) > 0).all()
    with pytest.raises(ValueError, match="magnitude"):
        LGBMDirectionForecaster(magnitude="x").fit(X, y)
