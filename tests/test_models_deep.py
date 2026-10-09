import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

pytest.importorskip("torch")

from cryptopredict.evaluation import walk_forward_evaluate  # noqa: E402
from cryptopredict.models import load_model, save_model  # noqa: E402
from cryptopredict.models.deep import GRUForecaster, LSTMForecaster  # noqa: E402

FAST = {"lookback": 24, "hidden_size": 16, "max_epochs": 40, "patience": 5, "max_train_seconds": 20.0}
MODELS = [LSTMForecaster, GRUForecaster]


def _sine(n=900, seed=0):
    """One feature (a noisy daily sine); the target is its next value."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n + 1, freq="1h", tz="UTC", name="open_time")
    values = np.sin(2 * np.pi * np.arange(n + 1) / 24) + rng.normal(0.0, 0.1, n + 1)
    X = pd.DataFrame({"x": values[:-1]}, index=index[:-1])
    y = pd.Series(values[1:], index=index[:-1], name="target")
    return X, y


@pytest.fixture(scope="module", params=MODELS, ids=lambda cls: cls.name)
def fitted(request):
    X, y = _sine()
    model = request.param(**FAST).fit(X.iloc[:700], y.iloc[:700])
    return model, X, y


def test_learns_a_sequence_pattern_better_than_zero(fitted):
    model, X, y = fitted
    pred = model.predict(X.iloc[700:])
    assert pred.shape == (200,)
    mae = np.mean(np.abs(y.iloc[700:] - pred))
    assert mae < 0.5 * np.mean(np.abs(y.iloc[700:]))
    assert model.n_epochs_ >= 1 and not model.stopped_by_time_


def test_same_seed_gives_identical_predictions(fitted):
    model, X, y = fitted
    again = type(model)(**FAST).fit(X.iloc[:700], y.iloc[:700])
    np.testing.assert_array_equal(again.predict(X.iloc[700:]), model.predict(X.iloc[700:]))


def test_predictions_never_look_ahead(fitted):
    model, X, _ = fitted
    expected = model.predict(X)
    changed = X.copy()
    changed.iloc[600:, 0] += 50.0
    np.testing.assert_array_equal(model.predict(changed)[:600], expected[:600])
    assert not np.allclose(model.predict(changed)[600:], expected[600:])


def test_rows_after_training_use_training_history(fitted):
    model, X, _ = fitted
    # A test block right after training equals predicting the joined series.
    np.testing.assert_allclose(model.predict(X.iloc[700:]), model.predict(X).astype("float64")[700:], rtol=0, atol=1e-6)
    # Live use: the final `lookback` rows alone give the same last forecast.
    tail = X.iloc[-model.lookback:]
    assert model.predict(tail)[-1] == pytest.approx(model.predict(X)[-1], abs=1e-6)


def test_windows_do_not_cross_a_gap(fitted):
    model, X, _ = fitted
    gapped = X.drop(X.index[750])
    expected = model.predict(gapped)
    changed = gapped.copy()
    changed.iloc[:750, 0] -= 50.0  # everything before the gap
    np.testing.assert_array_equal(model.predict(changed)[750:], expected[750:])


def test_clone_and_walk_forward(fitted):
    model, X, y = fitted
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert not hasattr(copy, "network_")
    assert type(model).heavy is True and model.lookback == FAST["lookback"]

    result = walk_forward_evaluate(type(model)(**FAST), X, y, n_splits=2, gap=1)
    assert len(result.predictions) > 0
    assert np.isfinite(result.predictions["y_pred"]).all()
    assert result.overall["mae"] < np.mean(np.abs(result.predictions["y_true"]))


def test_save_load_round_trip(fitted, tmp_path):
    model, X, _ = fitted
    path = tmp_path / f"{model.name}.joblib"
    save_model(model, path)
    loaded, metadata = load_model(path)
    assert metadata["name"] == model.name
    np.testing.assert_array_equal(loaded.predict(X.iloc[700:]), model.predict(X.iloc[700:]))


@pytest.mark.parametrize("cls", MODELS, ids=lambda cls: cls.name)
def test_too_little_data_is_a_clear_error(cls):
    X, y = _sine(30)
    with pytest.raises(ValueError, match="lookback=48"):
        cls().fit(X.iloc[:20], y.iloc[:20])
    gapped_index = X.index[:20].append(X.index[25:30])
    with pytest.raises(ValueError, match="gap-free window"):
        cls(lookback=21).fit(X.loc[gapped_index], y.loc[gapped_index])


def test_invalid_hyperparameters_are_rejected():
    X, y = _sine(100)
    for params in ({"lookback": 0}, {"dropout": 1.0}, {"learning_rate": 0}, {"validation_fraction": 1.0}):
        with pytest.raises(ValueError):
            LSTMForecaster(**params).fit(X, y)
    with pytest.raises(ValueError, match="not fitted"):
        GRUForecaster().predict(X)


def test_default_fit_on_2000_rows_is_fast_and_time_budget_holds():
    rng = np.random.default_rng(1)
    index = pd.date_range("2026-01-01", periods=2000, freq="1h", tz="UTC", name="open_time")
    X = pd.DataFrame(rng.normal(size=(2000, 40)), index=index).add_prefix("f")
    y = pd.Series(rng.normal(0.0, 0.01, 2000), index=index)

    started = time.monotonic()
    model = LSTMForecaster().fit(X, y)
    assert time.monotonic() - started < 30.0
    assert model.predict(X).shape == (2000,)

    budget = GRUForecaster(max_epochs=10_000, patience=10_000, max_train_seconds=0.5).fit(X, y)
    assert budget.stopped_by_time_
    assert budget.fit_seconds_ < 5.0
