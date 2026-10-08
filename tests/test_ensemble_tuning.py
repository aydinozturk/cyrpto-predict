import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from cryptopredict import tuning
from cryptopredict.core.types import LOG_RET_1, Forecaster
from cryptopredict.evaluation.walkforward import walk_forward_evaluate
from cryptopredict.models import RidgeForecaster, load_model, save_model
from cryptopredict.models.ensemble import EnsembleForecaster, StackingForecaster
from cryptopredict.tuning import DEFAULT_SPACES, candidates, nested_walk_forward, tune

FAST = ("ridge", ("gbm", {"max_iter": 30}))


def make_xy(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
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


# --- ensemble / stacking -------------------------------------------------------------


@pytest.mark.parametrize("cls", [EnsembleForecaster, StackingForecaster])
def test_clone_walk_forward_and_round_trip(cls, tmp_path):
    X, y = make_xy()
    model = cls(members=FAST)
    assert isinstance(model, Forecaster)
    copy = clone(model)
    assert copy.get_params() == model.get_params() and not hasattr(copy, "models_")
    res = walk_forward_evaluate(model, X, y, n_splits=3)
    assert len(res.folds) == 3 and res.overall["relative_mae"] < 1.0

    fitted = cls(members=FAST).fit(X.iloc[:500], y.iloc[:500])
    assert fitted.member_names_ == ["ridge", "gbm"]
    save_model(fitted, tmp_path / "m.joblib")
    loaded, meta = load_model(tmp_path / "m.joblib")
    assert meta["name"] == cls.name
    np.testing.assert_array_equal(loaded.predict(X.iloc[500:]), fitted.predict(X.iloc[500:]))
    test = X.iloc[500:]
    np.testing.assert_array_equal(fitted.predict(test[test.columns[::-1]]), fitted.predict(test))


def test_ensemble_weights_and_missing_member():
    X, y = make_xy()
    with pytest.warns(UserWarning, match="'nope' skipped"):
        model = EnsembleForecaster(members=("ridge", "nope", "mean"), weights=(3.0, 5.0, 1.0)).fit(X, y)
    assert model.member_names_ == ["ridge", "mean"]
    np.testing.assert_allclose(model.weights_, [0.75, 0.25])
    ridge = RidgeForecaster().fit(X, y).predict(X)
    np.testing.assert_allclose(model.predict(X), 0.75 * ridge + 0.25 * y.mean())
    with pytest.raises(ValueError, match="no usable"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            EnsembleForecaster(members=("nope",)).fit(X, y)
    with pytest.raises(ValueError, match="weights"):
        EnsembleForecaster(members=("ridge",), weights=(1.0, 1.0)).fit(X, y)


def test_default_members_fit_without_optional_dependencies():
    X, y = make_xy(300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lgbm may be unavailable
        model = EnsembleForecaster().fit(X, y)
    assert {"ridge", "gbm"} <= set(model.member_names_)


def test_stack_beats_worst_member_and_uses_oof_only():
    X, y = make_xy(900)
    members = ("ridge", ("gbm", {"max_iter": 50}), "mean")
    train, test = slice(0, 700), slice(700, None)
    stack = StackingForecaster(members=members).fit(X.iloc[train], y.iloc[train])
    assert (stack.coef_ >= 0).all() and stack.intercept_ == 0.0
    assert stack.n_oof_ < 700  # the first inner training window never gets an OOF forecast
    stack_mae = mae(y.iloc[test], stack.predict(X.iloc[test]))
    member_maes = [mae(y.iloc[test], m.predict(X.iloc[test])) for m in stack.models_]
    assert stack_mae < max(member_maes)
    assert stack_mae <= 1.05 * min(member_maes)
    # The linear signal dominates: ridge should get the largest weight.
    assert int(np.argmax(stack.coef_ * np.std([m.predict(X) for m in stack.models_], axis=1))) == 0


# --- tuning ---------------------------------------------------------------------------


def test_candidates_defaults_first_and_seeded():
    a = candidates("ridge", n_trials=4, seed=1)
    assert a[0] == {} and len(a) == 4
    assert a == candidates("ridge", n_trials=4, seed=1)
    assert a != candidates("ridge", n_trials=4, seed=2)
    assert all(set(p) <= set(DEFAULT_SPACES["ridge"]) for p in a)
    grid = candidates("ridge", param_grid={"alpha": [1.0, 5.0, 9.0]}, n_trials=None)
    assert grid == [{}, {"alpha": 1.0}, {"alpha": 5.0}, {"alpha": 9.0}]
    assert candidates("zero") == [{}]
    with pytest.raises(ValueError, match="either"):
        candidates("ridge", param_grid={"alpha": [1]}, param_distributions={"alpha": [1]})


def test_tune_reproducible_and_picks_best():
    X, y = make_xy()
    a = tune("ridge", X, y, n_trials=5, seed=3)
    b = tune("ridge", X, y, n_trials=5, seed=3)
    pd.testing.assert_frame_equal(a.trials.drop(columns="seconds"), b.trials.drop(columns="seconds"))
    assert a.best_params == b.best_params and not a.timed_out
    assert a.best_score == a.trials["score"].min()
    # The heavily shrunk ridge cannot be the winner on this strong linear signal.
    r = tune("ridge", X, y, param_grid={"alpha": [1e-4, 1e4]})
    assert r.best_params == {"alpha": 1e-4}
    hi = tune("ridge", X, y, param_grid={"alpha": [1e-4, 1e4]}, metric="directional_accuracy")
    assert hi.best_score == hi.trials["score"].max()
    with pytest.raises(ValueError, match="unknown metric"):
        tune("ridge", X, y, metric="r2")


def test_tune_time_budget_returns_best_so_far():
    X, y = make_xy()
    r = tune("ridge", X, y, n_trials=6, time_budget_sec=0)
    assert r.timed_out and len(r.trials) == 1 and r.best_params == {}
    assert np.isfinite(r.best_score)


def test_tune_records_failed_candidates():
    X, y = make_xy()
    r = tune("ridge", X, y, param_grid={"alpha": [-1.0]}, base_params={"alpha": 2.0})
    assert r.trials["error"].iloc[1] is not None and np.isnan(r.trials["score"].iloc[1])
    assert r.best_params == {"alpha": 2.0}


def test_nested_outer_test_rows_never_reach_tuning(monkeypatch):
    X, y = make_xy(700)
    seen: list[pd.Index] = []

    class Recorder(RidgeForecaster):
        def fit(self, X, y):
            seen.append(X.index)
            return super().fit(X, y)

        def predict(self, X):
            seen.append(X.index)
            return super().predict(X)

    monkeypatch.setattr(tuning, "get_model", lambda name, **kw: Recorder(**kw))
    tuned_on: list[tuple[pd.Index, list[pd.Index]]] = []
    real_tune = tuning.tune

    def spy(model_name, X, y, **kw):
        start = len(seen)
        out = real_tune(model_name, X, y, **kw)
        tuned_on.append((X.index, seen[start:]))
        return out

    monkeypatch.setattr(tuning, "tune", spy)
    res = nested_walk_forward("ridge", X, y, outer_splits=3, inner_splits=2, n_trials=3, gap=2)

    assert len(tuned_on) == 3 and len(res.params) == 3
    for fold, (train_index, touched) in enumerate(tuned_on):
        test_index = res.predictions.index[res.predictions["fold"] == fold]
        assert train_index.intersection(test_index).empty
        assert train_index.max() < test_index.min()
        assert touched and all(ix.isin(train_index).all() for ix in touched)
    assert set(res.folds.columns) >= {"params", "inner_score", "n_trials", "mae", "relative_mae"}


def test_nested_reproducible_and_scored_once():
    X, y = make_xy(700)
    kw = dict(outer_splits=3, inner_splits=2, n_trials=4, seed=7)
    a = nested_walk_forward("ridge", X, y, **kw)
    b = nested_walk_forward("ridge", X, y, **kw)
    assert a.params == b.params and a.overall == b.overall
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    assert not a.predictions.index.has_duplicates
    assert a.overall["n"] == len(a.predictions) == 3 * (700 // 4)
    assert a.overall["relative_mae"] < 1.0
