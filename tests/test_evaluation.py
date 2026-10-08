import numpy as np
import pandas as pd
import pytest

from cryptopredict.evaluation import (
    backtest,
    directional_accuracy,
    mae,
    regression_report,
    relative_mae,
    rmse,
    smape,
    walk_forward_evaluate,
    walk_forward_splits,
)


class MeanForecaster:
    """Stub: predicts the training mean; records what it was fitted on."""

    name = "mean"
    fitted_on: list[pd.Index] = []

    def fit(self, X, y):
        self.mean_ = float(y.mean())
        MeanForecaster.fitted_on.append(X.index)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class LookupForecaster:
    """Stub: predicts the value of column ``lookup`` (an oracle when it equals y)."""

    name = "lookup"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lookup"].to_numpy()


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC", name="open_time")
    y = pd.Series(rng.normal(0, 0.01, n), index=idx, name="target")
    X = pd.DataFrame({"log_ret_1": rng.normal(0, 0.01, n), "lookup": y.to_numpy()}, index=idx)
    return X, y


# --- splits -----------------------------------------------------------------

@pytest.mark.parametrize("expanding", [True, False])
@pytest.mark.parametrize("gap", [0, 1, 3])
def test_splits_train_precedes_test_and_no_overlap(expanding, gap):
    splits = list(walk_forward_splits(100, n_splits=4, expanding=expanding, gap=gap))
    assert len(splits) == 4
    seen_test = set()
    for train, test in splits:
        assert len(train) > 0 and len(test) > 0
        assert not set(train) & set(test)
        assert train.max() < test.min()
        assert test.min() - train.max() - 1 == gap
        assert not seen_test & set(test)
        seen_test |= set(test)
    assert splits[-1][1][-1] == 99  # test blocks reach the end of the series


def test_splits_expanding_vs_rolling_sizes():
    exp = list(walk_forward_splits(100, n_splits=3, train_size=40, test_size=20))
    assert [len(tr) for tr, _ in exp] == [40, 60, 80]
    assert all(tr[0] == 0 for tr, _ in exp)
    roll = list(walk_forward_splits(100, n_splits=3, train_size=40, test_size=20, expanding=False))
    assert [len(tr) for tr, _ in roll] == [40, 40, 40]
    assert [tr[0] for tr, _ in roll] == [0, 20, 40]


def test_splits_default_sizes_and_errors():
    splits = list(walk_forward_splits(60, n_splits=5))
    assert [len(te) for _, te in splits] == [10] * 5
    assert len(splits[0][0]) == 10
    with pytest.raises(ValueError):
        list(walk_forward_splits(10, n_splits=3, train_size=9, test_size=2))
    with pytest.raises(ValueError):
        list(walk_forward_splits(3, n_splits=5))
    with pytest.raises(ValueError):
        list(walk_forward_splits(100, n_splits=0))


# --- metrics ----------------------------------------------------------------

def test_metrics_known_values():
    t = [1.0, -2.0, 3.0, 0.0]
    p = [2.0, -2.0, 1.0, 0.0]
    assert mae(t, p) == pytest.approx(0.75)
    assert rmse(t, p) == pytest.approx(np.sqrt(5 / 4))
    # sMAPE terms: 2/3, 0, 1, 0 (0/0 -> 0) -> mean 5/12
    assert smape(t, p) == pytest.approx(100 * 5 / 12)
    assert relative_mae(t, p) == pytest.approx(0.75 / 1.5)
    assert relative_mae(t, t) == 0.0


def test_directional_accuracy_with_neutral_threshold():
    t = [0.02, -0.01, 0.001, -0.03]
    p = [0.01, 0.01, -0.5, -0.01]
    assert directional_accuracy(t, p) == pytest.approx(0.5)
    # |0.001| is neutral at threshold 0.005 -> 2 of 3 remaining are right
    assert directional_accuracy(t, p, threshold=0.005) == pytest.approx(2 / 3)
    assert np.isnan(directional_accuracy(t, p, threshold=1.0))


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    with pytest.raises(ValueError):
        rmse([], [])


def test_regression_report_keys():
    rep = regression_report([0.1, -0.1], [0.1, 0.1])
    assert set(rep) == {"mae", "rmse", "smape", "directional_accuracy", "relative_mae", "n"}
    assert rep["n"] == 2


# --- walk-forward -------------------------------------------------------------

def test_walk_forward_fits_only_on_past_data():
    X, y = _frame(120)
    MeanForecaster.fitted_on = []
    res = walk_forward_evaluate(MeanForecaster, X, y, n_splits=4, gap=1)
    assert len(res.folds) == 4 and len(MeanForecaster.fitted_on) == 4
    for train_index, (_, row) in zip(MeanForecaster.fitted_on, res.folds.iterrows()):
        assert train_index.max() < row["test_start"]
        assert train_index.max() == row["train_end"]
    pred = res.predictions
    assert list(pred.columns) == ["y_true", "y_pred", "fold"]
    assert pred.index.is_monotonic_increasing and not pred.index.has_duplicates
    assert len(pred) == 4 * (119 // 5)
    pd.testing.assert_series_equal(pred["y_true"], y.loc[pred.index], check_names=False)
    # each fold predicts the mean of its own training window
    first = res.folds.iloc[0]
    expected = y.loc[first["train_start"]:first["train_end"]].mean()
    assert pred.loc[pred["fold"] == 0, "y_pred"].iloc[0] == pytest.approx(expected)


def test_walk_forward_oracle_and_custom_splits():
    X, y = _frame(100)
    splits = [(np.arange(0, 50), np.arange(50, 75)), (np.arange(0, 75), np.arange(75, 100))]
    res = walk_forward_evaluate(LookupForecaster, X, y, splits=splits)
    assert res.overall["mae"] == 0.0
    assert res.overall["directional_accuracy"] == 1.0
    assert res.overall["relative_mae"] == 0.0
    assert res.overall["n"] == 50


def test_walk_forward_accepts_sklearn_estimator_and_validates():
    from sklearn.linear_model import Ridge

    X, y = _frame(100)
    res = walk_forward_evaluate(Ridge(alpha=1.0), X, y, n_splits=3)
    assert np.isfinite(res.overall["rmse"])
    with pytest.raises(ValueError):
        walk_forward_evaluate(MeanForecaster, X, y.iloc[:-1])
    y_nan = y.copy()
    y_nan.iloc[-1] = np.nan
    with pytest.raises(ValueError):
        walk_forward_evaluate(MeanForecaster, X, y_nan)
    with pytest.raises(ValueError):
        walk_forward_evaluate(MeanForecaster, X, y, splits=[(np.arange(10, 20), np.arange(0, 10))])


def test_walk_forward_on_fixture(ohlcv):
    close = ohlcv["close"]
    X = pd.DataFrame({"log_ret_1": np.log(close / close.shift(1))})
    y = np.log(close.shift(-1) / close).rename("target")
    keep = X["log_ret_1"].notna() & y.notna()
    res = walk_forward_evaluate(MeanForecaster, X[keep], y[keep], n_splits=5, gap=1)
    assert len(res.folds) == 5
    assert 0 <= res.overall["directional_accuracy"] <= 1


# --- backtest -----------------------------------------------------------------

def test_backtest_known_values():
    r = np.log([1.10, 0.90, 1.05])
    res = backtest([1.0, -1.0, 1.0], r, fee_bps=0)
    assert list(res.frame["position"]) == [1.0, 0.0, 1.0]
    assert res.stats["total_return"] == pytest.approx(1.10 * 1.05 - 1)
    assert res.stats["bh_total_return"] == pytest.approx(1.10 * 0.90 * 1.05 - 1)
    assert res.stats["n_trades"] == 3
    assert res.stats["exposure"] == pytest.approx(2 / 3)
    short = backtest([1.0, -1.0, 1.0], r, fee_bps=0, allow_short=True)
    assert list(short.frame["position"]) == [1.0, -1.0, 1.0]
    assert short.stats["total_return"] == pytest.approx(1.10 * 1.10 * 1.05 - 1)


def test_backtest_fees_reduce_return():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 300)
    p = rng.normal(0, 0.01, 300)
    free = backtest(p, r, fee_bps=0, slippage_bps=0)
    paid = backtest(p, r, fee_bps=10, slippage_bps=5)
    assert paid.stats["n_trades"] == free.stats["n_trades"] > 0
    assert paid.stats["total_return"] < free.stats["total_return"]
    assert paid.stats["total_cost"] == pytest.approx(paid.stats["n_trades"] * 15 / 1e4)
    assert paid.stats["bh_total_return"] < free.stats["bh_total_return"]


def test_perfect_forecast_beats_buy_and_hold():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 500)
    res = backtest(r, r, fee_bps=10)
    assert res.stats["total_return"] > res.stats["bh_total_return"]
    assert res.stats["sharpe"] > res.stats["bh_sharpe"]
    assert res.stats["max_drawdown"] <= res.stats["bh_max_drawdown"]


def test_backtest_has_no_look_ahead():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 100)
    p = rng.normal(0, 0.01, 100)
    base = backtest(p, r, fee_bps=10, allow_short=True).frame["equity"]
    p2, r2 = p.copy(), r.copy()
    p2[60:] = -p2[60:]
    r2[60:] = rng.normal(0, 0.05, 40)
    changed = backtest(p2, r2, fee_bps=10, allow_short=True).frame["equity"]
    np.testing.assert_allclose(base[:60], changed[:60])
    # A "forecast" that is just the last observed return (known at t) is not an oracle.
    lagged = np.concatenate([[0.0], r[:-1]])
    assert backtest(lagged, r, fee_bps=0).stats["total_return"] < backtest(r, r, fee_bps=0).stats["total_return"]


def test_backtest_drawdown_sharpe_and_validation():
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC", name="open_time")
    r = pd.Series(np.log([1.10, 0.50, 1.20, 1.00]), index=idx)
    res = backtest(pd.Series(1.0, index=idx), r, fee_bps=0)
    assert res.frame.index.equals(idx)
    assert res.stats["max_drawdown"] == pytest.approx(0.5)
    assert res.stats["sharpe"] == pytest.approx(res.stats["bh_sharpe"])
    flat = backtest(np.full(4, -1.0), r.to_numpy(), fee_bps=10)
    assert flat.stats["total_return"] == 0.0 and flat.stats["n_trades"] == 0
    assert np.isnan(flat.stats["sharpe"])
    with pytest.raises(ValueError):
        backtest([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        backtest([0.1, np.nan], [0.1, 0.2])
    with pytest.raises(ValueError):
        backtest(pd.Series([0.1, 0.2], index=idx[:2]), pd.Series([0.1, 0.2], index=idx[2:]))
