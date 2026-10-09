import numpy as np
import pandas as pd
import pytest

from cryptopredict.evaluation import diebold_mariano


def test_diebold_mariano_known_sign_and_p_value():
    actual = np.array([0.9, -1.1, 0.7, -0.8, 1.3, -1.0, 0.6, -1.4])
    better = actual + np.array([0.05, -0.10, 0.08, -0.04, 0.12, -0.07, 0.03, -0.09])
    worse = actual + np.array([0.80, -0.70, 0.90, -0.60, 0.75, -0.85, 0.65, -0.95])

    result = diebold_mariano(actual, better, worse, loss="squared")

    assert result.statistic == pytest.approx(-9.0911478739)
    assert result.p_value == pytest.approx(0.00003994905817551091)
    assert result.mean_loss_diff < 0
    assert result.n_obs == len(actual)


def test_diebold_mariano_is_symmetric_and_zero_defaults_to_baseline():
    actual = np.array([0.4, -0.2, 0.6, -0.7, 0.3, -0.5])
    first = np.array([0.3, -0.1, 0.4, -0.6, 0.2, -0.3])
    second = np.zeros_like(actual)

    ab = diebold_mariano(actual, first, second)
    ba = diebold_mariano(actual, second, first)
    default = diebold_mariano(actual, first)

    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.p_value == pytest.approx(ba.p_value)
    assert default == ab


def test_diebold_mariano_identical_losses_are_no_difference():
    actual = np.arange(10, dtype=float)
    result = diebold_mariano(actual, actual, actual, horizon=2, kernel="uniform")
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.variance == 0.0


def test_diebold_mariano_horizon_uses_hac_and_hln():
    actual = np.array([0.2, -0.1, 0.5, -0.4, 0.7, -0.2, 0.1, -0.6, 0.8, -0.3])
    first = actual + np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1])
    second = np.zeros_like(actual)
    one = diebold_mariano(actual, first, second, horizon=1)
    three = diebold_mariano(actual, first, second, horizon=3)
    assert three.horizon == 3
    assert three.variance != pytest.approx(one.variance)
    assert np.isfinite(three.statistic)


def test_diebold_mariano_requires_paired_finite_data():
    index = pd.date_range("2025-01-01", periods=5, tz="UTC")
    actual = pd.Series(np.arange(5.0), index=index)
    shifted = pd.Series(np.arange(5.0), index=index.shift(1, freq="h"))
    with pytest.raises(ValueError, match="matching indexes"):
        diebold_mariano(actual, actual, shifted)
    with pytest.raises(ValueError, match="finite"):
        diebold_mariano([1.0, np.nan, 2.0], [1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="more observations"):
        diebold_mariano([1.0, 2.0], [1.0, 2.0], horizon=2)


def test_diebold_mariano_rejects_nonpositive_nonconstant_hac_variance():
    actual = np.ones(6)
    first = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    second = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="non-positive HAC variance"):
        diebold_mariano(actual, first, second, horizon=2, kernel="uniform")
