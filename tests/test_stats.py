import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cryptopredict.evaluation import diebold_mariano, dm_vs_zero, long_run_variance


def test_one_step_hln_statistic_is_the_paired_t_test():
    rng = np.random.default_rng(1)
    errors_a = rng.normal(0.0, 1.0, 60)
    errors_b = rng.normal(0.0, 1.3, 60)
    differential = errors_a**2 - errors_b**2
    reference = stats.ttest_1samp(differential, 0.0)

    result = diebold_mariano(errors_a, errors_b, horizon=1)

    assert result.statistic == pytest.approx(reference.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-10)
    assert result.mean_loss_diff == pytest.approx(differential.mean())
    assert result.n == result.n_obs == 60
    assert result.max_lag == 0 and result.kernel == "bartlett"


def test_multi_step_statistic_matches_bartlett_hand_computation():
    errors_a = np.array([0.5, -1.0, 0.2, 1.5, -0.3, 0.8, -0.6, 0.1, 1.1, -0.9])
    errors_b = np.array([1.0, -1.2, 0.9, 1.4, -1.1, 0.7, -1.3, 0.6, 1.0, -1.5])
    differential = np.abs(errors_a) - np.abs(errors_b)
    n, horizon = differential.size, 3
    centered = differential - differential.mean()
    gamma = [centered @ centered / n] + [centered[k:] @ centered[:-k] / n for k in (1, 2)]
    variance = gamma[0] + 2 * ((2 / 3) * gamma[1] + (1 / 3) * gamma[2])
    expected = differential.mean() / math.sqrt(variance / n)
    expected *= math.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )

    result = diebold_mariano(
        errors_a, errors_b, horizon=horizon, loss="absolute", alternative="less"
    )

    assert result.statistic == pytest.approx(expected, rel=1e-12)
    assert result.p_value == pytest.approx(stats.t(df=n - 1).cdf(expected), rel=1e-12)
    assert result.statistic < 0


def test_known_better_forecast_has_correct_sign_and_p_value():
    actual = np.array([0.9, -1.1, 0.7, -0.8, 1.3, -1.0, 0.6, -1.4])
    better = actual + np.array([0.05, -0.10, 0.08, -0.04, 0.12, -0.07, 0.03, -0.09])
    worse = actual + np.array([0.80, -0.70, 0.90, -0.60, 0.75, -0.85, 0.65, -0.95])

    result = diebold_mariano(actual - better, actual - worse)

    assert result.statistic == pytest.approx(-9.0911478739)
    assert result.p_value == pytest.approx(0.00003994905817551091)
    assert result.mean_loss_diff < 0


def test_dm_vs_zero_sign_and_one_sided_p_values():
    rng = np.random.default_rng(7)
    actual = rng.normal(0.0, 1.0, 2000)
    good = actual - rng.normal(0.0, 0.5, 2000)
    result = dm_vs_zero(actual, good)
    assert result.statistic < -5
    assert result.p_value < 1e-6
    assert result.alternative == "less"

    two_sided = dm_vs_zero(actual, good, alternative="two-sided")
    greater = dm_vs_zero(actual, good, alternative="greater")
    assert two_sided.p_value == pytest.approx(2 * result.p_value)
    assert greater.p_value == pytest.approx(1 - result.p_value)


def test_equal_forecasts_are_not_different():
    errors = np.linspace(-1.0, 1.0, 50)
    result = diebold_mariano(errors, errors, horizon=2, kernel="uniform")
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.variance == 0.0


def test_size_is_close_to_nominal_under_the_null():
    rng = np.random.default_rng(3)
    rejections = 0
    for _ in range(400):
        actual = rng.normal(size=300)
        first = actual - rng.normal(0.0, 0.3, 300)
        second = actual - rng.normal(0.0, 0.3, 300)
        rejections += diebold_mariano(actual - first, actual - second).p_value < 0.05
    assert 0.02 < rejections / 400 < 0.09


def test_long_run_variance_kernels_and_normal_option():
    differential = np.array([1.0, -1.0] * 20)
    assert long_run_variance(differential, 0) == pytest.approx(1.0)
    assert long_run_variance(differential, 1, "uniform") == pytest.approx(
        long_run_variance(differential, 1, "rectangular")
    )
    assert long_run_variance(differential, 1, "uniform") < long_run_variance(
        differential, 1, "bartlett"
    )

    rng = np.random.default_rng(0)
    actual = rng.normal(size=100)
    result = dm_vs_zero(actual, 0.5 * actual, harvey=False, max_lag=4)
    assert result.max_lag == 4 and not result.harvey
    assert result.p_value == pytest.approx(stats.norm.cdf(result.statistic))


def test_series_indexes_must_match_exactly():
    index = pd.date_range("2025-01-01", periods=20, tz="UTC")
    actual = pd.Series(np.arange(20.0), index=index)
    shifted = pd.Series(np.arange(20.0), index=index.shift(1, freq="h"))
    with pytest.raises(ValueError, match="matching indexes"):
        dm_vs_zero(actual, shifted)
    with pytest.raises(ValueError, match="matching indexes"):
        diebold_mariano(actual, shifted)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0}, "horizon"),
        ({"loss": "hinge"}, "loss"),
        ({"alternative": "bigger"}, "alternative"),
        ({"max_lag": -1}, "max_lag"),
        ({"kernel": "parzen"}, "kernel"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, match):
    errors = np.arange(10.0)
    with pytest.raises(ValueError, match=match):
        diebold_mariano(errors, errors[::-1], **kwargs)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="length"):
        diebold_mariano(np.ones(5), np.ones(6))
    with pytest.raises(ValueError, match="finite"):
        diebold_mariano([1.0, np.nan, 1.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="at least"):
        diebold_mariano(np.ones(4), np.zeros(4), horizon=2)


def test_nonpositive_nonconstant_hac_variance_is_rejected():
    first = np.array([0.0, 1.0] * 10)
    second = np.array([1.0, 0.0] * 10)
    with pytest.raises(ValueError, match="non-positive HAC variance"):
        diebold_mariano(first, second, horizon=2, loss="absolute", kernel="uniform")
