"""Statistical comparisons for paired out-of-sample forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

Loss = Literal["absolute", "squared"]
Kernel = Literal["bartlett", "uniform"]
Alternative = Literal["two-sided", "less", "greater"]


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult:
    """Result of a paired Diebold-Mariano forecast comparison.

    ``mean_loss_diff`` is ``loss(forecast_a) - loss(forecast_b)``. Therefore a
    negative statistic means that forecast A has the smaller average loss.
    """

    statistic: float
    p_value: float
    mean_loss_diff: float
    variance: float
    n_obs: int
    horizon: int
    loss: Loss
    kernel: Kernel
    alternative: Alternative


def _as_paired_arrays(y_true, forecast_a, forecast_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = (y_true, forecast_a, forecast_b)
    series = [value for value in values if isinstance(value, pd.Series)]
    if series:
        index = series[0].index
        if any(not value.index.equals(index) for value in series[1:]):
            raise ValueError("paired Series must have exactly matching indexes")

    arrays = tuple(np.asarray(value, dtype="float64").ravel() for value in values)
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "length mismatch: "
            f"y_true={arrays[0].size}, forecast_a={arrays[1].size}, "
            f"forecast_b={arrays[2].size}"
        )
    if not arrays[0].size:
        raise ValueError("forecasts must not be empty")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("forecasts and observations must contain only finite values")
    return arrays


def _loss(error: np.ndarray, kind: Loss) -> np.ndarray:
    if kind == "absolute":
        return np.abs(error)
    if kind == "squared":
        return np.square(error)
    raise ValueError("loss must be 'absolute' or 'squared'")


def _long_run_variance(values: np.ndarray, lag: int, kernel: Kernel) -> float:
    centered = values - values.mean()
    n_obs = values.size
    variance = float(centered @ centered / n_obs)
    if kernel not in {"bartlett", "uniform"}:
        raise ValueError("kernel must be 'bartlett' or 'uniform'")
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / n_obs)
        weight = 1.0 - offset / (lag + 1.0) if kernel == "bartlett" else 1.0
        variance += 2.0 * weight * covariance
    return variance


def diebold_mariano(
    y_true,
    forecast_a,
    forecast_b=None,
    *,
    horizon: int = 1,
    loss: Loss = "absolute",
    kernel: Kernel = "bartlett",
    alternative: Alternative = "two-sided",
    small_sample: bool = True,
) -> DieboldMarianoResult:
    """Compare two forecasts with a HAC Diebold-Mariano test.

    Forecast B defaults to the zero-return forecast. Loss differential is
    ``L(A) - L(B)``; negative values favour A. Autocovariances through
    ``horizon - 1`` are included, with a positive-semidefinite Bartlett
    (Newey-West) kernel by default. ``kernel="uniform"`` selects the original
    unweighted DM estimator.

    With ``small_sample=True``, the Harvey-Leybourne-Newbold correction is
    applied and the p-value uses Student's t with ``n - 1`` degrees of freedom.
    Identical loss series return statistic 0 and p-value 1. Other non-positive
    HAC variance estimates raise ``ValueError``.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    if forecast_b is None:
        forecast_b = np.zeros_like(np.asarray(forecast_a, dtype="float64"))
    actual, first, second = _as_paired_arrays(y_true, forecast_a, forecast_b)
    n_obs = actual.size
    if n_obs <= horizon:
        raise ValueError(f"need more observations than horizon; got n={n_obs}, horizon={horizon}")

    differential = _loss(actual - first, loss) - _loss(actual - second, loss)
    mean = float(differential.mean())
    if np.allclose(differential, 0.0, rtol=0.0, atol=np.finfo("float64").eps):
        return DieboldMarianoResult(
            0.0, 1.0, 0.0, 0.0, n_obs, int(horizon), loss, kernel, alternative
        )

    long_run_variance = _long_run_variance(differential, int(horizon) - 1, kernel)
    tolerance = np.finfo("float64").eps * max(1.0, float(np.mean(differential**2)))
    if not np.isfinite(long_run_variance) or long_run_variance <= tolerance:
        raise ValueError(
            "loss differential has a non-positive HAC variance; "
            "the Diebold-Mariano statistic is undefined"
        )

    statistic = mean / np.sqrt(long_run_variance / n_obs)
    if small_sample:
        correction_sq = (
            n_obs + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n_obs
        ) / n_obs
        if correction_sq <= 0.0:
            raise ValueError(f"HLN correction is undefined for n={n_obs}, horizon={horizon}")
        statistic *= np.sqrt(correction_sq)

    cdf = float(student_t.cdf(statistic, df=n_obs - 1))
    if alternative == "less":
        p_value = cdf
    elif alternative == "greater":
        p_value = 1.0 - cdf
    else:
        p_value = 2.0 * min(cdf, 1.0 - cdf)
    return DieboldMarianoResult(
        statistic=float(statistic),
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        mean_loss_diff=mean,
        variance=float(long_run_variance / n_obs),
        n_obs=n_obs,
        horizon=int(horizon),
        loss=loss,
        kernel=kernel,
        alternative=alternative,
    )


def diebold_mariano_test(*args, **kwargs) -> DieboldMarianoResult:
    """Explicit alias for :func:`diebold_mariano`."""
    return diebold_mariano(*args, **kwargs)
