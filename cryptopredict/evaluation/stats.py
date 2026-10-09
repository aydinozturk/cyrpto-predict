"""Forecast-comparison tests: Diebold-Mariano with HAC variance and HLN correction.

Convention: ``d[t] = loss(errors_a[t]) - loss(errors_b[t])``. A negative
statistic means forecast A has the smaller loss. ``dm_vs_zero`` converts paired
observations and forecasts to errors against the zero-return baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

LossName = Literal["squared", "absolute"]
Kernel = Literal["bartlett", "uniform", "rectangular"]
Alternative = Literal["two-sided", "less", "greater"]

_LOSSES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "squared": np.square,
    "absolute": np.abs,
}


@dataclass(frozen=True, slots=True)
class DieboldMarianoResult:
    """Outcome of :func:`diebold_mariano`."""

    statistic: float
    p_value: float
    mean_loss_diff: float
    variance: float
    n: int
    horizon: int
    max_lag: int
    loss: str
    kernel: str
    alternative: str
    harvey: bool

    @property
    def n_obs(self) -> int:
        """Number of paired observations (descriptive alias for ``n``)."""
        return self.n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype="float64").ravel()
    if not array.size:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def long_run_variance(d, max_lag: int, kernel: Kernel = "bartlett") -> float:
    """HAC long-run variance of the mean-zero version of ``d``.

    ``bartlett`` is the positive-semidefinite Newey-West estimator. ``uniform``
    (also accepted as ``rectangular``) is the original unweighted DM estimator
    and can be non-positive in finite samples.
    """
    values = _as_array(d, "loss differential")
    if isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError(f"max_lag must be a non-negative integer, got {max_lag!r}")
    if kernel == "rectangular":
        kernel = "uniform"
    if kernel not in {"bartlett", "uniform"}:
        raise ValueError("kernel must be 'bartlett', 'uniform' or 'rectangular'")
    centered = values - values.mean()
    n = values.size
    variance = float(centered @ centered / n)
    for lag in range(1, min(int(max_lag), n - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0) if kernel == "bartlett" else 1.0
        variance += 2.0 * weight * float(centered[lag:] @ centered[:-lag] / n)
    return variance


def diebold_mariano(
    errors_a,
    errors_b,
    horizon: int = 1,
    loss: LossName | Callable[[np.ndarray], np.ndarray] = "squared",
    alternative: Alternative = "two-sided",
    max_lag: int | None = None,
    kernel: Kernel = "bartlett",
    harvey: bool = True,
) -> DieboldMarianoResult:
    """Test equal predictive accuracy for two paired forecast-error series.

    ``horizon``-step losses use HAC lag ``horizon - 1`` by default. The
    Bartlett/Newey-West kernel is the safe default; ``kernel="uniform"`` chooses
    the original DM estimator. With ``harvey=True`` the
    Harvey-Leybourne-Newbold small-sample correction and Student-t(n-1)
    reference distribution are used. Otherwise the reference is normal.

    The HLN factor was derived for the uniform kernel with ``horizon - 1``
    lags; combined with the Bartlett default it is the usual small-sample
    approximation rather than an exact correction.

    The test is scale-free: rescaling both error series gives the same
    statistic. Equal loss series return statistic 0 and p-value 1. A different
    loss series whose HAC variance is non-positive relative to ``mean(d**2)``
    is undefined and raises ``ValueError``.
    """
    if isinstance(errors_a, pd.Series) and isinstance(errors_b, pd.Series):
        if not errors_a.index.equals(errors_b.index):
            raise ValueError("paired error Series must have exactly matching indexes")
    a = _as_array(errors_a, "errors_a")
    b = _as_array(errors_b, "errors_b")
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: errors_a={a.size}, errors_b={b.size}")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    n = a.size
    if n < 2 * horizon + 1:
        raise ValueError(f"need at least {2 * horizon + 1} errors for horizon {horizon}, got {n}")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less' or 'greater'")
    if callable(loss):
        loss_fn, loss_name = loss, getattr(loss, "__name__", "custom")
    elif loss in _LOSSES:
        loss_fn, loss_name = _LOSSES[loss], loss
    else:
        raise ValueError("loss must be 'squared', 'absolute' or a callable")
    if max_lag is None:
        lags = int(horizon) - 1
    elif isinstance(max_lag, bool) or not isinstance(max_lag, (int, np.integer)) or max_lag < 0:
        raise ValueError(f"max_lag must be a non-negative integer, got {max_lag!r}")
    else:
        lags = int(max_lag)

    differential = np.asarray(loss_fn(a), dtype="float64") - np.asarray(loss_fn(b), dtype="float64")
    if differential.shape != a.shape or not np.isfinite(differential).all():
        raise ValueError("loss must return one finite value per error")
    mean = float(differential.mean())
    normalized_kernel = "uniform" if kernel == "rectangular" else kernel
    # Work on d / max|d| so the tolerance is relative and tiny losses cannot underflow.
    scale = float(np.max(np.abs(differential)))
    if scale == 0.0:
        statistic, p_value, variance = 0.0, 1.0, 0.0
    else:
        unit = differential / scale
        lrv = long_run_variance(unit, lags, kernel)
        tolerance = 64.0 * np.finfo("float64").eps * float(np.mean(unit**2))
        if not np.isfinite(lrv) or lrv <= tolerance:
            raise ValueError(
                "loss differential has a non-positive HAC variance; "
                "the Diebold-Mariano statistic is undefined"
            )
        variance = lrv / n * scale * scale
        statistic = float(unit.mean()) / math.sqrt(lrv / n)
        if harvey:
            correction_sq = (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
            if correction_sq <= 0.0:
                raise ValueError(f"HLN correction is undefined for n={n}, horizon={horizon}")
            statistic *= math.sqrt(correction_sq)
        distribution = scipy_stats.t(df=n - 1) if harvey else scipy_stats.norm()
        if alternative == "two-sided":
            p_value = 2.0 * float(distribution.sf(abs(statistic)))
        elif alternative == "less":
            p_value = float(distribution.cdf(statistic))
        else:
            p_value = float(distribution.sf(statistic))

    return DieboldMarianoResult(
        statistic=float(statistic),
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        mean_loss_diff=mean,
        variance=float(variance),
        n=n,
        horizon=int(horizon),
        max_lag=lags,
        loss=loss_name,
        kernel=normalized_kernel,
        alternative=alternative,
        harvey=harvey,
    )


def dm_vs_zero(
    y_true,
    y_pred,
    horizon: int = 1,
    loss: LossName = "squared",
    alternative: Alternative = "less",
    **kwargs: Any,
) -> DieboldMarianoResult:
    """Compare a forecast with zero return on exactly paired observations."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("paired Series must have exactly matching indexes")
    actual = _as_array(y_true, "y_true")
    predicted = _as_array(y_pred, "y_pred")
    if actual.shape != predicted.shape:
        raise ValueError(f"length mismatch: y_true={actual.size}, y_pred={predicted.size}")
    return diebold_mariano(
        actual - predicted,
        actual,
        horizon=horizon,
        loss=loss,
        alternative=alternative,
        **kwargs,
    )


def diebold_mariano_test(*args, **kwargs) -> DieboldMarianoResult:
    """Explicit alias for :func:`diebold_mariano`."""
    return diebold_mariano(*args, **kwargs)
