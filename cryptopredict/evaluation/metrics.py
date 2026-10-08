"""Point-forecast metrics for log-return predictions."""

from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true={t.size}, y_pred={p.size}")
    if t.size == 0:
        raise ValueError("empty input")
    return t, p


def mae(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (0-200); pairs where both values are 0 count as 0."""
    t, p = _pair(y_true, y_pred)
    denom = np.abs(t) + np.abs(p)
    ratio = np.divide(2.0 * np.abs(t - p), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(100.0 * np.mean(ratio))


def directional_accuracy(y_true, y_pred, threshold: float = 0.0) -> float:
    """Share of samples where ``sign(y_pred) == sign(y_true)``.

    Samples with ``|y_true| <= threshold`` are neutral (noise) and excluded.
    Returns NaN if no sample is left.
    """
    t, p = _pair(y_true, y_pred)
    mask = np.abs(t) > threshold
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(p[mask]) == np.sign(t[mask])))


def relative_mae(y_true, y_pred, y_baseline=None) -> float:
    """``MAE(model) / MAE(baseline)``; the default baseline predicts a zero return.

    Values below 1 mean the model beats the baseline.
    """
    t, p = _pair(y_true, y_pred)
    b = np.zeros_like(t) if y_baseline is None else _pair(t, y_baseline)[1]
    base = mae(t, b)
    if base == 0:
        return float("nan")
    return mae(t, p) / base


def regression_report(y_true, y_pred, threshold: float = 0.0) -> dict[str, float]:
    """All metrics at once, keyed by name."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred, threshold),
        "relative_mae": relative_mae(y_true, y_pred),
        "n": float(np.asarray(y_true).size),
    }
