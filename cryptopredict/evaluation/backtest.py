"""Vectorised long/flat(/short) backtest of a log-return forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Outcome of :func:`backtest`.

    frame: per bar ``prediction``, ``realized``, ``position``, ``turnover``,
           ``cost``, ``net_return`` and ``equity`` (strategy), ``bh_net_return``
           and ``bh_equity`` (buy-and-hold).
    stats: summary numbers, see :func:`backtest`.
    """

    frame: pd.DataFrame
    stats: dict[str, float]


def _as_series(values, name: str, index=None) -> pd.Series:
    if isinstance(values, pd.Series):
        s = values.astype("float64")
    else:
        s = pd.Series(np.asarray(values, dtype="float64").ravel(), index=index)
    return s.rename(name)


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough loss as a positive fraction (starting capital 1.0 counts as a peak)."""
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity.to_numpy()]))[1:]
    return float(np.max(1.0 - equity.to_numpy() / peak, initial=0.0))


def _sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if len(returns) < 2:
        return float("nan")
    sd = returns.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def backtest(
    predictions,
    realized_log_returns,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    threshold: float = 0.0,
    allow_short: bool = False,
    periods_per_year: float = 24 * 365,
) -> BacktestResult:
    """Trade the sign of a one-step-ahead log-return forecast.

    Timing (no look-ahead): ``predictions[t]`` is made with data up to the close of
    bar ``t``; the resulting position is held from close ``t`` to close ``t+1`` and
    earns ``realized_log_returns[t] = log(close[t+1] / close[t])`` -- i.e. the
    ``target`` column with horizon 1, *not* the backward ``log_ret_1``.

    Position: +1 if prediction > threshold, -1 if prediction < -threshold and
    ``allow_short``, else 0 (flat). Every change of position costs
    ``|delta position| * (fee_bps + slippage_bps) / 1e4`` of equity; entering from
    flat at the first bar counts as a change. Buy-and-hold pays one entry cost.

    ``periods_per_year`` annualises the Sharpe ratio (default: hourly bars).

    stats: ``total_return``, ``bh_total_return``, ``excess_return``, ``sharpe``,
    ``bh_sharpe``, ``max_drawdown``, ``bh_max_drawdown`` (positive fractions),
    ``n_trades`` (number of position changes), ``exposure`` (share of bars in the
    market), ``total_cost`` and ``n_bars``.
    """
    pred = _as_series(predictions, "prediction")
    real = _as_series(realized_log_returns, "realized", index=pred.index)
    if len(pred) != len(real):
        raise ValueError(f"length mismatch: predictions={len(pred)}, realized={len(real)}")
    if not pred.index.equals(real.index):
        raise ValueError("predictions and realized_log_returns must share the same index")
    if len(pred) == 0:
        raise ValueError("empty input")
    if pred.isna().any() or real.isna().any():
        raise ValueError("inputs must not contain NaN")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    cost_rate = (fee_bps + slippage_bps) / 1e4
    position = pd.Series(np.where(pred > threshold, 1.0, 0.0), index=pred.index)
    if allow_short:
        position[pred < -threshold] = -1.0
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost = turnover * cost_rate
    simple = np.expm1(real)
    net = position * simple - cost
    equity = (1.0 + net).cumprod()

    bh_net = simple.copy()
    bh_net.iloc[0] -= cost_rate
    bh_equity = (1.0 + bh_net).cumprod()

    frame = pd.DataFrame({
        "prediction": pred,
        "realized": real,
        "position": position,
        "turnover": turnover,
        "cost": cost,
        "net_return": net,
        "equity": equity,
        "bh_net_return": bh_net,
        "bh_equity": bh_equity,
    })
    total = float(equity.iloc[-1] - 1.0)
    bh_total = float(bh_equity.iloc[-1] - 1.0)
    stats = {
        "total_return": total,
        "bh_total_return": bh_total,
        "excess_return": total - bh_total,
        "sharpe": _sharpe(net, periods_per_year),
        "bh_sharpe": _sharpe(bh_net, periods_per_year),
        "max_drawdown": _max_drawdown(equity),
        "bh_max_drawdown": _max_drawdown(bh_equity),
        "n_trades": int((turnover > 0).sum()),
        "exposure": float((position != 0).mean()),
        "total_cost": float(cost.sum()),
        "n_bars": int(len(frame)),
    }
    return BacktestResult(frame=frame, stats=stats)
