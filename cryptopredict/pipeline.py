"""End-to-end helpers behind the CLI: load data, train, predict, backtest.

Every function takes an already loaded OHLCV frame (see :func:`load_ohlcv`), so
the whole pipeline also runs offline from a CSV file. Not investment advice.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cryptopredict.core.types import DEFAULT_HORIZON, Forecaster
from cryptopredict.evaluation import WalkForwardResult, backtest, walk_forward_evaluate
from cryptopredict.evaluation.backtest import BacktestResult
from cryptopredict.features import (
    FeatureConfig,
    build_features,
    forward_log_return,
    latest_features,
    make_dataset,
)
from cryptopredict.models import default_compare_models, get_model, load_model, save_model

# Rolling mean return feature used by the "ma" baseline when present.
MA_FEATURE = "ret_mean_24"

_INTERVAL_RE = re.compile(r"^(\d+)([smhdwM])$")
_INTERVAL_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
    "M": timedelta(days=30),  # approximate calendar month
}


def interval_to_timedelta(interval: str) -> timedelta:
    """Bar length of a Binance interval string, e.g. ``"1h"`` -> 1 hour."""
    match = _INTERVAL_RE.match(interval or "")
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"invalid interval {interval!r}; expected e.g. 1m, 15m, 1h, 4h, 1d, 1w")
    return int(match.group(1)) * _INTERVAL_UNITS[match.group(2)]


def periods_per_year(interval: str) -> float:
    """Number of bars per (365-day) year, used to annualise the Sharpe ratio."""
    return timedelta(days=365) / interval_to_timedelta(interval)


class OpenBarWarning(UserWarning):
    """Unclosed bars were dropped before predicting."""


def drop_open_bars(
    df: pd.DataFrame,
    interval: str,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, int]:
    """Drop bars that have not closed yet (``open_time + bar > now``).

    Returns the closed bars and the number of dropped rows. ``now`` defaults to
    the current UTC time; a naive ``now`` is taken as UTC.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    closed = df.index + interval_to_timedelta(interval) <= now
    return df.loc[closed], int((~closed).sum())


def load_ohlcv(
    *,
    csv: str | Path | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """OHLCV from ``csv`` (offline) or from Binance through the local cache."""
    from cryptopredict import data

    if csv is not None:
        return data.load_csv(csv)
    if not symbol or not interval or not start:
        raise ValueError("either a CSV path or symbol, interval and start are required")
    return data.get_ohlcv(symbol, interval, start, end, cache_dir=cache_dir)


def make_model(
    name: str,
    feature_columns: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> Forecaster:
    """Unfitted model by registry name; ``ma`` uses :data:`MA_FEATURE` when available."""
    options = dict(params or {})
    if name == "ma" and feature_columns is not None and MA_FEATURE in feature_columns:
        options.setdefault("column", MA_FEATURE)
    return get_model(name, **options)


def _period(index: pd.Index) -> dict[str, str]:
    return {"start": index[0].isoformat(), "end": index[-1].isoformat()}


def train(
    df: pd.DataFrame,
    model_name: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    feature_config: FeatureConfig | None = None,
    model_params: dict[str, Any] | None = None,
) -> tuple[Forecaster, dict[str, Any]]:
    """Fit ``model_name`` on every complete row of ``df``.

    Returns the fitted model and the metadata to store next to it (horizon,
    feature config, training period, sample count).
    """
    cfg = feature_config or FeatureConfig()
    X, y = make_dataset(df, horizon=horizon, config=cfg)
    if X.empty:
        raise ValueError("not enough history to build a training set")
    params = dict(model_params or {})
    model = make_model(model_name, list(X.columns), params=params)
    model.fit(X, y)
    info = {
        "horizon": horizon,
        "feature_columns": list(X.columns),
        "feature_config": cfg.to_dict(),
        "train_period": _period(X.index),
        "n_samples": len(X),
        "model_params": params,
    }
    return model, info


def save_trained(
    model: Forecaster,
    info: dict[str, Any],
    path: str | Path,
    **extra: Any,
) -> dict[str, Any]:
    """Persist a model returned by :func:`train`; ``extra`` (symbol, interval...) is kept."""
    return save_model(
        model,
        path,
        feature_columns=info["feature_columns"],
        horizon=info["horizon"],
        extra={
            "feature_config": info["feature_config"],
            "train_period": info["train_period"],
            "n_samples": info["n_samples"],
            "model_params": info.get("model_params", {}),
            **extra,
        },
    )


def predict_latest(
    model: Forecaster,
    metadata: dict[str, Any],
    df: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    include_open_bar: bool = False,
) -> dict[str, Any]:
    """Forecast from the most recent complete, closed bar of ``df``.

    When ``metadata`` has an ``interval``, bars still open at ``now`` (default:
    current UTC time) are dropped with an :class:`OpenBarWarning`, unless
    ``include_open_bar``. Without an interval every bar is assumed closed.

    Returns the predicted ``horizon``-bar log return, the implied price and the
    direction, plus the timestamps the forecast refers to.
    """
    dropped = 0
    if metadata.get("interval") and not include_open_bar:
        df, dropped = drop_open_bars(df, metadata["interval"], now)
        if dropped:
            warnings.warn(f"ignored {dropped} unclosed bar(s) at the end of the data", OpenBarWarning, stacklevel=2)
        if df.empty:
            raise ValueError("no closed bars to predict from")
    cfg = FeatureConfig.from_dict(metadata.get("feature_config") or {})
    columns = metadata["feature_columns"]
    lookback = int(getattr(model, "lookback", 1))
    if lookback < 1:
        raise ValueError(f"model lookback must be >= 1, got {lookback}")
    # Validates the final row of the model's columns, with a clear error after a candle gap.
    window = latest_features(df, config=cfg, columns=columns)
    if lookback > 1:
        # Sequence models read the last ``lookback`` rows and forecast from the final one.
        window = build_features(df, config=cfg)[columns].tail(lookback)
        if len(window) < lookback or window.isna().any(axis=None):
            raise ValueError(
                f"the model needs {lookback} complete feature rows; more contiguous history is needed"
            )
    predictions = np.asarray(model.predict(window), dtype="float64").ravel()
    if predictions.size == 0 or not np.isfinite(predictions[-1]):
        raise ValueError("model returned no finite forecast for the latest feature row")
    log_return = float(predictions[-1])
    as_of = window.index[-1]
    last_close = float(df.loc[as_of, "close"])
    horizon = int(metadata.get("horizon", DEFAULT_HORIZON))
    result = {
        "model": metadata.get("name"),
        "symbol": metadata.get("symbol"),
        "interval": metadata.get("interval"),
        "horizon": horizon,
        "as_of": as_of.isoformat(),
        "target_time": None,
        "last_close": last_close,
        "predicted_log_return": log_return,
        "predicted_pct_change": math.expm1(log_return) * 100,
        "expected_price": last_close * math.exp(log_return),
        "direction": "up" if log_return > 0 else "down" if log_return < 0 else "flat",
        "dropped_open_bars": dropped,
    }
    if metadata.get("interval"):
        bar = interval_to_timedelta(metadata["interval"])
        # open_time indexes the bar; its close is one bar later, the target h bars after that.
        result["target_time"] = (as_of + (horizon + 1) * bar).isoformat()
    return result


def load_and_predict(model_path: str | Path, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """:func:`predict_latest` for a model file saved by :func:`save_trained`."""
    model, metadata = load_model(model_path)
    return predict_latest(model, metadata, df, **kwargs)


@dataclass
class BacktestReport:
    """Walk-forward evaluation of one model and the trading backtest of its forecasts."""

    model: str
    evaluation: WalkForwardResult
    backtest: BacktestResult

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, **self.evaluation.overall, **self.backtest.stats}


def run_backtest(
    df: pd.DataFrame,
    model_name: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    n_splits: int = 5,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    threshold: float = 0.0,
    da_threshold: float = 0.0,
    allow_short: bool = False,
    interval: str | None = None,
    feature_config: FeatureConfig | None = None,
    model_params: dict[str, Any] | None = None,
    dataset: tuple[pd.DataFrame, pd.Series] | None = None,
) -> BacktestReport:
    """Walk-forward evaluate ``model_name`` and trade the sign of its forecasts.

    Folds are separated by ``gap=horizon`` rows so no training target overlaps
    the test period. Positions are held one bar and earn the forward 1-bar log
    return, whatever the forecast horizon.

    ``threshold`` is the minimum ``|forecast|`` to take a position;
    ``da_threshold`` is the neutral band on ``|y_true|`` excluded from
    ``directional_accuracy``.
    """
    X, y = dataset if dataset is not None else make_dataset(df, horizon=horizon, config=feature_config)
    columns = list(X.columns)
    evaluation = walk_forward_evaluate(
        lambda: make_model(model_name, columns, params=model_params),
        X,
        y,
        n_splits=n_splits,
        gap=horizon,
        threshold=da_threshold,
    )
    preds = evaluation.predictions
    realized = forward_log_return(df["close"], 1).reindex(preds.index)
    if realized.isna().any():
        raise ValueError("missing forward returns for some test rows")
    result = backtest(
        preds["y_pred"],
        realized,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        threshold=threshold,
        allow_short=allow_short,
        periods_per_year=periods_per_year(interval) if interval else 24 * 365,
    )
    return BacktestReport(model_name, evaluation, result)


def compare_models(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, BacktestReport]]:
    """:func:`run_backtest` for several models (default: all) on the same dataset.

    Returns a summary table (one row per model, sorted by ``rmse``) and the reports.
    """
    names = model_names or default_compare_models()
    horizon = kwargs.get("horizon", DEFAULT_HORIZON)
    dataset = make_dataset(df, horizon=horizon, config=kwargs.get("feature_config"))
    reports = {name: run_backtest(df, name, dataset=dataset, **kwargs) for name in names}
    table = pd.DataFrame([r.summary() for r in reports.values()]).set_index("model")
    table = table.sort_values("rmse", kind="stable")
    return table.replace([np.inf, -np.inf], np.nan), reports
