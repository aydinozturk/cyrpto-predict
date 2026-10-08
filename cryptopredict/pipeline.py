"""End-to-end helpers behind the CLI: load data, train, predict, backtest.

Every function takes an already loaded OHLCV frame (see :func:`load_ohlcv`), so
the whole pipeline also runs offline from a CSV file. Not investment advice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cryptopredict.core.types import DEFAULT_HORIZON, Forecaster
from cryptopredict.evaluation import WalkForwardResult, backtest, walk_forward_evaluate
from cryptopredict.evaluation.backtest import BacktestResult
from cryptopredict.features import FeatureConfig, forward_log_return, latest_features, make_dataset
from cryptopredict.models import available_models, get_model, load_model, save_model

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


def make_model(name: str, feature_columns: list[str] | None = None) -> Forecaster:
    """Unfitted model by registry name; ``ma`` uses :data:`MA_FEATURE` when available."""
    if name == "ma" and feature_columns is not None and MA_FEATURE in feature_columns:
        return get_model("ma", column=MA_FEATURE)
    return get_model(name)


def _period(index: pd.Index) -> dict[str, str]:
    return {"start": index[0].isoformat(), "end": index[-1].isoformat()}


def train(
    df: pd.DataFrame,
    model_name: str,
    *,
    horizon: int = DEFAULT_HORIZON,
    feature_config: FeatureConfig | None = None,
) -> tuple[Forecaster, dict[str, Any]]:
    """Fit ``model_name`` on every complete row of ``df``.

    Returns the fitted model and the metadata to store next to it (horizon,
    feature config, training period, sample count).
    """
    cfg = feature_config or FeatureConfig()
    X, y = make_dataset(df, horizon=horizon, config=cfg)
    if X.empty:
        raise ValueError("not enough history to build a training set")
    model = make_model(model_name, list(X.columns))
    model.fit(X, y)
    info = {
        "horizon": horizon,
        "feature_columns": list(X.columns),
        "feature_config": cfg.to_dict(),
        "train_period": _period(X.index),
        "n_samples": len(X),
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
            **extra,
        },
    )


def predict_latest(model: Forecaster, metadata: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Forecast from the most recent complete bar of ``df`` (assumed closed).

    Returns the predicted ``horizon``-bar log return, the implied price and the
    direction, plus the timestamps the forecast refers to.
    """
    cfg = FeatureConfig.from_dict(metadata.get("feature_config") or {})
    row = latest_features(df, config=cfg)
    columns = metadata["feature_columns"]
    missing = [c for c in columns if c not in row.columns]
    if missing:
        raise ValueError(f"data lacks feature column(s) the model was trained on: {missing}")
    log_return = float(model.predict(row[columns])[0])
    as_of = row.index[0]
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
    }
    if metadata.get("interval"):
        bar = interval_to_timedelta(metadata["interval"])
        # open_time indexes the bar; its close is one bar later, the target h bars after that.
        result["target_time"] = (as_of + (horizon + 1) * bar).isoformat()
    return result


def load_and_predict(model_path: str | Path, df: pd.DataFrame) -> dict[str, Any]:
    """:func:`predict_latest` for a model file saved by :func:`save_trained`."""
    model, metadata = load_model(model_path)
    return predict_latest(model, metadata, df)


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
    allow_short: bool = False,
    interval: str | None = None,
    feature_config: FeatureConfig | None = None,
    dataset: tuple[pd.DataFrame, pd.Series] | None = None,
) -> BacktestReport:
    """Walk-forward evaluate ``model_name`` and trade the sign of its forecasts.

    Folds are separated by ``gap=horizon`` rows so no training target overlaps
    the test period. Positions are held one bar and earn the forward 1-bar log
    return, whatever the forecast horizon.
    """
    X, y = dataset if dataset is not None else make_dataset(df, horizon=horizon, config=feature_config)
    columns = list(X.columns)
    evaluation = walk_forward_evaluate(
        lambda: make_model(model_name, columns),
        X,
        y,
        n_splits=n_splits,
        gap=horizon,
        threshold=threshold,
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
    names = model_names or available_models()
    horizon = kwargs.get("horizon", DEFAULT_HORIZON)
    dataset = make_dataset(df, horizon=horizon, config=kwargs.get("feature_config"))
    reports = {name: run_backtest(df, name, dataset=dataset, **kwargs) for name in names}
    table = pd.DataFrame([r.summary() for r in reports.values()]).set_index("model")
    table = table.sort_values("rmse", kind="stable")
    return table.replace([np.inf, -np.inf], np.nan), reports
