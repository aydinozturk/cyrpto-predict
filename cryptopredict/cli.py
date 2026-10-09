"""Command line interface: ``cryptopredict {fetch,train,predict,backtest,tune}``.

Run ``cryptopredict <command> --help`` for the options. ``--csv`` makes every
command work offline. Forecasts are not investment advice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from cryptopredict import __version__, pipeline, tuning
from cryptopredict.core.types import DEFAULT_HORIZON
from cryptopredict.features import FeatureConfig, make_dataset, required_history
from cryptopredict.models import available_models, default_compare_models, load_model

DISCLAIMER = "Not investment advice / Yatırım tavsiyesi değildir."
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_CACHE_DIR = "data"
# Minimum bars of history fetched by ``predict`` when no CSV is given.
PREDICT_LOOKBACK_BARS = 500
# Extra bars on top of the feature warm-up and the model lookback (open bar, missing candles).
PREDICT_MARGIN_BARS = 50
# Columns of the `backtest --model all` text table (--json prints every metric).
COMPARISON_COLUMNS = [
    "mae", "rmse", "directional_accuracy", "relative_mae",
    "total_return", "excess_return", "sharpe", "max_drawdown", "n_trades",
]


def _add_data_args(parser: argparse.ArgumentParser, *, default_interval: str | None = "1h") -> None:
    group = parser.add_argument_group("data (Binance or --csv)")
    group.add_argument("--csv", type=Path, help="read OHLCV from this CSV instead of the network")
    group.add_argument("--symbol", help=f"trading pair (default: {DEFAULT_SYMBOL})")
    group.add_argument("--interval", default=default_interval, help="bar interval, e.g. 15m, 1h, 1d (default: %(default)s)")
    group.add_argument("--start", help="first bar, e.g. 2024-01-01 (required without --csv)")
    group.add_argument("--end", help="last bar (default: now)")
    group.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_CACHE_DIR), help="CSV cache directory (default: %(default)s)")


def _add_json_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def _add_param_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help=help_text)


def _parse_value(text: str) -> Any:
    """JSON value (``50``, ``0.1``, ``true``, ``null``, ``[1, 2]``), Python literals ``True``/``False``/``None``, else the string."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    return {"True": True, "False": False, "None": None}.get(text.strip(), text)


def _parse_params(items: Sequence[str], option: str = "--param") -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key.isidentifier():
            raise ValueError(f"{option} expects KEY=VALUE, got {item!r}")
        params[key] = _parse_value(value.strip())
    return params


def _model_params(args: argparse.Namespace) -> dict[str, Any]:
    params = _parse_params(args.param)
    pipeline.check_model_params(args.model, params)
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptopredict",
        description="Crypto log-return forecasting, walk-forward evaluation and backtesting. " + DISCLAIMER,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("fetch", help="download closed klines into the local cache")
    _add_data_args(p)
    p.add_argument("--out", type=Path, help="also write the bars to this CSV")
    _add_json_arg(p)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("train", help="fit a model on all available history and save it")
    _add_data_args(p)
    p.add_argument("--model", default="ridge", choices=available_models(), help="(default: %(default)s)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="bars ahead to forecast (default: %(default)s)")
    p.add_argument("--out", type=Path, required=True, help="model file to write, e.g. models/btc_ridge.joblib")
    _add_param_arg(p, "model parameter, repeatable, e.g. --param n_estimators=200 --param learning_rate=0.05")
    _add_json_arg(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="forecast from the last closed bar with a saved model")
    p.add_argument("--model-path", type=Path, required=True, help="file written by `train` (trusted files only)")
    p.add_argument("--include-open-bar", action="store_true", help="also use a last bar that has not closed yet")
    _add_data_args(p, default_interval=None)
    _add_json_arg(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward metrics and strategy vs buy-and-hold")
    _add_data_args(p)
    p.add_argument(
        "--model", default="ridge", choices=[*available_models(), "all"],
        help="model or `all` to compare (default: %(default)s)",
    )
    p.add_argument("--include-heavy", action="store_true", help="with --model all, also run slow models (lstm, gru)")
    _add_param_arg(p, "model parameter, repeatable (single --model only)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="bars ahead to forecast (default: %(default)s)")
    p.add_argument("--splits", type=int, default=5, help="walk-forward folds (default: %(default)s)")
    p.add_argument("--fee-bps", type=float, default=10.0, help="fee per position change, bps (default: %(default)s)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="slippage per position change, bps (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=0.0, help="min |forecast| to take a position (default: %(default)s)")
    p.add_argument("--da-threshold", type=float, default=0.0, help="ignore |actual return| <= this in directional accuracy (default: %(default)s)")
    p.add_argument("--allow-short", action="store_true", help="go short on negative forecasts")
    _add_json_arg(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("tune", help="nested walk-forward hyperparameter search")
    _add_data_args(p)
    p.add_argument("--model", default="ridge", choices=available_models(), help="(default: %(default)s)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="bars ahead to forecast (default: %(default)s)")
    p.add_argument("--splits", type=int, default=5, help="outer walk-forward folds (default: %(default)s)")
    p.add_argument("--inner-splits", type=int, default=3, help="inner folds per search (default: %(default)s)")
    p.add_argument("--n-trials", type=int, default=20, help="candidates per search, defaults included (default: %(default)s)")
    p.add_argument("--time-budget", type=float, help="seconds per search; stops after the running candidate")
    p.add_argument("--metric", default="mae", choices=sorted(tuning.LOWER_IS_BETTER | tuning.HIGHER_IS_BETTER), help="(default: %(default)s)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default: %(default)s)")
    p.add_argument("--da-threshold", type=float, default=0.0, help="ignore |actual return| <= this in directional accuracy (default: %(default)s)")
    _add_param_arg(p, "fixed model parameter, repeatable")
    p.add_argument(
        "--space", action="append", default=[], metavar="KEY=[V1,V2,...]",
        help="search values for one parameter, repeatable (default: the model's built-in space)",
    )
    _add_json_arg(p)
    p.set_defaults(func=cmd_tune)
    return parser


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv is None and not args.start:
        raise ValueError("--start is required unless --csv is given")
    return pipeline.load_ohlcv(
        csv=args.csv,
        symbol=args.symbol or DEFAULT_SYMBOL,
        interval=args.interval,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
    )


def _source(args: argparse.Namespace) -> dict[str, Any]:
    source = {"symbol": args.symbol or (None if args.csv else DEFAULT_SYMBOL), "interval": args.interval}
    if args.csv is not None:
        source["csv"] = str(args.csv)
    return source


def _clean(value: Any) -> Any:
    """JSON-safe copy: NaN/inf -> null, numpy scalars -> Python numbers."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _emit(args: argparse.Namespace, payload: dict[str, Any], text: str) -> None:
    if args.json:
        print(json.dumps(_clean(payload), indent=2))
    else:
        print(text)


def _fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.6g}")


def cmd_fetch(args: argparse.Namespace) -> None:
    df = _load(args)
    if args.out is not None:
        from cryptopredict.data import save_csv

        save_csv(df, args.out)
    payload = {
        **_source(args),
        "bars": len(df),
        "start": df.index[0].isoformat() if len(df) else None,
        "end": df.index[-1].isoformat() if len(df) else None,
        "cache_dir": None if args.csv else str(args.cache_dir),
        "out": str(args.out) if args.out else None,
    }
    text = f"{payload['bars']} bars {payload['start']} .. {payload['end']}"
    if args.out:
        text += f" -> {args.out}"
    _emit(args, payload, text)


def cmd_train(args: argparse.Namespace) -> None:
    params = _model_params(args)
    df = _load(args)
    model, info = pipeline.train(df, args.model, horizon=args.horizon, model_params=params)
    metadata = pipeline.save_trained(model, info, args.out, **_source(args))
    payload = {"out": str(args.out), **metadata}
    text = (
        f"trained {args.model} (horizon={args.horizon}) on {info['n_samples']} rows "
        f"{info['train_period']['start']} .. {info['train_period']['end']}\n"
        + (f"params {json.dumps(info['model_params'])}\n" if info["model_params"] else "")
        + f"saved -> {args.out}"
    )
    _emit(args, payload, text)


def predict_history_bars(model: Any, metadata: dict[str, Any], interval: str) -> int:
    """Bars to download for a live forecast: feature warm-up + model lookback + margin."""
    cfg = FeatureConfig.from_dict(metadata.get("feature_config") or {})
    warmup = required_history(cfg, pipeline.interval_to_timedelta(interval))
    lookback = max(int(getattr(model, "lookback", 1)), 1)
    return max(PREDICT_LOOKBACK_BARS, warmup + lookback - 1 + PREDICT_MARGIN_BARS)


def cmd_predict(args: argparse.Namespace) -> None:
    model, metadata = load_model(args.model_path)
    if args.csv is None:
        args.symbol = args.symbol or metadata.get("symbol")
        args.interval = args.interval or metadata.get("interval")
        if not args.symbol or not args.interval:
            raise ValueError("model has no symbol/interval metadata: pass --symbol and --interval, or --csv")
        if not args.start:
            lookback = predict_history_bars(model, metadata, args.interval) * pipeline.interval_to_timedelta(args.interval)
            args.start = (datetime.now(timezone.utc) - lookback).strftime("%Y-%m-%dT%H:%M:%S")
    elif args.interval:
        metadata = {**metadata, "interval": args.interval}
    df = _load(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pipeline.OpenBarWarning)
        result = pipeline.predict_latest(model, metadata, df, include_open_bar=args.include_open_bar)
    if result["dropped_open_bars"]:
        print(
            f"note: ignored {result['dropped_open_bars']} unclosed bar(s) at the end of the data "
            "(--include-open-bar to use them)",
            file=sys.stderr,
        )
    label = " ".join(str(result[k]) for k in ("symbol", "interval") if result.get(k))
    text = "\n".join(
        [
            f"{label or 'forecast'} | model={result['model']} horizon={result['horizon']}",
            f"last closed bar : {result['as_of']}  close={result['last_close']:.6g}",
            f"forecast        : log-return {result['predicted_log_return']:+.6f} "
            f"({result['predicted_pct_change']:+.4f}%) -> {result['direction']}",
            f"expected price  : {result['expected_price']:.6g}"
            + (f" at {result['target_time']}" if result["target_time"] else ""),
            DISCLAIMER,
        ]
    )
    _emit(args, result, text)


def cmd_backtest(args: argparse.Namespace) -> None:
    if args.model == "all" and args.param:
        raise ValueError("--param needs a single --model, not `all`")
    params = {} if args.model == "all" else _model_params(args)
    df = _load(args)
    options = dict(
        horizon=args.horizon,
        n_splits=args.splits,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        threshold=args.threshold,
        da_threshold=args.da_threshold,
        allow_short=args.allow_short,
        interval=args.interval,
    )
    if args.model == "all":
        names = available_models() if args.include_heavy else default_compare_models()
        table, _ = pipeline.compare_models(df, names, **options)
        payload = {
            **_source(args),
            **options,
            "include_heavy": args.include_heavy,
            "models": table.reset_index().to_dict(orient="records"),
        }
        text = (
            f"model comparison (walk-forward, {args.splits} folds, sorted by rmse)\n"
            f"{_fmt_table(table[COMPARISON_COLUMNS])}\n{DISCLAIMER}"
        )
        _emit(args, payload, text)
        return

    report = pipeline.run_backtest(df, args.model, model_params=params, **options)
    folds = report.evaluation.folds
    stats = report.backtest.stats
    payload = {
        **_source(args),
        **options,
        "model": args.model,
        "model_params": pipeline.resolve_model_params(args.model, params, horizon=args.horizon),
        "folds": json.loads(folds.to_json(orient="records", date_format="iso")),
        "overall": report.evaluation.overall,
        "backtest": stats,
    }
    overall = pd.Series(report.evaluation.overall, name=args.model).to_frame().T
    strategy = pd.DataFrame(
        {
            "strategy": [stats["total_return"], stats["sharpe"], stats["max_drawdown"]],
            "buy_and_hold": [stats["bh_total_return"], stats["bh_sharpe"], stats["bh_max_drawdown"]],
        },
        index=["total_return", "sharpe", "max_drawdown"],
    )
    text = "\n\n".join(
        [
            f"walk-forward folds ({args.model}, horizon={args.horizon})\n{_fmt_table(folds)}",
            f"overall out-of-sample\n{_fmt_table(overall)}",
            f"strategy vs buy-and-hold (fee={args.fee_bps}bps, slippage={args.slippage_bps}bps)\n"
            f"{_fmt_table(strategy)}\n"
            f"trades={stats['n_trades']} exposure={stats['exposure']:.2%} bars={stats['n_bars']}",
            DISCLAIMER,
        ]
    )
    _emit(args, payload, text)


def _search_space(items: Sequence[str]) -> dict[str, list] | None:
    space = _parse_params(items, "--space")
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"--space {key}= needs a non-empty JSON list, e.g. {key}=[0.1,1,10]")
    return space or None


def _param_flags(params: dict[str, Any]) -> str:
    return " ".join(f"--param {k}={json.dumps(v, separators=(',', ':'))}" for k, v in params.items())


TUNE_FOLD_COLUMNS = ["test_start", "test_end", "n_train", "inner_score", "mae", "rmse", "directional_accuracy", "n_trials", "params"]


def cmd_tune(args: argparse.Namespace) -> None:
    base = pipeline.resolve_model_params(args.model, _model_params(args), horizon=args.horizon)
    space = _search_space(args.space)
    df = _load(args)
    X, y = make_dataset(df, horizon=args.horizon)
    if X.empty:
        raise ValueError("not enough history to build a dataset")
    if args.model == "ma" and pipeline.MA_FEATURE in X.columns:
        base.setdefault("column", pipeline.MA_FEATURE)  # as pipeline.make_model does
    search = dict(
        param_distributions=space,
        n_trials=args.n_trials,
        inner_splits=args.inner_splits,
        gap=args.horizon,
        metric=args.metric,
        seed=args.seed,
        time_budget_sec=args.time_budget,
        base_params=base,
    )
    # Honest estimate: every outer fold is tuned on its own training rows only.
    nested = tuning.nested_walk_forward(
        args.model, X, y, outer_splits=args.splits, threshold=args.da_threshold, **search
    )
    # Parameters to deploy: the same search over every row (no score is claimed for it).
    final = tuning.tune(args.model, X, y, **search)
    options = {
        "model": args.model,
        "horizon": args.horizon,
        "splits": args.splits,
        "inner_splits": args.inner_splits,
        "n_trials": args.n_trials,
        "time_budget": args.time_budget,
        "metric": args.metric,
        "seed": args.seed,
        "da_threshold": args.da_threshold,
        "base_params": base,
        "space": space,
    }
    payload = {
        **_source(args),
        **options,
        "folds": json.loads(nested.folds.reset_index().to_json(orient="records", date_format="iso")),
        "overall": nested.overall,
        "best_params": final.best_params,
        "best_inner_score": final.best_score,
        "final_trials": len(final.trials),
        "timed_out": final.timed_out or bool(nested.folds["timed_out"].any()),
    }
    overall = pd.Series(nested.overall, name=args.model).to_frame().T
    text = "\n\n".join(
        [
            f"nested walk-forward ({args.model}, horizon={args.horizon}, {args.splits} outer x "
            f"{args.inner_splits} inner folds, metric={args.metric})\n"
            f"{_fmt_table(nested.folds[TUNE_FOLD_COLUMNS])}",
            f"overall out-of-sample (tuned per fold)\n{_fmt_table(overall)}",
            f"best parameters on all rows ({len(final.trials)} trials, inner {args.metric}="
            f"{final.best_score:.6g}{', time budget hit' if payload['timed_out'] else ''})\n"
            f"{json.dumps(final.best_params)}\n"
            f"train with: cryptopredict train --model {args.model} {_param_flags(final.best_params)}".rstrip(),
            DISCLAIMER,
        ]
    )
    _emit(args, payload, text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
