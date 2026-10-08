"""Command line interface: ``cryptopredict {fetch,train,predict,backtest}``.

Run ``cryptopredict <command> --help`` for the options. ``--csv`` makes every
command work offline. Forecasts are not investment advice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from cryptopredict import __version__, pipeline
from cryptopredict.core.types import DEFAULT_HORIZON
from cryptopredict.models import available_models, load_model

DISCLAIMER = "Not investment advice / Yatırım tavsiyesi değildir."
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_CACHE_DIR = "data"
# Bars of history fetched by ``predict`` when no CSV is given (covers indicator warm-up).
PREDICT_LOOKBACK_BARS = 500
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
    _add_json_arg(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="forecast from the last closed bar with a saved model")
    p.add_argument("--model-path", type=Path, required=True, help="file written by `train` (trusted files only)")
    _add_data_args(p, default_interval=None)
    _add_json_arg(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("backtest", help="walk-forward metrics and strategy vs buy-and-hold")
    _add_data_args(p)
    p.add_argument("--model", default="ridge", choices=[*available_models(), "all"], help="model or `all` to compare (default: %(default)s)")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="bars ahead to forecast (default: %(default)s)")
    p.add_argument("--splits", type=int, default=5, help="walk-forward folds (default: %(default)s)")
    p.add_argument("--fee-bps", type=float, default=10.0, help="fee per position change, bps (default: %(default)s)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="slippage per position change, bps (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=0.0, help="min |forecast| to take a position (default: %(default)s)")
    p.add_argument("--allow-short", action="store_true", help="go short on negative forecasts")
    _add_json_arg(p)
    p.set_defaults(func=cmd_backtest)
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
        "out": str(args.out) if args.out else None,
    }
    text = f"{payload['bars']} bars {payload['start']} .. {payload['end']}"
    if args.out:
        text += f" -> {args.out}"
    _emit(args, payload, text)


def cmd_train(args: argparse.Namespace) -> None:
    df = _load(args)
    model, info = pipeline.train(df, args.model, horizon=args.horizon)
    metadata = pipeline.save_trained(model, info, args.out, **_source(args))
    payload = {"out": str(args.out), **metadata}
    text = (
        f"trained {args.model} (horizon={args.horizon}) on {info['n_samples']} rows "
        f"{info['train_period']['start']} .. {info['train_period']['end']}\n"
        f"saved -> {args.out}"
    )
    _emit(args, payload, text)


def cmd_predict(args: argparse.Namespace) -> None:
    model, metadata = load_model(args.model_path)
    if args.csv is None:
        args.symbol = args.symbol or metadata.get("symbol")
        args.interval = args.interval or metadata.get("interval")
        if not args.symbol or not args.interval:
            raise ValueError("model has no symbol/interval metadata: pass --symbol and --interval, or --csv")
        if not args.start:
            lookback = PREDICT_LOOKBACK_BARS * pipeline.interval_to_timedelta(args.interval)
            args.start = (datetime.now(timezone.utc) - lookback).strftime("%Y-%m-%dT%H:%M:%S")
    elif args.interval:
        metadata = {**metadata, "interval": args.interval}
    df = _load(args)
    result = pipeline.predict_latest(model, metadata, df)
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
    df = _load(args)
    options = dict(
        horizon=args.horizon,
        n_splits=args.splits,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        threshold=args.threshold,
        allow_short=args.allow_short,
        interval=args.interval,
    )
    if args.model == "all":
        table, _ = pipeline.compare_models(df, **options)
        payload = {**_source(args), **options, "models": table.reset_index().to_dict(orient="records")}
        text = (
            f"model comparison (walk-forward, {args.splits} folds, sorted by rmse)\n"
            f"{_fmt_table(table[COMPARISON_COLUMNS])}\n{DISCLAIMER}"
        )
        _emit(args, payload, text)
        return

    report = pipeline.run_backtest(df, args.model, **options)
    folds = report.evaluation.folds
    stats = report.backtest.stats
    payload = {
        **_source(args),
        **options,
        "model": args.model,
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
