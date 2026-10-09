#!/usr/bin/env python3
"""Reproducible real-data benchmark used by ``docs/results.md``.

Raw Binance candles and generated tables live below ``data/`` (gitignored).
The script deliberately keeps model failures/skips as rows instead of silently
dropping them, so a partial run remains auditable.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cryptopredict import pipeline
from cryptopredict.evaluation import backtest, dm_vs_zero
from cryptopredict.features import FeatureConfig, forward_log_return, make_dataset
from cryptopredict.models import available_models
from cryptopredict.models.ensemble import DEFAULT_MEMBERS
from cryptopredict.pipeline import load_ohlcv, periods_per_year, run_backtest

ALL_MODELS = [
    "zero", "mean", "last", "ma", "ridge", "gbm", "lgbm", "lgbm_cls",
    "ensemble", "stack", "lstm", "gru",
]
DEEP_MODELS = {"lstm", "gru"}
SEEDED_MODELS = {"gbm", "lgbm", "lgbm_cls", *DEEP_MODELS}
# Holm family: every model except ``zero`` (its loss difference is identically 0)
# run on the same symbol/interval/horizon/feature set, for one loss.
HOLM_GROUP = ["symbol", "interval", "horizon", "feature_set", "loss"]
HOLM_EXCLUDED = {"zero"}
SLICE = ["symbol", "interval", "horizon"]
LEGACY_PREFIXES = (
    "log_ret_", "ret_mean_", "close_sma_", "close_ema_", "rsi_", "macd",
    "bollinger_", "volatility_", "volume_zscore_",
)
RESULT_COLUMNS = [
    "run_id", "commit_sha", "symbol", "interval", "horizon", "feature_set",
    "model", "loss", "hac_kernel", "data_start", "data_end", "n_bars",
    "test_start", "test_end", "n_obs", "mae", "rmse", "relative_mae",
    "directional_accuracy", "dm_mean_diff", "dm_stat", "dm_p_value",
    "dm_p_holm", "total_return", "sharpe", "max_drawdown", "n_trades",
    "bh_total_return", "bh_sharpe", "bh_max_drawdown", "excess_return", "exposure",
    "total_cost", "backtest_n_bars", "fee_bps", "threshold", "seed",
    "max_train_seconds", "deep_epochs", "deep_time_stops", "runtime_seconds",
    "status", "reason",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cryptopredict models on cached/public Binance klines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--intervals", nargs="+", default=["1h", "4h", "1d"])
    parser.add_argument(
        "--horizons", nargs="+", type=int,
        help="apply these horizons to every interval; default is 1/4/24 for 1h and 1 otherwise",
    )
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--ablation-models", nargs="+", default=["ridge", "gbm", "lgbm"])
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--start-1h", default="2025-01-01")
    parser.add_argument("--start-slow", default="2022-01-01")
    parser.add_argument("--end", default="2026-10-01")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/benchmark/cache"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark/results.csv"))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument(
        "--threshold-bps", nargs="+", type=float, default=[0.0, 2.5, 5.0, 10.0],
        help="signal thresholds re-applied to the OOS forecasts of every successful model",
    )
    parser.add_argument(
        "--threshold-focus", nargs="+", default=["ridge", "lgbm"],
        help="models fixed in advance for the per-slice threshold table (no test-set selection)",
    )
    parser.add_argument("--loss", choices=["absolute", "squared"], default="absolute")
    parser.add_argument("--hac-kernel", choices=["bartlett", "uniform"], default="bartlett")
    parser.add_argument("--seed", type=int, default=42, help="random_state of every stochastic model")
    parser.add_argument("--deep-max-train-seconds", type=float, default=60.0)
    parser.add_argument("--deep-max-epochs", type=int, default=100)
    parser.add_argument(
        "--deep-cases", nargs="+", default=["BTCUSDT:1h:1", "BTCUSDT:4h:1"],
        metavar="SYMBOL:INTERVAL:HORIZON",
        help="cases in which LSTM/GRU are run; other deep rows remain explicit skips",
    )
    parser.add_argument("--quick", action="store_true", help="one-symbol smoke matrix with light models")
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _versions() -> dict[str, str]:
    packages = ["cryptopredict", "numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "torch"]
    result = {"python": platform.python_version()}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _holm(values: pd.Series) -> pd.Series:
    """Holm step-down adjustment, preserving missing values and row order."""
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna().sort_values(kind="stable")
    count = len(valid)
    running = 0.0
    for rank, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, (count - rank) * float(value)))
        output.loc[index] = running
    return output


def _legacy_dataset(df: pd.DataFrame, horizon: int, config: FeatureConfig):
    X, y = make_dataset(df, horizon=horizon, config=config)
    columns = [column for column in X if column.startswith(LEGACY_PREFIXES)]
    if not columns:
        raise ValueError("legacy feature selection produced no columns")
    return X[columns], y


def _model_params(name: str, args: argparse.Namespace) -> dict[str, Any] | None:
    if name in DEEP_MODELS:
        return {
            "random_state": args.seed,
            "max_train_seconds": args.deep_max_train_seconds,
            "max_epochs": args.deep_max_epochs,
        }
    if name in SEEDED_MODELS:
        return {"random_state": args.seed}
    if name in {"ensemble", "stack"}:
        members = tuple(
            (member, {"random_state": args.seed}) if member in SEEDED_MODELS else member
            for member in DEFAULT_MEMBERS
        )
        return {"members": members}
    return None


@contextmanager
def _recording_models():
    """Collect the per-fold models ``run_backtest`` builds, to read their fit attributes."""
    built: list[Any] = []
    original = pipeline.make_model

    def make_and_record(*args: Any, **kwargs: Any):
        model = original(*args, **kwargs)
        built.append(model)
        return model

    pipeline.make_model = make_and_record
    try:
        yield built
    finally:
        pipeline.make_model = original


def _deep_budget(models: list[Any]) -> dict[str, Any]:
    """Epochs per fold and how many folds hit the wall-clock limit (hardware dependent)."""
    if not models or not all(hasattr(model, "n_epochs_") for model in models):
        return {}
    return {
        "deep_epochs": ";".join(str(int(model.n_epochs_)) for model in models),
        "deep_time_stops": sum(bool(getattr(model, "stopped_by_time_", False)) for model in models),
    }


def _empty_row(context: dict[str, Any], model: str, args: argparse.Namespace) -> dict[str, Any]:
    row = dict.fromkeys(RESULT_COLUMNS)
    row.update(context)
    row.update({
        "model": model,
        "loss": args.loss,
        "hac_kernel": args.hac_kernel,
        "fee_bps": args.fee_bps,
        "threshold": 0.0,
        "seed": args.seed,
        "max_train_seconds": args.deep_max_train_seconds if model in DEEP_MODELS else None,
    })
    return row


def _evaluate(
    df: pd.DataFrame,
    model: str,
    horizon: int,
    interval: str,
    dataset,
    context: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any | None]:
    row = _empty_row(context, model, args)
    deep_case = f"{context['symbol']}:{interval}:{horizon}"
    if model in DEEP_MODELS and deep_case not in args.deep_cases:
        row.update(status="skipped", reason=f"deep time budget excludes {deep_case}")
        return row, None
    if model not in available_models():
        row.update(status="skipped", reason="optional dependency/model unavailable")
        return row, None
    started = time.monotonic()
    try:
        with _recording_models() as fold_models:
            report = run_backtest(
                df, model, horizon=horizon, n_splits=args.splits, fee_bps=args.fee_bps,
                interval=interval, feature_config=FeatureConfig(), dataset=dataset,
                model_params=_model_params(model, args),
            )
        row.update(_deep_budget(fold_models))
        predictions = report.evaluation.predictions
        dm = dm_vs_zero(
            predictions["y_true"], predictions["y_pred"], horizon=horizon,
            loss=args.loss, alternative="two-sided", kernel=args.hac_kernel,
        )
        row.update(report.evaluation.overall)
        backtest_stats = dict(report.backtest.stats)
        row["backtest_n_bars"] = backtest_stats.pop("n_bars")
        row.update(backtest_stats)
        row.update({
            "test_start": predictions.index[0].isoformat(),
            "test_end": predictions.index[-1].isoformat(),
            "n_obs": len(predictions),
            "dm_mean_diff": dm.mean_loss_diff,
            "dm_stat": dm.statistic,
            "dm_p_value": dm.p_value,
            "runtime_seconds": time.monotonic() - started,
            "status": "ok",
            "reason": "",
        })
        return row, report
    except Exception as exc:  # keep partial matrices auditable
        row.update(
            runtime_seconds=time.monotonic() - started,
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
        )
        return row, None


def _threshold_rows(
    report,
    context: dict[str, Any],
    model: str,
    interval: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    predictions = report.evaluation.predictions
    realized = forward_log_return(context["frame"]["close"], 1).reindex(predictions.index)
    rows = []
    for threshold_bps in args.threshold_bps:
        threshold = threshold_bps / 10_000.0
        result = backtest(
            predictions["y_pred"], realized, fee_bps=args.fee_bps,
            threshold=threshold, periods_per_year=periods_per_year(interval),
        )
        rows.append({
            "run_id": context["run_id"], "commit_sha": context["commit_sha"],
            "symbol": context["symbol"], "interval": interval,
            "horizon": context["horizon"], "feature_set": context["feature_set"],
            "model": model, "threshold_bps": threshold_bps, **result.stats,
        })
    return rows


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame[columns].copy()
    for column in values.select_dtypes(include="number"):
        values[column] = values[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6g}"
        )
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def _summary_tables(
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    args: argparse.Namespace,
) -> list[tuple[str, str, pd.DataFrame]]:
    """``(title, description, table)`` sections that ``docs/results.md`` quotes."""
    ok = results[results["status"] == "ok"].copy()
    ok["holm_better"] = (ok["dm_stat"] < 0) & (ok["dm_p_holm"] < 0.05)
    ok["holm_worse"] = (ok["dm_stat"] > 0) & (ok["dm_p_holm"] < 0.05)
    ok["raw_better"] = (ok["dm_stat"] < 0) & (ok["dm_p_value"] < 0.05)
    rich = ok[ok["feature_set"] == "rich"]
    learned = rich[~rich["model"].isin(HOLM_EXCLUDED)]
    sections: list[tuple[str, str, pd.DataFrame]] = []

    best = (
        learned.sort_values([*SLICE, "mae"], kind="stable").groupby(SLICE, sort=False).head(1)
    )
    sections.append((
        "Lowest-MAE non-zero model per slice (rich features)",
        "Descriptive only: picking the minimum over models is itself a selection; see dm_p_holm.",
        best[[*SLICE, "model", "mae", "rmse", "relative_mae", "directional_accuracy",
              "dm_stat", "dm_p_value", "dm_p_holm", "total_return", "bh_total_return",
              "n_trades"]],
    ))

    family = ok[~ok["model"].isin(HOLM_EXCLUDED)]
    if not family.empty:
        significance = family.groupby([*SLICE, "feature_set"], sort=False).agg(
            models=("model", "size"),
            raw_better=("raw_better", "sum"),
            holm_better=("holm_better", "sum"),
            holm_worse=("holm_worse", "sum"),
            min_p_better=("dm_p_value", lambda p: p[family.loc[p.index, "dm_stat"] < 0].min()),
        ).reset_index()
        sections.append((
            "DM test against zero per Holm family",
            "better/worse = DM statistic sign with p < 0.05; min_p_better is the smallest raw p "
            "among models with a negative statistic.",
            significance,
        ))

    if not learned.empty:
        by_model = learned.groupby("model", sort=False).agg(
            conditions=("relative_mae", "size"),
            mean_relative_mae=("relative_mae", "mean"),
            best_relative_mae=("relative_mae", "min"),
            below_one=("relative_mae", lambda values: int((values < 1).sum())),
            holm_better=("holm_better", "sum"),
            holm_worse=("holm_worse", "sum"),
            median_total_return=("total_return", "median"),
        ).reset_index().sort_values("mean_relative_mae", kind="stable")
        sections.append(("Model summary over rich-feature slices", "", by_model))

    legacy = ok[ok["feature_set"] == "legacy"]
    if not legacy.empty:
        pairs = rich.merge(legacy, on=[*SLICE, "model"], suffixes=("_rich", "_legacy"))
        pairs["rich_better"] = pairs["relative_mae_rich"] < pairs["relative_mae_legacy"]
        pairs["difference"] = pairs["relative_mae_rich"] - pairs["relative_mae_legacy"]
        ablation = pairs.groupby("model", sort=False).agg(
            conditions=("difference", "size"),
            rich_better=("rich_better", "sum"),
            legacy_mean=("relative_mae_legacy", "mean"),
            rich_mean=("relative_mae_rich", "mean"),
            mean_difference=("difference", "mean"),
        ).reset_index()
        sections.append((
            "Rich vs legacy features (relative MAE)",
            "Positive mean_difference means the rich set is worse.",
            ablation,
        ))

    if not thresholds.empty:
        rich_thresholds = thresholds[thresholds["feature_set"] == "rich"].copy()
        rich_thresholds["beats_bh"] = rich_thresholds["total_return"] > rich_thresholds["bh_total_return"]
        rich_thresholds["positive"] = rich_thresholds["total_return"] > 0
        aggregate = rich_thresholds.groupby("threshold_bps").agg(
            model_slices=("total_return", "size"),
            median_total_return=("total_return", "median"),
            positive=("positive", "sum"),
            beats_bh=("beats_bh", "sum"),
            median_n_trades=("n_trades", "median"),
        ).reset_index()
        sections.append((
            "Threshold sensitivity over every successful non-zero model (rich features)",
            "Thresholds are applied after the fact to the same OOS forecasts; nothing is selected "
            "on the test period, so these are not achievable returns of a tuned rule.",
            aggregate,
        ))
        focus = rich_thresholds[rich_thresholds["model"].isin(args.threshold_focus)]
        if not focus.empty:
            wide = focus.pivot_table(
                index=[*SLICE, "model"], columns="threshold_bps",
                values=["total_return", "n_trades"], sort=False,
            )
            wide.columns = [
                f"{'ret' if name == 'total_return' else 'trades'}_{bps:g}bps" for name, bps in wide.columns
            ]
            bh = focus.groupby([*SLICE, "model"], sort=False)["bh_total_return"].first()
            wide = wide.join(bh).reset_index()
            sections.append((
                f"Threshold sensitivity for models fixed in advance ({', '.join(args.threshold_focus)})",
                "", wide,
            ))

    deep = ok[ok["model"].isin(DEEP_MODELS)]
    if not deep.empty:
        sections.append((
            "Time-budgeted LSTM/GRU",
            f"At most {args.deep_max_train_seconds:g} s and {args.deep_max_epochs} epochs per fold; "
            "deep_epochs lists the epochs reached per fold, deep_time_stops the folds stopped by the clock.",
            deep[[*SLICE, "feature_set", "model", "relative_mae", "directional_accuracy", "dm_stat",
                  "dm_p_value", "dm_p_holm", "total_return", "n_trades", "deep_epochs",
                  "deep_time_stops", "runtime_seconds"]],
        ))
    return sections


def _write_outputs(
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    threshold_path = args.output.with_name(f"{args.output.stem}_thresholds.csv")
    thresholds.to_csv(threshold_path, index=False)
    metadata_path = args.output.with_name(f"{args.output.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    markdown_path = args.output.with_suffix(".md")
    ok = results[results["status"] == "ok"].sort_values(
        ["symbol", "interval", "horizon", "feature_set", "mae"], kind="stable"
    )
    columns = [
        "symbol", "interval", "horizon", "feature_set", "model", "n_obs",
        "mae", "relative_mae", "directional_accuracy", "dm_stat", "dm_p_value",
        "dm_p_holm", "total_return", "sharpe", "max_drawdown", "n_trades",
    ]
    parts = [
        "# Generated benchmark tables\n",
        f"Run `{metadata['run_id']}`, commit `{metadata['commit_sha']}`.\n",
    ]
    for title, description, table in _summary_tables(results, thresholds, args):
        parts.append(f"## {title}\n")
        if description:
            parts.append(description + "\n")
        parts.append(_markdown_table(table, list(table.columns)) + "\n")
    parts.append("## All successful rows\n")
    parts.append(_markdown_table(ok, columns) + "\n")
    markdown_path.write_text("\n".join(parts))
    print(f"wrote {args.output}, {threshold_path}, {metadata_path}, {markdown_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.splits < 1 or any(value < 1 for value in (args.horizons or [1])):
        raise SystemExit("splits and horizons must be positive")
    if args.deep_max_train_seconds <= 0 or args.deep_max_epochs < 1:
        raise SystemExit("deep training limits must be positive")
    if any(value < 0 for value in args.threshold_bps):
        raise SystemExit("threshold-bps values must be non-negative")
    if args.quick:
        args.symbols = args.symbols[:1]
        args.intervals = ["1h"]
        args.horizons = [1]
        args.models = ["zero", "mean", "ridge"]
        args.no_ablation = True
        args.start_1h = max(args.start_1h, "2026-07-01")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit_sha = _git_sha()
    metadata = {
        "run_id": run_id,
        "commit_sha": commit_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": _versions(),
        "note": "No hyperparameter tuning; fixed defaults avoid test-set selection.",
        "holm_family": f"{HOLM_GROUP}, excluding {sorted(HOLM_EXCLUDED)}",
        "thresholds": "every successful non-zero model, applied post hoc; no selection",
        "deep_budget": "wall-clock stopping depends on hardware; see deep_epochs/deep_time_stops",
    }
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for symbol in args.symbols:
        for interval in args.intervals:
            start = args.start_1h if interval == "1h" else args.start_slow
            print(f"loading {symbol} {interval} {start}..{args.end}", flush=True)
            df = load_ohlcv(
                symbol=symbol, interval=interval, start=start, end=args.end,
                cache_dir=args.cache_dir,
            )
            horizons = args.horizons or ([1, 4, 24] if interval == "1h" else [1])
            for horizon in horizons:
                config = FeatureConfig()
                feature_cases: list[tuple[str, Any]] = [
                    ("rich", make_dataset(df, horizon=horizon, config=config)),
                ]
                if not args.no_ablation:
                    try:
                        feature_cases.append(("legacy", _legacy_dataset(df, horizon, config)))
                    except Exception as exc:  # recorded per model below
                        feature_cases.append(("legacy", exc))
                for feature_set, dataset in feature_cases:
                    models = args.models if feature_set == "rich" else [
                        model for model in args.models if model in args.ablation_models
                    ]
                    context = {
                        "run_id": run_id,
                        "commit_sha": commit_sha,
                        "symbol": symbol,
                        "interval": interval,
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "data_start": df.index[0].isoformat(),
                        "data_end": df.index[-1].isoformat(),
                        "n_bars": len(df),
                    }
                    for model in models:
                        print(f"  {interval} h={horizon} {feature_set} {model}", flush=True)
                        if isinstance(dataset, Exception):
                            row = _empty_row(context, model, args)
                            row.update(status="error", reason=f"{type(dataset).__name__}: {dataset}")
                            rows.append(row)
                            continue
                        row, report = _evaluate(df, model, horizon, interval, dataset, context, args)
                        rows.append(row)
                        if report is not None and model not in HOLM_EXCLUDED:
                            threshold_rows.extend(
                                _threshold_rows(report, {**context, "frame": df}, model, interval, args)
                            )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    family = (results["status"] == "ok") & ~results["model"].isin(HOLM_EXCLUDED)
    results["dm_p_holm"] = np.nan
    results.loc[family, "dm_p_holm"] = (
        results.loc[family].groupby(HOLM_GROUP, dropna=False)["dm_p_value"].transform(_holm)
    )
    thresholds = pd.DataFrame(threshold_rows)
    _write_outputs(results, thresholds, args, metadata)
    errors = int((results["status"] == "error").sum())
    print(f"completed {len(results)} rows ({errors} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
