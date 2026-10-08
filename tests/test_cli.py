import json
import math
import subprocess
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from cryptopredict import cli, data, pipeline
from cryptopredict.evaluation import directional_accuracy
from cryptopredict.features import forward_log_return, make_dataset
from cryptopredict.models import default_compare_models, load_model

from .conftest import SAMPLE_CSV, load_sample_ohlcv


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Fail loudly if any test reaches for the network."""

    def no_network(*args, **kwargs):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(data, "get_ohlcv", no_network)
    monkeypatch.setattr(data, "fetch_klines", no_network)


def run_json(capsys, *argv) -> dict:
    assert cli.main([*argv, "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_help_runs_as_module():
    out = subprocess.run(
        [sys.executable, "-m", "cryptopredict.cli", "--help"], capture_output=True, text=True, check=True
    ).stdout
    for command in ("fetch", "train", "predict", "backtest"):
        assert command in out


def test_package_main_help():
    proc = subprocess.run([sys.executable, "-m", "cryptopredict", "backtest", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--fee-bps" in proc.stdout


def test_train_predict_backtest_offline(tmp_path, capsys):
    model_path = tmp_path / "models" / "btc_ridge.joblib"

    trained = run_json(capsys, "train", "--csv", str(SAMPLE_CSV), "--symbol", "BTCUSDT", "--model", "ridge", "--out", str(model_path))
    assert model_path.exists()
    assert trained["name"] == "ridge"
    assert trained["horizon"] == 1
    assert trained["symbol"] == "BTCUSDT" and trained["interval"] == "1h"
    assert "log_ret_1" in trained["feature_columns"]
    _, metadata = load_model(model_path)
    assert metadata["feature_config"] == trained["feature_config"]

    pred = run_json(capsys, "predict", "--model-path", str(model_path), "--csv", str(SAMPLE_CSV))
    df = load_sample_ohlcv()
    assert pred["as_of"] == df.index[-1].isoformat()
    assert pred["last_close"] == pytest.approx(df["close"].iloc[-1])
    assert pred["expected_price"] == pytest.approx(pred["last_close"] * math.exp(pred["predicted_log_return"]))
    assert pred["direction"] == ("up" if pred["predicted_log_return"] > 0 else "down")
    assert pred["target_time"] == (df.index[-1] + timedelta(hours=2)).isoformat()

    bt = run_json(capsys, "backtest", "--csv", str(SAMPLE_CSV), "--model", "ridge", "--splits", "4", "--fee-bps", "10")
    assert len(bt["folds"]) == 4
    assert {"mae", "rmse", "directional_accuracy"} <= set(bt["overall"])
    assert {"total_return", "bh_total_return", "sharpe", "max_drawdown"} <= set(bt["backtest"])


def test_text_output(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    assert cli.main(["train", "--csv", str(SAMPLE_CSV), "--model", "gbm", "--out", str(model_path)]) == 0
    assert "saved" in capsys.readouterr().out
    assert cli.main(["predict", "--model-path", str(model_path), "--csv", str(SAMPLE_CSV)]) == 0
    out = capsys.readouterr().out
    assert "expected price" in out and cli.DISCLAIMER in out
    assert cli.main(["backtest", "--csv", str(SAMPLE_CSV), "--model", "zero", "--splits", "3"]) == 0
    out = capsys.readouterr().out
    assert "buy_and_hold" in out and "walk-forward folds" in out


def test_backtest_all_models(capsys):
    result = run_json(capsys, "backtest", "--csv", str(SAMPLE_CSV), "--model", "all", "--splits", "3", "--allow-short")
    names = [row["model"] for row in result["models"]]
    assert sorted(names) == sorted(default_compare_models())
    rmse = [row["rmse"] for row in result["models"]]
    assert rmse == sorted(rmse)


def test_backtest_trades_one_bar_returns_for_longer_horizon():
    df = load_sample_ohlcv()
    report = pipeline.run_backtest(df, "ridge", horizon=3, n_splits=3)
    preds = report.evaluation.predictions
    expected = forward_log_return(df["close"], 1).reindex(preds.index)
    np.testing.assert_allclose(report.backtest.frame["realized"].to_numpy(), expected.to_numpy())
    # horizon-3 targets differ from the 1-bar returns the strategy earns
    assert not np.allclose(preds["y_true"].to_numpy(), expected.to_numpy())


def test_ma_baseline_uses_rolling_return_feature_when_available():
    X, _ = make_dataset(load_sample_ohlcv())
    model = pipeline.make_model("ma", list(X.columns))
    if pipeline.MA_FEATURE in X.columns:
        assert model.column == pipeline.MA_FEATURE
    assert pipeline.make_model("ma", ["log_ret_1"]).column is None


def test_predict_uses_saved_feature_columns(tmp_path):
    df = load_sample_ohlcv()
    model, info = pipeline.train(df, "ridge")
    pipeline.save_trained(model, info, tmp_path / "m.joblib", symbol="BTCUSDT", interval="4h")
    result = pipeline.load_and_predict(tmp_path / "m.joblib", df)
    assert result["interval"] == "4h"
    assert result["target_time"] == (df.index[-1] + timedelta(hours=8)).isoformat()


def test_predict_without_data_source_errors(tmp_path, capsys):
    df = load_sample_ohlcv()
    model, info = pipeline.train(df, "zero")
    pipeline.save_trained(model, info, tmp_path / "m.joblib")  # no symbol/interval metadata
    assert cli.main(["predict", "--model-path", str(tmp_path / "m.joblib")]) == 1
    assert "--csv" in capsys.readouterr().err


def test_network_commands_require_start(capsys):
    assert cli.main(["backtest", "--symbol", "ETHUSDT"]) == 1
    assert "--start" in capsys.readouterr().err


def test_fetch_uses_cache_and_writes_csv(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_get_ohlcv(symbol, interval, start, end=None, cache_dir=None):
        calls.append((symbol, interval, start, end, cache_dir))
        return load_sample_ohlcv()

    monkeypatch.setattr(data, "get_ohlcv", fake_get_ohlcv)
    out = tmp_path / "eth.csv"
    result = run_json(
        capsys, "fetch", "--symbol", "ETHUSDT", "--interval", "1h", "--start", "2024-01-01",
        "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
    )
    assert calls == [("ETHUSDT", "1h", "2024-01-01", None, tmp_path / "cache")]
    assert result["bars"] == 600 and result["symbol"] == "ETHUSDT"
    assert result["cache_dir"] == str(tmp_path / "cache")
    pd.testing.assert_frame_equal(data.load_csv(out), load_sample_ohlcv(), check_freq=False)


def test_predict_fetches_recent_history_from_metadata(tmp_path, monkeypatch, capsys):
    df = load_sample_ohlcv()
    model, info = pipeline.train(df, "ridge")
    pipeline.save_trained(model, info, tmp_path / "m.joblib", symbol="BTCUSDT", interval="1h")
    seen = {}

    def fake_get_ohlcv(symbol, interval, start, end=None, cache_dir=None):
        seen.update(symbol=symbol, interval=interval, start=start)
        return df

    monkeypatch.setattr(data, "get_ohlcv", fake_get_ohlcv)
    result = run_json(capsys, "predict", "--model-path", str(tmp_path / "m.joblib"), "--cache-dir", str(tmp_path))
    assert seen["symbol"] == "BTCUSDT" and seen["interval"] == "1h"
    start = pd.Timestamp(seen["start"], tz="UTC")
    expected = pd.Timestamp.now(tz="UTC") - cli.PREDICT_LOOKBACK_BARS * pd.Timedelta(hours=1)
    assert abs(start - expected) < pd.Timedelta(minutes=5)
    assert result["symbol"] == "BTCUSDT"


@pytest.mark.parametrize(
    ("interval", "expected"),
    [("1m", timedelta(minutes=1)), ("15m", timedelta(minutes=15)), ("4h", timedelta(hours=4)), ("1d", timedelta(days=1)), ("1w", timedelta(weeks=1))],
)
def test_interval_to_timedelta(interval, expected):
    assert pipeline.interval_to_timedelta(interval) == expected


@pytest.mark.parametrize("interval", ["", "h", "0h", "1x", "1H"])
def test_interval_to_timedelta_rejects_invalid(interval):
    with pytest.raises(ValueError):
        pipeline.interval_to_timedelta(interval)


def test_periods_per_year():
    assert pipeline.periods_per_year("1h") == 24 * 365
    assert pipeline.periods_per_year("1d") == 365


def test_position_threshold_does_not_change_directional_accuracy():
    df = load_sample_ohlcv()
    base = pipeline.run_backtest(df, "ridge", n_splits=3)
    preds = base.evaluation.predictions
    threshold = float(preds["y_pred"].abs().median())

    report = pipeline.run_backtest(df, "ridge", n_splits=3, threshold=threshold)
    assert report.evaluation.overall["directional_accuracy"] == pytest.approx(
        directional_accuracy(preds["y_true"], preds["y_pred"], 0.0)
    )
    position = report.backtest.frame["position"].to_numpy()
    np.testing.assert_array_equal(position, (preds["y_pred"] > threshold).astype(float).to_numpy())
    assert 0 < position.sum() < len(position)


def test_da_threshold_does_not_change_positions():
    df = load_sample_ohlcv()
    base = pipeline.run_backtest(df, "ridge", n_splits=3)
    preds = base.evaluation.predictions
    da_threshold = float(preds["y_true"].abs().median())

    report = pipeline.run_backtest(df, "ridge", n_splits=3, da_threshold=da_threshold)
    expected = directional_accuracy(preds["y_true"], preds["y_pred"], da_threshold)
    assert report.evaluation.overall["directional_accuracy"] == pytest.approx(expected)
    assert expected != pytest.approx(base.evaluation.overall["directional_accuracy"])
    pd.testing.assert_frame_equal(report.backtest.frame, base.backtest.frame)
    assert report.backtest.stats == base.backtest.stats


def test_backtest_json_reports_both_thresholds(capsys):
    result = run_json(
        capsys, "backtest", "--csv", str(SAMPLE_CSV), "--model", "zero", "--splits", "3",
        "--threshold", "0.001", "--da-threshold", "0.002",
    )
    assert result["threshold"] == 0.001
    assert result["da_threshold"] == 0.002


def _trained(df, **extra):
    model, info = pipeline.train(df, "ridge")
    metadata = {"name": model.name, "horizon": info["horizon"], "feature_columns": info["feature_columns"],
                "feature_config": info["feature_config"], **extra}
    return model, metadata


def test_predict_drops_open_last_bar():
    df = load_sample_ohlcv()
    model, metadata = _trained(df, interval="1h")
    now = df.index[-1] + pd.Timedelta(minutes=20)  # last bar opened 20 min ago, still open
    with pytest.warns(pipeline.OpenBarWarning):
        result = pipeline.predict_latest(model, metadata, df, now=now)
    assert result["as_of"] == df.index[-2].isoformat()
    assert result["last_close"] == pytest.approx(df["close"].iloc[-2])
    assert result["dropped_open_bars"] == 1


def test_predict_include_open_bar_keeps_old_behaviour(recwarn):
    df = load_sample_ohlcv()
    model, metadata = _trained(df, interval="1h")
    now = df.index[-1] + pd.Timedelta(minutes=20)
    result = pipeline.predict_latest(model, metadata, df, now=now, include_open_bar=True)
    assert result["as_of"] == df.index[-1].isoformat()
    assert result["dropped_open_bars"] == 0
    assert not [w for w in recwarn if issubclass(w.category, pipeline.OpenBarWarning)]


def test_predict_keeps_bar_closing_exactly_now(recwarn):
    df = load_sample_ohlcv()
    model, metadata = _trained(df, interval="1h")
    result = pipeline.predict_latest(model, metadata, df, now=df.index[-1] + pd.Timedelta(hours=1))
    assert result["as_of"] == df.index[-1].isoformat()
    assert not [w for w in recwarn if issubclass(w.category, pipeline.OpenBarWarning)]


def test_predict_without_interval_assumes_closed_bars():
    df = load_sample_ohlcv()
    model, metadata = _trained(df)
    result = pipeline.predict_latest(model, metadata, df, now=df.index[-1])
    assert result["as_of"] == df.index[-1].isoformat()


def test_drop_open_bars_naive_now_and_all_open():
    df = load_sample_ohlcv()
    closed, dropped = pipeline.drop_open_bars(df, "1h", now=df.index[-3].tz_localize(None) + pd.Timedelta(hours=1))
    assert dropped == 2 and closed.index[-1] == df.index[-3]
    model, metadata = _trained(df, interval="1h")
    with pytest.raises(ValueError, match="no closed bars"), pytest.warns(pipeline.OpenBarWarning):
        pipeline.predict_latest(model, metadata, df, now=df.index[0])


def test_cli_predict_skips_open_bar_in_csv(tmp_path, capsys):
    model_path = tmp_path / "m.joblib"
    assert cli.main(["train", "--csv", str(SAMPLE_CSV), "--out", str(model_path)]) == 0
    capsys.readouterr()
    df = load_sample_ohlcv()
    current_bar = pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(minutes=1)  # open for 59 more minutes
    df.index = (df.index + (current_bar - df.index[-1])).rename(df.index.name)
    live_csv = tmp_path / "live.csv"
    data.save_csv(df, live_csv)

    assert cli.main(["predict", "--model-path", str(model_path), "--csv", str(live_csv), "--json"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["as_of"] == df.index[-2].isoformat()
    assert result["dropped_open_bars"] == 1
    assert "--include-open-bar" in captured.err

    args = ["predict", "--model-path", str(model_path), "--csv", str(live_csv), "--include-open-bar", "--json"]
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["as_of"] == df.index[-1].isoformat()
    assert captured.err == ""
