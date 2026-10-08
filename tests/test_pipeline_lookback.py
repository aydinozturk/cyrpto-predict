import numpy as np
import pandas as pd
import pytest

from cryptopredict import pipeline
from cryptopredict.features import FeatureConfig, latest_features
from cryptopredict.models import RidgeForecaster, ZeroReturn, load_model, registry

from .conftest import load_sample_ohlcv


class WindowRidge(RidgeForecaster):
    """Ridge that pretends to be a sequence model and records what predict sees."""

    lookback = 3

    def predict(self, X):
        self.seen_ = X.copy()
        return super().predict(X)


class SlowZero(ZeroReturn):
    name = "slow_zero"
    heavy = True


def _trained(model_name="ridge", **train_kw):
    df = load_sample_ohlcv()
    model, info = pipeline.train(df, model_name, **train_kw)
    metadata = {
        "name": model.name,
        "horizon": info["horizon"],
        "feature_columns": info["feature_columns"],
        "feature_config": info["feature_config"],
    }
    return df, model, metadata


def test_lookback_one_is_the_latest_row():
    df, model, metadata = _trained()
    row = latest_features(df, config=FeatureConfig.from_dict(metadata["feature_config"]))
    result = pipeline.predict_latest(model, metadata, df)
    assert result["as_of"] == df.index[-1].isoformat()
    assert result["predicted_log_return"] == pytest.approx(model.predict(row[metadata["feature_columns"]])[0])


def test_lookback_model_gets_last_rows_and_uses_last_forecast():
    df, ridge, metadata = _trained()
    model = WindowRidge()
    model.__dict__.update(ridge.__dict__)
    result = pipeline.predict_latest(model, metadata, df)
    seen = model.seen_
    assert len(seen) == 3
    assert list(seen.columns) == metadata["feature_columns"]
    assert list(seen.index) == list(df.index[-3:])
    assert not seen.isna().any(axis=None)
    assert result["as_of"] == df.index[-1].isoformat()
    assert result["predicted_log_return"] == pytest.approx(ridge.predict(seen)[-1])


def test_lookback_longer_than_complete_history_errors():
    df, ridge, metadata = _trained()
    model = WindowRidge()
    model.__dict__.update(ridge.__dict__)
    model.lookback = len(df)
    with pytest.raises(ValueError, match=f"needs {len(df)} complete feature rows"):
        pipeline.predict_latest(model, metadata, df)
    model.lookback = 0
    with pytest.raises(ValueError, match="lookback must be >= 1"):
        pipeline.predict_latest(model, metadata, df)


def test_model_params_reach_the_model_and_metadata(tmp_path):
    df, model, _ = _trained(model_params={"alpha": 1234.5})
    assert model.alpha == 1234.5
    _, info = pipeline.train(df, "ridge", model_params={"alpha": 1234.5})
    assert info["model_params"] == {"alpha": 1234.5}
    pipeline.save_trained(model, info, tmp_path / "m.joblib", symbol="BTCUSDT", interval="1h")
    loaded, metadata = load_model(tmp_path / "m.joblib")
    assert metadata["model_params"] == {"alpha": 1234.5} and loaded.alpha == 1234.5
    assert pipeline.train(df, "ridge")[1]["model_params"] == {}


def test_make_model_params_and_ma_column():
    assert pipeline.make_model("ridge", params={"alpha": 7.0}).alpha == 7.0
    ma = pipeline.make_model("ma", [pipeline.MA_FEATURE], params={"window": 5})
    assert ma.window == 5 and ma.column == pipeline.MA_FEATURE
    assert pipeline.make_model("ma", [pipeline.MA_FEATURE], params={"column": None}).column is None


def test_run_backtest_passes_model_params():
    df = load_sample_ohlcv()
    base = pipeline.run_backtest(df, "ridge", n_splits=2)
    shrunk = pipeline.run_backtest(df, "ridge", n_splits=2, model_params={"alpha": 1e9})
    a = base.evaluation.predictions["y_pred"].to_numpy()
    b = shrunk.evaluation.predictions["y_pred"].to_numpy()
    assert not np.allclose(a, b)
    assert np.abs(b - b.mean()).max() < np.abs(a - a.mean()).max()


def test_compare_models_skips_heavy_by_default(monkeypatch):
    monkeypatch.setitem(registry.LAZY_MODELS, "slow_zero", f"{__name__}:SlowZero")
    df = load_sample_ohlcv()
    table, reports = pipeline.compare_models(df, n_splits=2)
    assert "slow_zero" not in reports and "zero" in reports
    _, reports = pipeline.compare_models(df, ["zero", "slow_zero"], n_splits=2)
    assert set(reports) == {"zero", "slow_zero"}
    assert isinstance(table, pd.DataFrame)
