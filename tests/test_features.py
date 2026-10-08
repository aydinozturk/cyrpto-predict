import numpy as np
import pandas as pd
import pytest

from cryptopredict.core.types import LOG_RET_1, TARGET_COLUMN
from cryptopredict.features import (
    FeatureConfig,
    build_features,
    forward_log_return,
    latest_features,
    make_dataset,
    make_target,
)


def test_build_features_has_expected_shape_and_columns(ohlcv):
    features = build_features(ohlcv)

    assert features.index.equals(ohlcv.index)
    assert len(features) == len(ohlcv)
    assert LOG_RET_1 in features
    assert {
        "log_ret_24",
        "ret_mean_24",
        "close_sma_24_ratio",
        "close_ema_24_ratio",
        "rsi_14",
        "macd",
        "macd_signal",
        "bollinger_pct_b_20",
        "bollinger_width_20",
        "volatility_24",
        "volume_zscore_24",
    }.issubset(features.columns)
    assert all(dtype == np.dtype("float64") for dtype in features.dtypes)
    assert np.isfinite(features.dropna().to_numpy()).all()


def test_features_do_not_look_ahead(ohlcv):
    cutoff = ohlcv.index[300]
    expected = build_features(ohlcv).loc[cutoff]
    changed = ohlcv.copy()
    future = changed.index > cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= 25.0
    changed.loc[future, "volume"] += 1_000_000.0

    actual = build_features(changed).loc[cutoff]

    pd.testing.assert_series_equal(actual, expected)


def test_target_is_forward_shifted_log_return(ohlcv):
    horizon = 3
    target = make_target(ohlcv, horizon=horizon)
    expected = np.log(ohlcv["close"].shift(-horizon) / ohlcv["close"])

    assert target.name == TARGET_COLUMN
    pd.testing.assert_series_equal(target, expected.rename(TARGET_COLUMN))
    assert target.tail(horizon).isna().all()
    assert target.iloc[:-horizon].notna().all()


def test_make_dataset_is_aligned_and_nan_free(ohlcv):
    X, y = make_dataset(ohlcv, horizon=2)

    assert len(X) == len(y) > 0
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y.to_numpy()).all()
    assert y.name == TARGET_COLUMN
    assert X.index[-1] == ohlcv.index[-3]


def test_latest_features_uses_final_bar_without_requiring_target(ohlcv):
    latest = latest_features(ohlcv)

    assert len(latest) == 1
    assert latest.index[0] == ohlcv.index[-1]
    assert not latest.isna().any().any()


def test_feature_config_mapping_round_trip_and_time_features(ohlcv):
    config = FeatureConfig.from_dict(
        {
            "return_lags": [1, 4],
            "return_mean_windows": [8],
            "sma_windows": [8],
            "ema_windows": [8],
            "add_time_features": True,
        }
    )
    assert FeatureConfig.from_dict(config.to_dict()) == config

    features = build_features(ohlcv, config.to_dict())
    assert {"log_ret_4", "ret_mean_8", "hour_sin", "hour_cos", "dow_sin", "dow_cos"}.issubset(
        features.columns
    )
    assert "log_ret_2" not in features


def test_forward_log_return_matches_one_step_target(ohlcv):
    direct = forward_log_return(ohlcv["close"], horizon=1)
    pd.testing.assert_series_equal(direct, make_target(ohlcv, horizon=1))


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon_is_rejected(ohlcv, horizon):
    with pytest.raises(ValueError, match="horizon"):
        make_target(ohlcv, horizon=horizon)


def test_invalid_feature_config_is_rejected():
    with pytest.raises(ValueError, match="contain 1"):
        FeatureConfig(return_lags=(2,))
    with pytest.raises(ValueError, match="unknown"):
        FeatureConfig.from_dict({"future_window": 5})
