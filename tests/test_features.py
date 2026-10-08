import numpy as np
import pandas as pd
import pytest

from cryptopredict.core.types import LOG_RET_1, TARGET_COLUMN
from cryptopredict.features import (
    FeatureConfig,
    build_features,
    forward_log_return,
    infer_bar,
    latest_features,
    make_dataset,
    make_target,
    segment_ids,
)
from cryptopredict.features.indicators import _segment_features

HOUR = pd.Timedelta(hours=1)
GAP_AT = 300  # position of the candle removed from the fixture


def _with_gap(ohlcv, position=GAP_AT):
    """The fixture with one candle missing, like a Binance maintenance outage."""
    return ohlcv.drop(ohlcv.index[position])


def _warmup_rows(ohlcv, config=None):
    """Leading rows a gap-free series needs before its features are complete."""
    complete = build_features(ohlcv, config).notna().all(axis=1).to_numpy()
    return int(np.argmax(complete))


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


def test_infer_bar_and_segment_ids(ohlcv):
    assert infer_bar(ohlcv.index) == HOUR
    assert infer_bar(ohlcv.index[:1]) is None
    assert (segment_ids(ohlcv.index) == 0).all()

    gapped = _with_gap(ohlcv)
    segments = segment_ids(gapped.index)
    assert infer_bar(gapped.index) == HOUR
    assert (segments[:GAP_AT] == 0).all() and (segments[GAP_AT:] == 1).all()
    # An explicit bar longer than the outage treats the series as contiguous.
    assert (segment_ids(gapped.index, bar="2h") == 0).all()

    with pytest.raises(ValueError, match="bar"):
        segment_ids(ohlcv.index, bar=pd.Timedelta(0))
    with pytest.raises(ValueError, match="bar"):
        segment_ids(ohlcv.index, bar="soon")


@pytest.mark.parametrize("horizon", [1, 3])
def test_target_does_not_span_gap(ohlcv, horizon):
    gapped = _with_gap(ohlcv)
    target = make_target(gapped, horizon=horizon)
    close = gapped["close"]
    naive = np.log(close.shift(-horizon) / close)

    spanning = np.arange(GAP_AT - horizon, GAP_AT)
    assert target.iloc[spanning].isna().all()
    assert naive.iloc[spanning].notna().all()  # the bug: these were (h+1)-hour returns
    keep = np.setdiff1d(np.arange(len(gapped) - horizon), spanning)
    np.testing.assert_array_equal(target.iloc[keep].to_numpy(), naive.iloc[keep].to_numpy())

    X, y = make_dataset(gapped, horizon=horizon)
    assert not X.index.isin(gapped.index[spanning]).any()
    assert ((gapped.index.to_series().shift(-horizon) - gapped.index.to_series())
            .reindex(y.index) == horizon * HOUR).all()


def test_features_after_gap_restart_their_warmup(ohlcv):
    gapped = _with_gap(ohlcv)
    before, after = gapped.iloc[:GAP_AT], gapped.iloc[GAP_AT:]
    features = build_features(gapped)

    # Each side of the gap is computed as if it were its own series.
    pd.testing.assert_frame_equal(features.iloc[:GAP_AT], build_features(before))
    pd.testing.assert_frame_equal(features.iloc[GAP_AT:], build_features(after))

    warmup = _warmup_rows(ohlcv)
    assert warmup > 24
    X, _ = make_dataset(gapped, horizon=1)
    assert not X.index.isin(after.index[:warmup]).any()
    assert after.index[warmup] in X.index

    # Every kept log_ret_1 is a genuine one-bar return.
    previous = gapped.index.to_series().shift(1).reindex(X.index)
    assert ((X.index.to_series() - previous) == HOUR).all()


def test_gap_free_features_match_single_segment(ohlcv):
    cfg = FeatureConfig(add_time_features=True)
    pd.testing.assert_frame_equal(build_features(ohlcv, cfg), _segment_features(ohlcv, cfg))
    pd.testing.assert_frame_equal(build_features(ohlcv, cfg, bar="1h"), build_features(ohlcv, cfg))


def test_latest_features_rejects_bar_right_after_gap(ohlcv):
    warmup = _warmup_rows(ohlcv)
    near_end = _with_gap(ohlcv, position=len(ohlcv) - 5)
    with pytest.raises(ValueError, match="after a candle gap"):
        latest_features(near_end)

    far_back = _with_gap(ohlcv, position=len(ohlcv) - warmup - 5)
    latest = latest_features(far_back)
    assert latest.index[0] == far_back.index[-1]
    assert not latest.isna().any().any()


def test_flat_market_gets_neutral_features():
    index = pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC", name="open_time")
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 5.0},
        index=index,
    )
    X, y = make_dataset(flat, horizon=1)

    assert len(X) > 0
    assert (X["bollinger_pct_b_20"] == 0.5).all()
    assert (X["bollinger_width_20"] == 0.0).all()
    assert (X["volume_zscore_24"] == 0.0).all()
    assert (X["rsi_14"] == 50.0).all()
    assert (y == 0.0).all()


def test_forward_log_return_bar_needs_datetime_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        forward_log_return(pd.Series([1.0, 2.0, 3.0]), horizon=1, bar="1h")
