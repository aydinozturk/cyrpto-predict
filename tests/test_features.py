import numpy as np
import pandas as pd
import pytest

from cryptopredict.core.types import LOG_RET_1, TARGET_COLUMN
from cryptopredict.core.types import INDEX_NAME
from cryptopredict.features import (
    FeatureConfig,
    active_htf_intervals,
    build_features,
    forward_log_return,
    infer_bar,
    latest_features,
    make_dataset,
    make_target,
    required_history,
    segment_ids,
)
from cryptopredict.features.indicators import _segment_features

HOUR = pd.Timedelta(hours=1)
GAP_AT = 300  # position of the candle removed from the fixture
# Without HTF features the warm-up is short enough for both sides of GAP_AT.
NO_HTF = FeatureConfig(htf_intervals=())

# Columns of the default config before #44, with (non-NaN count, sum, last value)
# on the fixture, computed by that code.  New indicators must not change them.
LEGACY_COLUMNS = {
    "log_ret_1": (599, 0.06673481423154826, 0.0044107122049812375),
    "log_ret_2": (598, 0.12816668764900463, 0.006089271694184717),
    "log_ret_3": (597, 0.18706859614107962, 0.011210127882165466),
    "log_ret_6": (594, 0.3549851320312598, 0.024577759860914483),
    "log_ret_12": (588, 0.4749941612094819, 0.05425681407933425),
    "log_ret_24": (576, 0.7111085981248184, 0.06201770041003307),
    "ret_mean_24": (576, 0.029629524921867427, 0.0025840708504180445),
    "close_sma_6_ratio": (595, 0.16108770598609434, 0.008712395798683659),
    "close_sma_12_ratio": (589, 0.2899741856783473, 0.027959349425557667),
    "close_sma_24_ratio": (577, 0.5036473426010638, 0.04261046755000564),
    "close_ema_6_ratio": (595, 0.1419181612555921, 0.010698006098109447),
    "close_ema_12_ratio": (589, 0.24122459484841474, 0.02253034080614502),
    "close_ema_24_ratio": (577, 0.4277559282085439, 0.03577144513231012),
    "rsi_14": (586, 29633.591048480972, 85.19914467011728),
    "macd": (575, 13066.706714558008, 1188.5421693726967),
    "macd_signal": (567, 12330.31007194481, 777.6196811471033),
    "bollinger_pct_b_20": (581, 299.61051442141616, 0.9626403902758323),
    "bollinger_width_20": (581, 12.758825810214859, 0.08539890888321079),
    "volatility_24": (576, 1.882819861652214, 0.005820972457669655),
    "volume_zscore_24": (577, 35.659871583945865, 1.1297164498089347),
}
NEW_COLUMNS = [
    "atr_14",
    "stoch_k_14",
    "stoch_d_14",
    "obv_zscore_24",
    "signed_volume_ratio_24",
    "parkinson_vol_24",
    "garman_klass_vol_24",
    "candle_body",
    "candle_upper_wick",
    "candle_lower_wick",
    "candle_log_range",
    "ret_skew_72",
    "ret_kurt_72",
]
HTF_SUFFIXES = ["log_ret_1", "log_ret_3", "rsi_14", "close_ema_14_ratio"]


def _synthetic(periods, freq="1h", seed=0, start="2026-01-01"):
    """A random-walk OHLCV frame, long enough for daily features."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name=INDEX_NAME)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    open_ = np.concatenate([[100.0], close[:-1]])
    spread = np.abs(rng.normal(0.0, 0.004, (2, periods)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + spread[0]),
            "low": np.minimum(open_, close) * (1.0 - spread[1]),
            "close": close,
            "volume": rng.uniform(1.0, 50.0, periods),
        },
        index=index,
    )


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

    warmup = _warmup_rows(ohlcv, NO_HTF)
    assert warmup > 24
    X, _ = make_dataset(gapped, horizon=1, config=NO_HTF)
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

    # Daily candles restart after the gap too; the warm-up bound covers any alignment.
    assert warmup <= required_history(None, "1h")
    far_back = _with_gap(ohlcv, position=len(ohlcv) - required_history(None, "1h") - 5)
    latest = latest_features(far_back)
    assert latest.index[0] == far_back.index[-1]
    assert not latest.isna().any().any()


def test_flat_market_gets_neutral_features():
    index = pd.date_range("2026-01-01", periods=24 * 20, freq="1h", tz="UTC", name="open_time")
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
    assert (X[["stoch_k_14", "stoch_d_14", "htf_4h_rsi_14", "htf_1d_rsi_14"]] == 50.0).all(axis=None)
    zero = [c for c in X if c.startswith(("atr", "obv", "signed", "parkinson", "garman", "candle", "ret_skew", "ret_kurt"))]
    zero += [f"htf_{name}_{suffix}" for name in ("4h", "1d") for suffix in HTF_SUFFIXES if "rsi" not in suffix]
    assert len(zero) == len(NEW_COLUMNS) - 2 + 6
    assert (X[zero] == 0.0).all(axis=None)
    assert (y == 0.0).all()


def test_forward_log_return_bar_needs_datetime_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        forward_log_return(pd.Series([1.0, 2.0, 3.0]), horizon=1, bar="1h")


def test_legacy_default_columns_are_unchanged(ohlcv):
    features = build_features(ohlcv)

    assert list(features.columns[: len(LEGACY_COLUMNS)]) == list(LEGACY_COLUMNS)
    for column, (count, total, last) in LEGACY_COLUMNS.items():
        values = features[column]
        assert values.notna().sum() == count, column
        assert values.sum() == pytest.approx(total, rel=1e-12, abs=1e-12), column
        assert values.iloc[-1] == pytest.approx(last, rel=1e-12, abs=1e-15), column


def test_new_default_columns_are_present_and_finite(ohlcv):
    features = build_features(ohlcv)
    htf = [f"htf_{name}_{suffix}" for name in ("4h", "1d") for suffix in HTF_SUFFIXES]

    assert list(features.columns) == list(LEGACY_COLUMNS) + NEW_COLUMNS + htf
    assert all(dtype == np.dtype("float64") for dtype in features.dtypes)
    assert not np.isinf(features.to_numpy()).any()
    assert features.iloc[-1].notna().all()
    assert ((features["stoch_k_14"].dropna() >= 0.0) & (features["stoch_k_14"].dropna() <= 100.0)).all()
    wicks = features[["candle_upper_wick", "candle_lower_wick"]]
    assert ((wicks >= 0.0) & (wicks <= 1.0)).all(axis=None)
    assert (features["candle_body"].abs() + wicks.sum(axis=1)).round(12).eq(1.0).all()


@pytest.mark.parametrize("position", [100, 297, 298, 299, 300, 450, 598])
def test_all_features_are_causal(position):
    """No column at or before bar t depends on later bars (HTF included)."""
    ohlcv = _synthetic(24 * 30)
    cutoff = ohlcv.index[position]
    expected = build_features(ohlcv).loc[:cutoff]
    changed = ohlcv.copy()
    future = changed.index > cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= 25.0
    changed.loc[future, "volume"] += 1_000_000.0

    pd.testing.assert_frame_equal(build_features(changed).loc[:cutoff], expected)
    # Data ending at t (a live prediction) yields the same rows.
    pd.testing.assert_frame_equal(build_features(ohlcv.loc[:cutoff]), expected)


def test_htf_candle_is_used_only_after_it_closes():
    ohlcv = _synthetic(24 * 30)
    features = build_features(ohlcv)
    log_close = np.log(ohlcv["close"])
    day = pd.Timestamp("2026-01-20", tz="UTC")

    # The 4h candle [08:00, 12:00) is bars 08..11 and closes at 12:00, the close of bar 11:00.
    candle_return = log_close[day + pd.Timedelta(hours=11)] - log_close[day + pd.Timedelta(hours=7)]
    column = features["htf_4h_log_ret_1"]
    assert column[day + pd.Timedelta(hours=10)] != pytest.approx(candle_return)
    for hour in (11, 12, 13, 14):
        assert column[day + pd.Timedelta(hours=hour)] == pytest.approx(candle_return)
    assert column[day + pd.Timedelta(hours=15)] != pytest.approx(candle_return)

    # The last bar of the candle moves the feature only from its own row on.
    last_bar = day + pd.Timedelta(hours=11)
    bumped = ohlcv.copy()
    bumped.loc[last_bar, ["high", "close"]] *= 1.05
    changed = build_features(bumped)["htf_4h_log_ret_1"]
    assert changed[: last_bar - HOUR].equals(column[: last_bar - HOUR])
    assert changed[last_bar] != column[last_bar]

    # The daily candle of the 20th is seen from bar 23:00 (closing at midnight) on.
    daily = log_close[day + pd.Timedelta(hours=23)] - log_close[day - HOUR]
    assert features["htf_1d_log_ret_1"][day + pd.Timedelta(hours=22)] != pytest.approx(daily)
    assert features["htf_1d_log_ret_1"][day + pd.Timedelta(hours=23)] == pytest.approx(daily)


def test_htf_matches_resampled_closed_candles():
    ohlcv = _synthetic(24 * 30).iloc[5:]  # start mid-candle: the first 4h candle is partial
    features = build_features(ohlcv)
    close = ohlcv["close"].resample("4h").last()
    complete = ohlcv["close"].resample("4h").count() == 4
    closed = close[complete]
    expected = np.log(closed).diff()
    expected.index = expected.index + pd.Timedelta(hours=3)
    expected = expected.reindex(ohlcv.index, method="ffill")

    pd.testing.assert_series_equal(features["htf_4h_log_ret_1"], expected, check_names=False)
    assert features["htf_4h_log_ret_1"].first_valid_index() == pd.Timestamp("2026-01-01 15:00", tz="UTC")


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("15min", ["4h"]), ("1h", ["4h", "1d"]), ("4h", ["1d"]), ("1D", [])],
)
def test_htf_is_skipped_for_coarse_bars_and_long_ratios(freq, expected):
    ohlcv = _synthetic(24 * 30, freq=freq)
    features = build_features(ohlcv)

    assert [name for name, _ in active_htf_intervals(None, pd.Timedelta(freq))] == expected
    used = sorted({c.split("_")[1] for c in features if c.startswith("htf_")})
    assert used == sorted(expected)
    assert features.iloc[-1].notna().all()
    assert active_htf_intervals(None, None) == []


def test_htf_features_respect_gap_segments():
    ohlcv = _synthetic(24 * 40)
    position = 24 * 20 + 2  # mid-candle on both timeframes
    gapped = ohlcv.drop(ohlcv.index[position])
    before, after = gapped.iloc[:position], gapped.iloc[position:]
    features = build_features(gapped)

    pd.testing.assert_frame_equal(features.iloc[:position], build_features(before))
    pd.testing.assert_frame_equal(features.iloc[position:], build_features(after))
    assert not np.isinf(features.to_numpy()).any()
    # No daily candle spans the gap: the first one after it starts the next midnight.
    first = features.iloc[position:]["htf_1d_log_ret_1"].first_valid_index()
    assert first == pd.Timestamp("2026-01-23 23:00", tz="UTC")


def test_required_history_is_enough_for_a_live_row(ohlcv):
    needed = required_history(None, "1h")
    assert needed == 16 * 24
    latest = latest_features(ohlcv.tail(needed))
    assert latest.notna().all(axis=None)
    with pytest.raises(ValueError, match="not enough history"):
        latest_features(ohlcv.tail(needed - 30))

    assert required_history(NO_HTF, "1h") == 73
    assert required_history(None, "15min") == 16 * 16  # no daily features below 1h
    synthetic = _synthetic(4 * 24 * 10, freq="15min", start="2026-01-01 00:45")
    assert latest_features(synthetic.tail(required_history(None, "15min"))).notna().all(axis=None)


def test_latest_features_can_check_only_model_columns(ohlcv):
    near_gap = _with_gap(ohlcv, position=len(ohlcv) - 100)  # long enough for legacy columns only
    with pytest.raises(ValueError, match="after a candle gap"):
        latest_features(near_gap)

    latest = latest_features(near_gap, columns=list(LEGACY_COLUMNS))
    assert list(latest.columns) == list(LEGACY_COLUMNS)
    assert latest.notna().all(axis=None)
    with pytest.raises(ValueError, match="lacks feature column"):
        latest_features(ohlcv, columns=["log_ret_1", "future_magic"])


def test_new_config_fields_round_trip_and_legacy_dicts_get_defaults():
    config = FeatureConfig(
        atr_window=7, stoch_window=5, obv_window=12, range_vol_window=10,
        add_candle_features=False, moment_window=24, htf_intervals=("2h",), htf_window=6,
    )
    data = config.to_dict()
    assert data["htf_intervals"] == ["2h"]
    assert FeatureConfig.from_dict(data) == config

    legacy = {key: value for key, value in FeatureConfig().to_dict().items() if key in LEGACY_KEYS}
    assert FeatureConfig.from_dict(legacy) == FeatureConfig()

    features = build_features(_synthetic(200), config)
    assert {"atr_7", "stoch_k_5", "obv_zscore_12", "parkinson_vol_10", "ret_kurt_24", "htf_2h_rsi_6"} <= set(features)
    assert not any(c.startswith("candle_") for c in features)


LEGACY_KEYS = {
    "return_lags", "return_mean_windows", "sma_windows", "ema_windows", "rsi_window",
    "macd_fast", "macd_slow", "macd_signal", "bollinger_window", "bollinger_std",
    "volatility_window", "volume_zscore_window", "add_time_features",
}


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"htf_intervals": ["1w"]}, "not an interval"),
        ({"htf_intervals": ["7h"]}, "divide one day"),
        ({"htf_intervals": ["60m", "1h"]}, "duplicates"),
        ({"htf_intervals": "4h"}, "must be a list"),
        ({"htf_intervals": [4]}, "interval strings"),
        ({"moment_window": 3}, "at least 4"),
        ({"atr_window": 0}, "positive integer"),
        ({"add_candle_features": 1}, "boolean"),
    ],
)
def test_invalid_new_config_values_are_rejected(values, match):
    with pytest.raises(ValueError, match=match):
        FeatureConfig.from_dict(values)
