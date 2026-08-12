import numpy as np
import pandas as pd
import pytest

from core.fetch_data import (
    OHLCV_COLS,
    SELECTED_FEATURE_COLS,
    compute_technical_features,
    fibonacci_weighted_moving_average,
    holt_winter_moving_average,
    hull_moving_average,
    rolling_zscore,
    smooth_ohlcv_outliers,
    triple_exponential_moving_average,
    verify_ohlcv_rows,
    weighted_moving_average,
)


def test_weighted_moving_average_matches_manual_calculation():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = weighted_moving_average(series, window=3)

    # Default weights are [1, 2, 3] (most recent day weighted highest), normalized.
    assert result.iloc[4] == pytest.approx((3 * 1 + 4 * 2 + 5 * 3) / 6)
    assert result.iloc[:2].isna().all()


def test_weighted_moving_average_with_custom_weights():
    series = pd.Series([10.0, 20.0, 30.0])
    weights = np.array([1.0, 1.0])  # equal weights behave like a simple average

    result = weighted_moving_average(series, window=2, weights=weights)

    assert result.iloc[1] == pytest.approx(15.0)
    assert result.iloc[2] == pytest.approx(25.0)


def test_triple_exponential_moving_average_tracks_a_flat_series():
    series = pd.Series([100.0] * 10)

    result = triple_exponential_moving_average(series, window=3)

    assert result.iloc[-1] == pytest.approx(100.0)


def test_fibonacci_weighted_moving_average_weights_most_recent_day_highest():
    # Fibonacci weights favor the most recent day, so for a rising series the
    # result should sit above the plain average.
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    result = fibonacci_weighted_moving_average(series, window=5)

    assert result.iloc[-1] > series.mean()


def test_holt_winter_moving_average_returns_flat_series_for_flat_input():
    series = pd.Series([50.0] * 15)

    result = holt_winter_moving_average(series)

    assert result.iloc[-1] == pytest.approx(50.0, abs=1e-6)


def test_holt_winter_moving_average_handles_empty_series():
    series = pd.Series([], dtype=float)

    result = holt_winter_moving_average(series)

    assert result.empty


def test_hull_moving_average_settles_near_a_flat_series_value():
    series = pd.Series([42.0] * 30)

    result = hull_moving_average(series, window=8)

    assert result.iloc[-1] == pytest.approx(42.0, abs=1e-6)


def _valid_ohlcv_row(**overrides):
    row = {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5, "Volume": 1000.0}
    row.update(overrides)
    return row


def test_verify_ohlcv_rows_flags_non_positive_values():
    data = pd.DataFrame([_valid_ohlcv_row(), _valid_ohlcv_row(Volume=0.0)])

    checks = verify_ohlcv_rows(data)

    assert not checks["has_issue"].iloc[0]
    assert checks["has_issue"].iloc[1]
    assert checks["non_positive_ohlcv_values"].iloc[1] == 1


def test_verify_ohlcv_rows_flags_invalid_price_range():
    # High below Close should never happen for valid OHLCV data.
    data = pd.DataFrame([_valid_ohlcv_row(High=9.0, Close=10.5)])

    checks = verify_ohlcv_rows(data)

    assert checks["invalid_price_range"].iloc[0]
    assert checks["has_issue"].iloc[0]


def test_verify_ohlcv_rows_raises_for_missing_columns():
    data = pd.DataFrame([{"Open": 10.0}])

    with pytest.raises(ValueError):
        verify_ohlcv_rows(data)


def test_smooth_ohlcv_outliers_replaces_extreme_spike():
    values = [100.0] * 10
    values[5] = 100_000.0  # an obvious data error
    data = pd.DataFrame({col: list(values) for col in OHLCV_COLS})

    smoothed, report = smooth_ohlcv_outliers(data)

    assert smoothed["Close"].iloc[5] < 1000.0
    assert report.loc[report["column"] == "Close", "outlier_count"].iloc[0] == 1


def _synthetic_ohlcv(periods: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, size=periods))
    # Real trading volume is never perfectly flat day to day. A constant series
    # here would give rolling_zscore a zero-variance window (see rolling_zscore's
    # NaN-on-zero-std guard), which would drop every row once Volume_z joins
    # SELECTED_FEATURE_COLS.
    volume = 1_000_000.0 + rng.normal(0, 50_000.0, size=periods)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def test_compute_technical_features_produces_all_selected_columns():
    data = _synthetic_ohlcv()

    features = compute_technical_features(data)

    assert not features.empty
    for col in SELECTED_FEATURE_COLS:
        assert col in features.columns
    assert not features[SELECTED_FEATURE_COLS].isna().any().any()
    # The raw Close column must survive untouched: targets and portfolio
    # prices need real dollar values, not the normalized "Close_z" feature.
    assert "Close" in features.columns


def test_rolling_zscore_leaves_early_rows_nan_until_the_window_fills():
    features = pd.DataFrame({"price": [10.0] * 5 + [20.0] * 5})

    result = rolling_zscore(features, ["price"], window=5)

    assert result["price_z"].iloc[:4].isna().all()


def test_rolling_zscore_handles_a_flat_window_without_producing_inf():
    features = pd.DataFrame({"price": [10.0] * 6})

    result = rolling_zscore(features, ["price"], window=5)

    # Zero rolling variance divides out to NaN, not +/-inf.
    assert not np.isinf(result["price_z"]).any()


def test_rolling_zscore_flags_a_value_above_its_own_trailing_average():
    features = pd.DataFrame({"price": [10.0, 10.0, 10.0, 10.0, 10.0, 20.0]})

    result = rolling_zscore(features, ["price"], window=5)

    assert result["price_z"].iloc[-1] > 0


def test_rolling_zscore_puts_a_cheap_and_an_expensive_ticker_on_the_same_scale():
    # Same relative pattern (a flat run, then a steady climb), wildly different
    # absolute price levels (expensive is exactly 40x cheap throughout). Both
    # should land on the same z-score, which is the whole point: a pooled model
    # shouldn't be able to tell these two tickers apart just from the feature's
    # raw level.
    cheap = pd.DataFrame({"price": [10.0, 10.0, 10.0, 10.0, 11.0, 12.0, 13.0, 14.0]})
    expensive = pd.DataFrame({"price": [p * 40 for p in cheap["price"]]})

    cheap_z = rolling_zscore(cheap, ["price"], window=4)["price_z"].iloc[-1]
    expensive_z = rolling_zscore(expensive, ["price"], window=4)["price_z"].iloc[-1]

    assert cheap_z == pytest.approx(expensive_z, rel=1e-6)
