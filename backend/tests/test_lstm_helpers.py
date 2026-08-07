import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from core.lstm_helpers import (
    add_scaled_targets,
    append_window,
    build_sector_data,
    finalise_split,
    inverse_scale_predictions,
    make_split_parts,
    regression_metrics,
    sector_slug,
)


def test_sector_slug_lowercases_and_joins_with_underscores():
    assert sector_slug("Technology and Communication Services") == "technology_and_communication_services"


def test_sector_slug_strips_punctuation():
    assert sector_slug("Consumer Staples & Discretionary!") == "consumer_staples_discretionary"


def test_inverse_scale_predictions_reverses_min_max_scaling():
    scaler = MinMaxScaler()
    original = np.array([-0.05, 0.0, 0.1]).reshape(-1, 1)
    scaler.fit(original)
    scaled = scaler.transform(original).flatten()

    result = inverse_scale_predictions(scaler, scaled)

    np.testing.assert_allclose(result, original.flatten(), atol=1e-8)


def test_regression_metrics_zero_for_perfect_predictions():
    current_close = np.array([100.0, 200.0])
    target_close = np.array([105.0, 190.0])
    y_pred_return = (target_close / current_close) - 1

    metrics = regression_metrics(y_pred_return, current_close, target_close)

    assert metrics["mae"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["mape"] == pytest.approx(0.0, abs=1e-8)


def test_regression_metrics_known_error():
    current_close = np.array([100.0])
    target_close = np.array([110.0])
    # Predict no change, so predicted close stays at 100 vs an actual close of 110.
    y_pred_return = np.array([0.0])

    metrics = regression_metrics(y_pred_return, current_close, target_close)

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["mape"] == pytest.approx(10 / 110)


def test_make_split_parts_returns_empty_lists_for_all_keys():
    parts = make_split_parts()

    assert set(parts.keys()) == {
        "X",
        "y",
        "current_close",
        "target_close",
        "ticker",
        "prediction_date",
        "target_date",
    }
    assert all(value == [] for value in parts.values())


def test_append_window_computes_return_and_slices_correctly():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    closes = np.array([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 20.0])
    scaled = np.arange(20, dtype=float).reshape(10, 2)

    parts = make_split_parts()
    append_window(
        parts,
        scaled,
        closes,
        dates,
        ticker="TEST",
        end_position=4,
        window_size=3,
        forecast_horizon=2,
    )

    # window_size=3 ending at position 4 covers positions 2, 3, 4.
    np.testing.assert_array_equal(parts["X"][0], scaled[2:5])
    assert parts["current_close"][0] == pytest.approx(14.0)
    assert parts["target_close"][0] == pytest.approx(16.0)
    assert parts["y"][0] == pytest.approx((16.0 / 14.0) - 1)
    assert parts["ticker"][0] == "TEST"
    assert parts["prediction_date"][0] == dates[4]
    assert parts["target_date"][0] == dates[6]


def test_finalise_split_builds_arrays_with_correct_shapes():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    closes = np.array([10.0, 11, 12, 13, 14, 15])
    scaled = np.arange(12, dtype=float).reshape(6, 2)

    parts = make_split_parts()
    append_window(parts, scaled, closes, dates, "TEST", end_position=2, window_size=2, forecast_horizon=1)
    append_window(parts, scaled, closes, dates, "TEST", end_position=3, window_size=2, forecast_horizon=1)

    split = finalise_split(parts, window_size=2, feature_cols=["a", "b"])

    assert split["X"].shape == (2, 2, 2)
    assert split["y"].shape == (2,)
    assert list(split["metadata"].columns) == ["ticker", "prediction_date", "target_date"]


def test_finalise_split_handles_empty_input():
    parts = make_split_parts()

    split = finalise_split(parts, window_size=3, feature_cols=["a", "b", "c"])

    assert split["X"].shape == (0, 3, 3)
    assert split["y"].shape == (0,)
    assert split["metadata"].empty


def test_add_scaled_targets_scales_within_zero_one():
    y = np.array([0.0, 0.05, -0.05], dtype=np.float32)
    split = {"y": y}

    scaler = MinMaxScaler()
    scaler.fit(y.reshape(-1, 1))

    result = add_scaled_targets(split, scaler)

    assert result["y_scaled"].min() >= 0.0
    assert result["y_scaled"].max() <= 1.0


def test_add_scaled_targets_handles_empty_split():
    split = {"y": np.empty((0,), dtype=np.float32)}
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0.0], [1.0]]))

    result = add_scaled_targets(split, scaler)

    assert result["y_scaled"].shape == (0,)


def _make_ticker_df(start: str, periods: int, base_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    prices = base_price + np.arange(periods, dtype=float)
    return pd.DataFrame({"feature_a": prices, "Close": prices}, index=dates)


def test_build_sector_data_splits_by_date_ranges():
    df = _make_ticker_df("2022-01-03", periods=300)

    dataset = build_sector_data(
        sector_tickers=["TEST"],
        ticker_features={"TEST": df},
        feature_cols=["feature_a"],
        train_start="2022-01-03",
        train_model_end="2022-06-01",
        validation_start="2022-06-02",
        validation_end="2022-09-01",
        test_start="2022-09-02",
        test_end=str(df.index[-1].date()),
        window_size=5,
        forecast_horizon=2,
    )

    assert dataset is not None
    assert len(dataset["train"]["y"]) > 0
    assert len(dataset["validation"]["y"]) > 0
    assert len(dataset["test"]["y"]) > 0
    assert dataset["train"]["metadata"]["prediction_date"].max() <= pd.Timestamp("2022-06-01")


def test_build_sector_data_returns_none_when_no_tickers_available():
    dataset = build_sector_data(
        sector_tickers=["MISSING"],
        ticker_features={},
        feature_cols=["feature_a"],
        train_start="2022-01-01",
        train_model_end="2022-06-01",
        validation_start="2022-06-02",
        validation_end="2022-09-01",
        test_start="2022-09-02",
        test_end="2022-12-01",
        window_size=5,
        forecast_horizon=2,
    )

    assert dataset is None
