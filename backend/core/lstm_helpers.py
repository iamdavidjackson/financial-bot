import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler


def make_split_parts() -> dict[str, list[Any]]:
    """Create the temporary lists used while building LSTM samples."""
    return {
        "X": [],
        "y": [],
        "current_close": [],
        "target_close": [],
        "ticker": [],
        "prediction_date": [],
        "target_date": [],
    }


def append_window(
    split_parts: dict[str, list[Any]],
    scaled: np.ndarray,
    closes: np.ndarray,
    dates: pd.DatetimeIndex,
    ticker: str,
    end_position: int,
    window_size: int,
    forecast_horizon: int,
) -> None:
    """Append one LSTM input window and its future return target."""
    start_position = end_position - window_size + 1
    target_position = end_position + forecast_horizon

    current_close = float(closes[end_position])
    target_close = float(closes[target_position])
    target_return = (target_close / current_close) - 1

    split_parts["X"].append(scaled[start_position : end_position + 1])
    split_parts["y"].append(target_return)
    split_parts["current_close"].append(current_close)
    split_parts["target_close"].append(target_close)
    split_parts["ticker"].append(ticker)
    split_parts["prediction_date"].append(dates[end_position])
    split_parts["target_date"].append(dates[target_position])


def finalise_split(
    split_parts: dict[str, list[Any]],
    window_size: int,
    feature_cols: Sequence[str],
) -> dict[str, Any]:
    """Convert temporary split lists into arrays and metadata."""
    metadata_cols = ["ticker", "prediction_date", "target_date"]

    if not split_parts["X"]:
        return {
            "X": np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "current_close": np.empty((0,), dtype=np.float32),
            "target_close": np.empty((0,), dtype=np.float32),
            "metadata": pd.DataFrame(columns=metadata_cols),
        }

    metadata = pd.DataFrame(
        {
            "ticker": split_parts["ticker"],
            "prediction_date": split_parts["prediction_date"],
            "target_date": split_parts["target_date"],
        }
    )

    return {
        "X": np.asarray(split_parts["X"], dtype=np.float32),
        "y": np.asarray(split_parts["y"], dtype=np.float32),
        "current_close": np.asarray(split_parts["current_close"], dtype=np.float32),
        "target_close": np.asarray(split_parts["target_close"], dtype=np.float32),
        "metadata": metadata,
    }


def add_scaled_targets(
    split: dict[str, Any],
    target_scaler: MinMaxScaler,
) -> dict[str, Any]:
    """Add a scaled return target column to a train, validation, or test split."""
    if len(split["y"]) == 0:
        split["y_scaled"] = np.empty((0,), dtype=np.float32)
    else:
        split["y_scaled"] = (
            target_scaler.transform(split["y"].reshape(-1, 1))
            .flatten()
            .astype(np.float32)
        )

    return split


def build_sector_data(
    sector_tickers: Sequence[str],
    ticker_features: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    train_start: str | pd.Timestamp,
    train_model_end: str | pd.Timestamp,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    window_size: int,
    forecast_horizon: int,
) -> dict[str, Any] | None:
    """Build train, validation, and test LSTM datasets for a group of tickers."""
    feature_cols = list(dict.fromkeys(feature_cols))
    train_start = pd.Timestamp(train_start)
    train_model_end = pd.Timestamp(train_model_end)
    validation_start = pd.Timestamp(validation_start)
    validation_end = pd.Timestamp(validation_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    available_tickers = [ticker for ticker in sector_tickers if ticker in ticker_features]

    training_feature_rows = []
    for ticker in available_tickers:
        rows = ticker_features[ticker].loc[
            (ticker_features[ticker].index >= train_start)
            & (ticker_features[ticker].index <= train_model_end),
            feature_cols,
        ].dropna()

        if not rows.empty:
            training_feature_rows.append(rows)

    if not training_feature_rows:
        return None

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(pd.concat(training_feature_rows, ignore_index=True))

    splits = {
        "train": make_split_parts(),
        "validation": make_split_parts(),
        "test": make_split_parts(),
    }

    for ticker in available_tickers:
        required_cols = list(dict.fromkeys([*feature_cols, "Close"]))
        df = ticker_features[ticker][required_cols].dropna().copy()
        scaled = feature_scaler.transform(df[feature_cols])
        closes = df["Close"].to_numpy()
        dates = pd.DatetimeIndex(df.index)

        for end_position in range(window_size - 1, len(df) - forecast_horizon):
            prediction_date = dates[end_position]
            target_date = dates[end_position + forecast_horizon]

            if prediction_date >= train_start and target_date <= train_model_end:
                append_window(
                    splits["train"],
                    scaled,
                    closes,
                    dates,
                    ticker,
                    end_position,
                    window_size,
                    forecast_horizon,
                )
            elif prediction_date >= validation_start and target_date <= validation_end:
                append_window(
                    splits["validation"],
                    scaled,
                    closes,
                    dates,
                    ticker,
                    end_position,
                    window_size,
                    forecast_horizon,
                )
            elif prediction_date >= test_start and prediction_date <= test_end:
                append_window(
                    splits["test"],
                    scaled,
                    closes,
                    dates,
                    ticker,
                    end_position,
                    window_size,
                    forecast_horizon,
                )

    train = finalise_split(splits["train"], window_size, feature_cols)
    validation = finalise_split(splits["validation"], window_size, feature_cols)
    test = finalise_split(splits["test"], window_size, feature_cols)

    if min(len(train["y"]), len(validation["y"]), len(test["y"])) == 0:
        return None

    target_scaler = MinMaxScaler()
    target_scaler.fit(train["y"].reshape(-1, 1))

    train = add_scaled_targets(train, target_scaler)
    validation = add_scaled_targets(validation, target_scaler)
    test = add_scaled_targets(test, target_scaler)

    return {
        "tickers": available_tickers,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "train": train,
        "validation": validation,
        "test": test,
    }


def build_lstm_model(
    input_shape: tuple[int, int],
    first_lstm_units: int = 128,
    second_lstm_units: int = 64,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    learning_rate: float = 0.001,
):
    """Build the LSTM return-regression model used by the notebook."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(first_lstm_units, return_sequences=True),
            layers.Dropout(dropout_rate),
            layers.LSTM(second_lstm_units),
            layers.Dropout(dropout_rate),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    return model


def inverse_scale_predictions(
    target_scaler: MinMaxScaler,
    scaled_predictions: np.ndarray,
) -> np.ndarray:
    """Convert scaled return predictions back to normal return values."""
    return target_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).flatten()


def regression_metrics(
    y_pred_return: np.ndarray,
    current_close: np.ndarray,
    target_close: np.ndarray,
) -> dict[str, float]:
    """Calculate price-based MAE, RMSE, and MAPE from return predictions."""
    predicted_close = current_close * (1 + y_pred_return)

    return {
        "mae": mean_absolute_error(target_close, predicted_close),
        "rmse": np.sqrt(mean_squared_error(target_close, predicted_close)),
        "mape": mean_absolute_percentage_error(target_close, predicted_close),
    }


def sector_slug(sector: str) -> str:
    """Convert a sector name into a readable filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", sector.lower()).strip("_")


def predict_return_series(
    model: Any,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window_size: int,
    batch_size: int = 128,
) -> pd.Series:
    # Run the frozen model forward one date at a time, so nothing here can see future data.
    scaled = feature_scaler.transform(df[feature_cols])
    dates = pd.DatetimeIndex(df.index)

    end_positions = range(window_size - 1, len(df))
    windows = np.stack(
        [scaled[end - window_size + 1 : end + 1] for end in end_positions]
    ).astype(np.float32)

    scaled_predictions = model.predict(windows, batch_size=batch_size, verbose=0).flatten()
    predicted_returns = inverse_scale_predictions(target_scaler, scaled_predictions)

    return pd.Series(predicted_returns, index=dates[window_size - 1 :], name="predicted_return")


def convert_returns_to_signals(predicted_returns: pd.DataFrame) -> pd.DataFrame:
    # Rank each date's tickers against each other, so the signal stays between 0 and 1.
    return predicted_returns.rank(axis=1, pct=True).astype(np.float32)
