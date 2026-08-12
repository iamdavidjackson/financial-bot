"""Train the pooled all-tickers LSTM return-regression model.

Extracted from notebooks/04_lstm_all_tickers.ipynb: trains one LSTM across every
ticker in core/tickers.py and saves the model plus the scalers it was trained with.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from core.fetch_data import SELECTED_FEATURE_COLS, get_ticker_features
from core.lstm_helpers import (
    build_lstm_model,
    build_sector_data,
    inverse_scale_predictions,
    regression_metrics,
)
from core.tickers import TICKERS

# Anchor the train/validation/test windows to today instead of fixed calendar
# dates, so each run trains on the most recent data available. Window sizes match
# what the notebook used: 2yr train, 1yr validation, 1yr test, ~3mo feature warmup.
TODAY = pd.Timestamp.today().normalize()

# The last test-period row still needs a completed 5-trading-day-ahead target, so
# TEST_END has to sit far enough in the past for that target date to have data.
TEST_END_BUFFER_DAYS = 10

DOWNLOAD_END = (TODAY + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TEST_END = (TODAY - pd.Timedelta(days=TEST_END_BUFFER_DAYS)).strftime("%Y-%m-%d")
TEST_START = (pd.Timestamp(TEST_END) - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
VALIDATION_END = (pd.Timestamp(TEST_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
VALIDATION_START = (pd.Timestamp(VALIDATION_END) - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TRAIN_MODEL_END = (pd.Timestamp(VALIDATION_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
TRAIN_START = (pd.Timestamp(TRAIN_MODEL_END) - pd.DateOffset(years=2) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
# Give moving averages like sma50, plus the 60-trading-day rolling z-score window
# core.fetch_data.rolling_zscore needs on top of that, enough warm-up data before
# the training window starts.
FEATURE_START = (pd.Timestamp(TRAIN_START) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")

# Use a 30-day input window to predict the 5-trading-day return.
FORECAST_HORIZON = 5
WINDOW_SIZE = 30

# These settings control the model training loop.
MAX_EPOCHS = 30
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

# Use every ticker from tickers.py once.
FEATURE_COLS = list(dict.fromkeys(SELECTED_FEATURE_COLS))
ALL_TICKERS = list(dict.fromkeys(TICKERS))

# Save the pooled model separately from the single-ticker and sector models.
MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "lstm_all_tickers_return_regression"
MODEL_PATH = MODEL_DIR / f"all_tickers_{FORECAST_HORIZON}d_return.keras"
SCALERS_PATH = MODEL_DIR / f"all_tickers_{FORECAST_HORIZON}d_return.scalers.joblib"


def load_ticker_features() -> dict[str, pd.DataFrame]:
    """Download and compute features for every ticker with enough history."""
    ticker_features = {}
    required_rows = WINDOW_SIZE + FORECAST_HORIZON + 1

    for ticker in ALL_TICKERS:
        df = get_ticker_features(
            ticker=ticker,
            start=FEATURE_START,
            end=DOWNLOAD_END,
            smooth_outliers=True,
        )

        # Skip tickers that cannot create one complete sequence and one future target.
        if df.empty or len(df) < required_rows:
            print(f"{ticker}: skipped (only {len(df)} rows)")
            continue

        ticker_features[ticker] = df
        print(f"{ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")

    print(f"Loaded {len(ticker_features)} of {len(ALL_TICKERS)} requested tickers.")
    return ticker_features


def train() -> pd.DataFrame:
    ticker_features = load_ticker_features()

    # Build one pooled dataset using every loaded ticker.
    dataset = build_sector_data(
        sector_tickers=ALL_TICKERS,
        ticker_features=ticker_features,
        feature_cols=FEATURE_COLS,
        train_start=TRAIN_START,
        train_model_end=TRAIN_MODEL_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        test_start=TEST_START,
        test_end=TEST_END,
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
    )
    if dataset is None:
        raise ValueError("No valid pooled dataset could be built from the available tickers.")

    print(f"Training samples: {len(dataset['train']['y'])}")
    print(f"Validation samples: {len(dataset['validation']['y'])}")
    print(f"Test samples: {len(dataset['test']['y'])}")
    print(f"Tickers used: {len(dataset['tickers'])}")

    X_train = dataset["train"]["X"]
    y_train = dataset["train"]["y_scaled"]
    X_validation = dataset["validation"]["X"]
    y_validation = dataset["validation"]["y_scaled"]
    X_test = dataset["test"]["X"]
    y_test = dataset["test"]["y"]
    current_close_test = dataset["test"]["current_close"]
    target_close_test = dataset["test"]["target_close"]

    # Reset the model state so this run is independent of earlier runs.
    keras.backend.clear_session()
    keras.utils.set_random_seed(RANDOM_SEED)

    model = build_lstm_model((WINDOW_SIZE, len(FEATURE_COLS)))

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    # Validation data controls early stopping. The test data is only used after training.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Convert scaled return predictions back before calculating price metrics.
    y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred = inverse_scale_predictions(dataset["target_scaler"], y_pred_scaled)
    naive_pred = np.zeros_like(y_test)

    model_metrics = regression_metrics(y_pred, current_close_test, target_close_test)
    naive_metrics = regression_metrics(naive_pred, current_close_test, target_close_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump((dataset["feature_scaler"], dataset["target_scaler"]), SCALERS_PATH)

    # A positive difference means the pooled LSTM beat the zero-return baseline.
    result = {
        "model": "All tickers LSTM",
        "forecast_horizon": f"{FORECAST_HORIZON}d",
        "tickers": len(dataset["tickers"]),
        "train_samples": len(dataset["train"]["y"]),
        "validation_samples": len(dataset["validation"]["y"]),
        "test_samples": len(y_test),
        "naive_mae": naive_metrics["mae"],
        "mae": model_metrics["mae"],
        "mae_difference": naive_metrics["mae"] - model_metrics["mae"],
        "naive_rmse": naive_metrics["rmse"],
        "rmse": model_metrics["rmse"],
        "rmse_difference": naive_metrics["rmse"] - model_metrics["rmse"],
        "naive_mape": naive_metrics["mape"],
        "mape": model_metrics["mape"],
        "mape_difference": naive_metrics["mape"] - model_metrics["mape"],
        "epochs": len(history.history["loss"]),
        "model_path": str(MODEL_PATH),
        "scalers_path": str(SCALERS_PATH),
    }
    result["beats_naive"] = result["mae_difference"] > 0

    results_df = pd.DataFrame([result])
    metric_cols = [
        "naive_mae",
        "mae",
        "mae_difference",
        "naive_rmse",
        "rmse",
        "rmse_difference",
        "naive_mape",
        "mape",
        "mape_difference",
    ]
    results_df[metric_cols] = results_df[metric_cols].round(4)
    return results_df


if __name__ == "__main__":
    print(train().to_string(index=False))
