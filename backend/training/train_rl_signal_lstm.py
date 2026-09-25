import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from core.fetch_data import SELECTED_FEATURE_COLS, get_ticker_features
from core.lstm_helpers import (
    build_lstm_model,
    build_sector_data,
    inverse_scale_predictions,
    regression_metrics,
)
from core.tickers import TICKERS
from tensorflow import keras

# Add timestamps for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Fixed calendar dates
FEATURE_START = "2014-07-01"
TRAIN_START = "2015-01-01"
TRAIN_MODEL_END = "2018-12-31"
VALIDATION_START = "2019-01-01"
VALIDATION_END = "2019-12-31"
# Testing dates
TEST_START = "2020-01-01"
TEST_END = "2020-12-31"
DOWNLOAD_END = "2021-01-01"

# Sizing values
FORECAST_HORIZON = 5
WINDOW_SIZE = 30
MAX_EPOCHS = 30
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

FEATURE_COLS = list(dict.fromkeys(SELECTED_FEATURE_COLS))
ALL_TICKERS = list(dict.fromkeys(TICKERS))

# Model Paths
MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "trained_models" / "lstm_rl_signal_source"
)
MODEL_PATH = MODEL_DIR / f"rl_signal_source_{FORECAST_HORIZON}d_return.keras"
SCALERS_PATH = MODEL_DIR / f"rl_signal_source_{FORECAST_HORIZON}d_return.scalers.joblib"


def load_ticker_features() -> dict[str, pd.DataFrame]:
    # Downloads and computes features for every ticker with enough history.
    ticker_features = {}
    required_rows = WINDOW_SIZE + FORECAST_HORIZON + 1

    for ticker_index, ticker in enumerate(ALL_TICKERS, start=1):
        df = get_ticker_features(
            ticker=ticker,
            start=FEATURE_START,
            end=DOWNLOAD_END,
            smooth_outliers=True,
        )

        if df.empty or len(df) < required_rows:
            logger.info(
                "[%d/%d] %s: skipped (only %d rows)",
                ticker_index,
                len(ALL_TICKERS),
                ticker,
                len(df),
            )
            continue

        ticker_features[ticker] = df
        logger.info(
            "[%d/%d] %s: %d rows (%s to %s)",
            ticker_index,
            len(ALL_TICKERS),
            ticker,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )

    logger.info(
        "Loaded %d of %d requested tickers.", len(ticker_features), len(ALL_TICKERS)
    )
    return ticker_features


class EpochLoggingCallback(keras.callbacks.Callback):
    # Logs a timestamped line after each training epoch (model.fit runs with verbose=0).
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(
            "Epoch %d | loss=%.4f | val_loss=%.4f | val_mae=%.4f",
            epoch + 1,
            logs.get("loss", float("nan")),
            logs.get("val_loss", float("nan")),
            logs.get("val_mae", float("nan")),
        )


def train() -> pd.DataFrame:
    logger.info(
        "Starting LSTM signal-source training run (tickers=%d)", len(ALL_TICKERS)
    )

    ticker_features = load_ticker_features()

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
        raise ValueError(
            "No valid pooled dataset could be built from the available tickers."
        )

    logger.info("Training samples: %d", len(dataset["train"]["y"]))
    logger.info("Validation samples: %d", len(dataset["validation"]["y"]))
    logger.info("Test samples: %d", len(dataset["test"]["y"]))
    logger.info("Tickers used: %d", len(dataset["tickers"]))

    X_train = dataset["train"]["X"]
    y_train = dataset["train"]["y_scaled"]
    X_validation = dataset["validation"]["X"]
    y_validation = dataset["validation"]["y_scaled"]
    X_test = dataset["test"]["X"]
    y_test = dataset["test"]["y"]
    current_close_test = dataset["test"]["current_close"]
    target_close_test = dataset["test"]["target_close"]

    keras.backend.clear_session()
    keras.utils.set_random_seed(RANDOM_SEED)

    model = build_lstm_model((WINDOW_SIZE, len(FEATURE_COLS)))

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, EpochLoggingCallback()],
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
    y_pred = inverse_scale_predictions(dataset["target_scaler"], y_pred_scaled)
    naive_pred = np.zeros_like(y_test)

    model_metrics = regression_metrics(y_pred, current_close_test, target_close_test)
    naive_metrics = regression_metrics(
        naive_pred, current_close_test, target_close_test
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump((dataset["feature_scaler"], dataset["target_scaler"]), SCALERS_PATH)
    logger.info("Saved LSTM signal-source model to %s", MODEL_PATH)

    result = {
        "model": "RL signal source LSTM",
        "forecast_horizon": f"{FORECAST_HORIZON}d",
        "train_model_end": TRAIN_MODEL_END,
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

    logger.info("Finished LSTM signal-source training run")
    return results_df


if __name__ == "__main__":
    print(train().to_string(index=False))
