from pathlib import Path

import joblib
import pandas as pd

from .fetch_data import SELECTED_FEATURE_COLS, get_ticker_features
from .lstm_helpers import inverse_scale_predictions
from .tickers import TICKERS

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "trained_models"
    / "lstm_all_tickers_return_regression"
)
MODEL_PATH = MODEL_DIR / "all_tickers_5d_return.keras"
SCALERS_PATH = MODEL_DIR / "all_tickers_5d_return.scalers.joblib"

FEATURE_COLS = list(dict.fromkeys(SELECTED_FEATURE_COLS))
WINDOW_SIZE = 30

# Accounts for SMA50 lookback, z-score window, and margin for weekends/holidays.
FEATURE_LOOKBACK_DAYS = 320

_model = None
_scalers = None

def _signal_from_return(predicted_return: float, buy: float = 0.01, sell: float = -0.01) -> str:
    """Turn a predicted 5-day return into a buy / hold / sell label."""
    if predicted_return >= buy:
        return "buy"
    if predicted_return <= sell:
        return "sell"
    return "hold"

def _load_model():
    global _model
    if _model is None:
        from tensorflow import keras

        _model = keras.models.load_model(MODEL_PATH)
    return _model

def _get_scalers():
    """Load the scalars used when training the model."""
    global _scalers
    if _scalers is None:
        if not SCALERS_PATH.exists():
            raise RuntimeError(
                f"Missing {SCALERS_PATH}. Run backend/training/train_all_tickers.py to train "
                "the model and save its scalers."
            )
        _scalers = joblib.load(SCALERS_PATH)

    return _scalers

def predict_stock_return(ticker: str) -> dict:
    """Predict a stock's return over the next 5 trading days using the pooled all-tickers LSTM model."""
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise ValueError(f"{ticker} is not one of the tickers this model was trained on.")

    feature_scaler, target_scaler = _get_scalers()

    today = pd.Timestamp.today()
    start = (today - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = get_ticker_features(
        ticker=ticker,
        start=start,
        end=end,
        smooth_outliers=True,
    )

    latest_window = df[FEATURE_COLS].dropna().tail(WINDOW_SIZE)
    if len(latest_window) < WINDOW_SIZE:
        raise RuntimeError(f"Not enough recent {ticker} data to build a prediction window.")

    scaled_window = feature_scaler.transform(latest_window)
    model_input = scaled_window.reshape(1, WINDOW_SIZE, len(FEATURE_COLS)).astype("float32")

    model = _load_model()
    scaled_prediction = model.predict(model_input, verbose=0)
    predicted_return = float(inverse_scale_predictions(target_scaler, scaled_prediction)[0])

    current_close = float(df["Close"].iloc[-1])
    predicted_close = current_close * (1 + predicted_return)
    as_of_date = df.index[-1]

    return {
        "ticker": ticker,
        "as_of_date": str(as_of_date.date()),
        "current_close": round(current_close, 2),
        "predicted_close_5d": round(predicted_close, 2),
        "predicted_return_5d": round(predicted_return, 4),
        "predicted_return_5d_pct": f"{predicted_return:+.2%}",
        "signal": _signal_from_return(predicted_return),
    }
