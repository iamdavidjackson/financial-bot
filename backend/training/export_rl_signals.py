# 
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from tensorflow import keras

from core.fetch_data import SELECTED_FEATURE_COLS, get_ticker_features
from core.lstm_helpers import convert_returns_to_signals, predict_return_series
from core.tickers import TICKER_GROUPS
from training.train_rl_signal_lstm import FORECAST_HORIZON, MODEL_PATH, SCALERS_PATH, WINDOW_SIZE

PORTFOLIO_TICKERS = TICKER_GROUPS["Utilities"][:5]

# Starts early enough to cover WINDOW_SIZE plus indicator/z-score warm-up before WALK_FORWARD_START.
FEATURE_START = "2019-06-01"
WALK_FORWARD_START = "2020-01-01"
DOWNLOAD_END = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

FEATURE_COLS = list(dict.fromkeys(SELECTED_FEATURE_COLS))

RL_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data" / "rl"
PRICE_PATH = RL_DATA_DIR / "prices.parquet"
SIGNAL_PATH = RL_DATA_DIR / f"lstm_signals_{FORECAST_HORIZON}d.parquet"

def load_signal_model():
    model = keras.models.load_model(MODEL_PATH)
    feature_scaler, target_scaler = joblib.load(SCALERS_PATH)
    return model, feature_scaler, target_scaler

def export() -> pd.DataFrame:
    model, feature_scaler, target_scaler = load_signal_model()

    predicted_returns = {}
    closes = {}

    for ticker in PORTFOLIO_TICKERS:
        # Get features for the ticker
        df = get_ticker_features(
            ticker=ticker,
            start=FEATURE_START,
            end=DOWNLOAD_END,
            smooth_outliers=True,
        )
        # Ensure we have enough rows for predicting returns
        required_rows = WINDOW_SIZE + 1
        if df.empty or len(df) < required_rows:
            raise ValueError(f"{ticker}: only {len(df)} rows, need at least {required_rows}")

        return_series = predict_return_series(
            model,
            feature_scaler,
            target_scaler,
            df,
            FEATURE_COLS,
            WINDOW_SIZE,
        )
        predicted_returns[ticker] = return_series
        closes[ticker] = df["Close"]
        print(f"{ticker}: {len(return_series)} walk-forward predictions "
              f"({return_series.index[0].date()} to {return_series.index[-1].date()})")

    predicted_returns_df = pd.DataFrame(predicted_returns).dropna()
    prices_df = pd.DataFrame(closes).dropna()

    common_dates = predicted_returns_df.index.intersection(prices_df.index)
    common_dates = common_dates[common_dates >= WALK_FORWARD_START]
    predicted_returns_df = predicted_returns_df.loc[common_dates, PORTFOLIO_TICKERS]
    prices_df = prices_df.loc[common_dates, PORTFOLIO_TICKERS].astype("float32")

    signals_df = convert_returns_to_signals(predicted_returns_df)

    RL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(PRICE_PATH)
    signals_df.to_parquet(SIGNAL_PATH)

    print(f"Saved {len(prices_df)} rows to {PRICE_PATH}")
    print(f"Saved {len(signals_df)} rows to {SIGNAL_PATH}")

    return pd.DataFrame(
        {
            "ticker": PORTFOLIO_TICKERS,
            "mean_predicted_return": [predicted_returns_df[t].mean() for t in PORTFOLIO_TICKERS],
            "mean_signal": [signals_df[t].mean() for t in PORTFOLIO_TICKERS],
        }
    )

if __name__ == "__main__":
    print(export().to_string(index=False))
