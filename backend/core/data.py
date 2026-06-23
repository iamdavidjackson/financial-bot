from pyexpat import features

from matplotlib.pyplot import close
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    df.index = df.index.tz_localize(None)
    return df

def compute_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features['rsi'] = RSIIndicator(close=features['Close'], window=14).rsi()

    macd = MACD(close=features['Close'])
    features['macd']        = macd.macd()
    features['macd_signal'] = macd.macd_signal()

    bb = BollingerBands(close=features['Close'], window=20, window_dev=2)
    features['bb_upper'] = bb.bollinger_hband()
    features['bb_mid']   = bb.bollinger_mavg()
    features['bb_lower'] = bb.bollinger_lband()

    features['atr'] = AverageTrueRange(
        high=features['High'], low=features['Low'], close=features['Close'], window=14
    ).average_true_range()

    features['obv'] = OnBalanceVolumeIndicator(
        close=features['Close'], volume=features['Volume']
    ).on_balance_volume()

    features['hlc3'] = (features['High'] + features['Low'] + features['Close']) / 3

    # rsi scaled
    features['rsi_scaled'] = features['rsi'] / 100

    # Apply transformation to make features scale-invariant and generalise across time periods
    close = features['Close']
    features['target_close'] = close

    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'hlc3']:
        features[f'{col}_return'] = features[col].pct_change()

    # Volume ratio: current volume relative to recent average, clipped to handle outliers
    features['volume_ratio'] = features['Volume'] / features['Volume'].rolling(20).mean()
    features['volume_ratio_clipped'] = features['volume_ratio'].clip(upper=3)

    features['obv_change_to_avg_volume'] = features['obv'].diff() / features['Volume'].rolling(20).mean()
    features['macd_to_close'] = features['macd'] / close
    features['macd_signal_to_close'] = features['macd_signal'] / close
    features['atr_to_close'] = features['atr'] / close
    features['bb_upper_to_close'] = features['bb_upper'] / close
    features['bb_mid_to_close'] = features['bb_mid'] / close
    features['bb_lower_to_close'] = features['bb_lower'] / close

    market_close = market_df['Close'].reindex(features.index).ffill()

    features['market_close'] = market_close

    # 1 day window growth calculations
    features['growth_1d'] = features['target_close'].pct_change(1)
    features['market_growth_1d'] = features['market_close'].pct_change(1)
    features['excess_growth_vs_market_1d'] = features['growth_1d'] - features['market_growth_1d']
    features['beat_market_1d'] = features['excess_growth_vs_market_1d'] > 0

    # 2 day window growth calculations
    features['growth_2d'] = features['target_close'].pct_change(2)
    features['market_growth_2d'] = features['market_close'].pct_change(2)
    features['excess_growth_vs_market_2d'] = features['growth_2d'] - features['market_growth_2d']
    features['beat_market_2d'] = features['excess_growth_vs_market_2d'] > 0

    # 3 day window growth calculations
    features['growth_3d'] = features['target_close'].pct_change(3)
    features['market_growth_3d'] = features['market_close'].pct_change(3)
    features['excess_growth_vs_market_3d'] = features['growth_3d'] - features['market_growth_3d']
    features['beat_market_3d'] = features['excess_growth_vs_market_3d'] > 0

    # 5 day window growth calculations
    features['growth_5d'] = features['target_close'].pct_change(5)
    features['market_growth_5d'] = features['market_close'].pct_change(5)
    features['excess_growth_vs_market_5d'] = features['growth_5d'] - features['market_growth_5d']
    features['beat_market_5d'] = features['excess_growth_vs_market_5d'] > 0

    # 10 day window growth calculations
    features['growth_10d'] = features['target_close'].pct_change(10)
    features['market_growth_10d'] = features['market_close'].pct_change(10)
    features['excess_growth_vs_market_10d'] = features['growth_10d'] - features['market_growth_10d']
    features['beat_market_10d'] = features['excess_growth_vs_market_10d'] > 0

    # 20 day window growth calculations
    features['growth_20d'] = features['target_close'].pct_change(20)
    features['market_growth_20d'] = features['market_close'].pct_change(20)
    features['excess_growth_vs_market_20d'] = features['growth_20d'] - features['market_growth_20d']
    features['beat_market_20d'] = features['excess_growth_vs_market_20d'] > 0

    return features.dropna()

def get_ticker_features(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = download_ticker(ticker, start, end)
    market_df = download_ticker('SPY', start=start, end=end)
    return compute_features(df, market_df)

def get_model_data(tickers: dict[str, pd.DataFrame], features: list[str], forecast_horizon: int) -> pd.DataFrame:
    target_col = f'target_up_{forecast_horizon}d'
    required_cols = features + ['target_close']
    model_frames = []

    for ticker, df in tickers.items():
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f'{ticker} is missing required columns: {missing_cols}')

        ticker_model_data = df.copy()
        future_close = ticker_model_data['target_close'].shift(-forecast_horizon)
        ticker_model_data[target_col] = np.where(
            future_close.notna(),
            future_close > ticker_model_data['target_close'],
            np.nan,
        )
        ticker_model_data['ticker'] = ticker
        model_frames.append(ticker_model_data)

    model_data = pd.concat(model_frames).replace([np.inf, -np.inf], np.nan)
    return model_data.dropna(subset=features + [target_col])

def get_train_test_data(
    model_data: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_end: str,
    test_start: str,
):
    required_cols = features + [target_col]
    missing_cols = [col for col in required_cols if col not in model_data.columns]
    if missing_cols:
        raise ValueError(f'model_data is missing required columns: {missing_cols}')

    train_rows = model_data[model_data.index <= train_end]
    test_rows = model_data[model_data.index >= test_start]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_rows[features])
    X_test = scaler.transform(test_rows[features])
    y_train = train_rows[target_col].astype(int).to_numpy()
    y_test = test_rows[target_col].astype(int).to_numpy()

    return train_rows, test_rows, scaler, X_train, X_test, y_train, y_test

def get_naive_baseline_results(y_test: np.ndarray, forecast_horizon: int) -> pd.DataFrame:
    naive_positive_rate = y_test.mean()
    naive_pred_class = int(naive_positive_rate >= 0.5)
    naive_pred = np.full(len(y_test), naive_pred_class)
    naive_prob = np.full(len(y_test), naive_positive_rate)

    naive_baseline_results = pd.DataFrame([
        {
            'model': 'Naive majority baseline',
            'forecast_horizon': f'{forecast_horizon}d',
            'samples': len(y_test),
            'up_rate': naive_positive_rate,
            'predicted_class': naive_pred_class,
            'accuracy': accuracy_score(y_test, naive_pred),
            'log_loss': log_loss(y_test, naive_prob, labels=[0, 1]),
            'auc': 0.5,
        }
    ])

    metric_cols = ['up_rate', 'accuracy', 'log_loss', 'auc']
    naive_baseline_results[metric_cols] = naive_baseline_results[metric_cols].round(4)
    return naive_baseline_results

def get_logistic_regression_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    forecast_horizon: int,
):
    logistic_model = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )
    logistic_model.fit(X_train, y_train)

    y_prob = logistic_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    logistic_results = pd.DataFrame([
        {
            'model': 'Logistic regression',
            'forecast_horizon': f'{forecast_horizon}d',
            'samples': len(y_test),
            'up_rate': y_test.mean(),
            'accuracy': accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_prob, labels=[0, 1]),
            'auc': roc_auc_score(y_test, y_prob),
        },
    ])

    metric_cols = ['up_rate', 'accuracy', 'log_loss', 'auc']
    logistic_results[metric_cols] = logistic_results[metric_cols].round(4)
    return logistic_model, logistic_results
