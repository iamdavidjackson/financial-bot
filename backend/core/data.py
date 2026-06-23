import pandas as pd
import yfinance as yf
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
