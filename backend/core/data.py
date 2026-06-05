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

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return features.dropna()

def get_ticker_features(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = download_ticker(ticker, start, end)
    return compute_features(df)
