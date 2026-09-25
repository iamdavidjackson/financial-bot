import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator

# Trading days in the rolling window used to normalize each level column.
ZSCORE_WINDOW = 60

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# these are raw prices or volume levels which need to be normalized
LEVEL_COLS = [
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "hlc3",
    "obv",
    "tema",
    "sma5",
    "bbl",
    "sma50",
    "sma14",
    "fwma",
    "hwma",
    "ichimoku",
    "hma",
    "pvt",
    "bbu",
]

# this is the feature set used for modeling, after z-scoring each level column
SELECTED_FEATURE_COLS = [f"{col}_z" for col in LEVEL_COLS]


def get_latest_close(ticker: str) -> float:
    """Fetch a ticker's most recent closing price, for pricing a trade someone reports today."""
    data = yf.Ticker(ticker).history(period="5d", auto_adjust=True)

    close_prices = data["Close"].dropna() if "Close" in data else pd.Series(dtype=float)
    if close_prices.empty:
        raise ValueError(f"No recent price data for {ticker}")

    return float(close_prices.iloc[-1])


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)

    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_localize(None)

    return data


def verify_ohlcv_rows(
    data: pd.DataFrame,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return row-level quality checks for the raw OHLCV data."""
    required_cols = required_cols or OHLCV_COLS
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

    checks = pd.DataFrame(index=data.index)

    # Missing OHLCV data means the row cannot be used safely for indicators.
    checks["missing_ohlcv_values"] = data[required_cols].isna().sum(axis=1)

    # Prices and volume should be positive. Zero or negative values are data errors.
    checks["non_positive_ohlcv_values"] = (data[required_cols] <= 0).sum(axis=1)

    # High/low should contain the open and close price for the same day.
    checks["invalid_price_range"] = (
        (data["High"] < data[["Open", "Close"]].max(axis=1))
        | (data["Low"] > data[["Open", "Close"]].min(axis=1))
        | (data["High"] < data["Low"])
    )

    checks["has_issue"] = (
        (checks["missing_ohlcv_values"] > 0)
        | (checks["non_positive_ohlcv_values"] > 0)
        | checks["invalid_price_range"]
    )

    return checks


def smooth_ohlcv_outliers(
    data: pd.DataFrame,
    cols: list[str] | None = None,
    iqr_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace extreme OHLCV outliers with linearly interpolated values."""
    cols = cols or OHLCV_COLS
    smoothed = data.copy()
    outlier_report = []

    for col in cols:
        q1 = smoothed[col].quantile(0.25)
        q3 = smoothed[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        outlier_mask = smoothed[col].lt(lower_bound) | smoothed[col].gt(upper_bound)

        outlier_report.append(
            {
                "column": col,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(outlier_mask.sum()),
            }
        )

        # Temporarily blank the outliers so interpolation can estimate replacement values.
        smoothed.loc[outlier_mask, col] = pd.NA
        smoothed[col] = smoothed[col].interpolate(method="linear").bfill().ffill()

    return smoothed, pd.DataFrame(outlier_report)


def weighted_moving_average(
    series: pd.Series,
    window: int,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """Calculate a weighted moving average over a rolling window."""
    if weights is None:
        weights = np.arange(1, window + 1)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    return series.rolling(window).apply(
        lambda values: np.dot(values, weights), raw=True
    )


def triple_exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate TEMA by combining three exponential moving averages."""
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()

    return (3 * ema1) - (3 * ema2) + ema3


def fibonacci_weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate FWMA using recent prices weighted by Fibonacci numbers."""
    fibonacci_weights = [1, 1]

    while len(fibonacci_weights) < window:
        fibonacci_weights.append(fibonacci_weights[-1] + fibonacci_weights[-2])

    return weighted_moving_average(
        series, window, np.array(fibonacci_weights[-window:])
    )


def hull_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate HMA, a smoother moving average with reduced lag."""
    half_window = max(int(window / 2), 1)
    sqrt_window = max(int(np.sqrt(window)), 1)

    half_wma = weighted_moving_average(series, half_window)
    full_wma = weighted_moving_average(series, window)

    return weighted_moving_average((2 * half_wma) - full_wma, sqrt_window)


def holt_winter_moving_average(
    series: pd.Series,
    alpha: float = 0.2,
    beta: float = 0.1,
    gamma: float = 0.1,
) -> pd.Series:
    """Calculate a simple Holt-Winter style smoothed series."""
    if series.empty:
        return series.copy()

    level = float(series.iloc[0])
    trend = 0.0
    acceleration = 0.0
    values = []

    for value in series:
        previous_level = level
        previous_trend = trend

        level = alpha * value + (1 - alpha) * (level + trend + acceleration)
        trend = beta * (level - previous_level) + (1 - beta) * (trend + acceleration)
        acceleration = gamma * (trend - previous_trend) + (1 - gamma) * acceleration

        values.append(level + trend + acceleration)

    return pd.Series(values, index=series.index)


# Normalize features using rolling z-scores
def rolling_zscore(
    features: pd.DataFrame, cols: list[str], window: int
) -> pd.DataFrame:
    """Calculate rolling z-scores for specified columns in a DataFrame."""
    normalized = features.copy()

    for col in cols:
        rolling_mean = features[col].rolling(window).mean()
        rolling_std = features[col].rolling(window).std().replace(0, np.nan)
        normalized[f"{col}_z"] = (features[col] - rolling_mean) / rolling_std

    return normalized


def compute_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the feature set used by the yfinance exploration notebook."""
    features = data.copy()

    features["hlc3"] = (features["High"] + features["Low"] + features["Close"]) / 3
    features["obv"] = OnBalanceVolumeIndicator(
        close=features["Close"],
        volume=features["Volume"],
    ).on_balance_volume()
    features["pvt"] = VolumePriceTrendIndicator(
        close=features["Close"],
        volume=features["Volume"],
    ).volume_price_trend()

    features["tema"] = triple_exponential_moving_average(features["Close"], 20)
    features["sma5"] = features["Close"].rolling(5).mean()
    features["sma14"] = features["Close"].rolling(14).mean()
    features["sma50"] = features["Close"].rolling(50).mean()
    features["fwma"] = fibonacci_weighted_moving_average(features["Close"], 20)
    features["hwma"] = holt_winter_moving_average(features["Close"])
    features["hma"] = hull_moving_average(features["Close"], 20)

    bollinger = BollingerBands(close=features["Close"], window=20, window_dev=2)
    features["bbu"] = bollinger.bollinger_hband()
    features["bbl"] = bollinger.bollinger_lband()

    ichimoku = IchimokuIndicator(
        high=features["High"],
        low=features["Low"],
        window1=9,
        window2=26,
        window3=52,
    )
    features["ichimoku"] = (ichimoku.ichimoku_a() + ichimoku.ichimoku_b()) / 2

    features = rolling_zscore(features, LEVEL_COLS, window=ZSCORE_WINDOW)

    return features.dropna(subset=SELECTED_FEATURE_COLS)


def get_ticker_features(
    ticker: str,
    start: str,
    end: str,
    smooth_outliers: bool = True,
) -> pd.DataFrame:
    """Download a ticker and return the cleaned technical feature set."""
    data = download_ticker(ticker, start, end)

    if smooth_outliers:
        data, _ = smooth_ohlcv_outliers(data)

    return compute_technical_features(data)
