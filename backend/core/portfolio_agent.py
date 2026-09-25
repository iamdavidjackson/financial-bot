from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .fetch_data import SELECTED_FEATURE_COLS, get_ticker_features
from .lstm_helpers import convert_returns_to_signals, predict_return_series
from .portfolio_env import PortfolioEnv
from .portfolio_store import get_portfolio
from .tickers import TICKER_GROUPS

# Only training the PPO on 5 Utility tickers for testing - each portfolio agent needs to be trained on tickers separately
PORTFOLIO_TICKERS = TICKER_GROUPS["Utilities"][:5]
PORTFOLIO_TICKER_NAMES = {
    "NEE": "NextEra Energy, Inc.",
    "SO": "The Southern Company",
    "DUK": "Duke Energy Corporation",
    "AEP": "American Electric Power Company, Inc.",
    "EXC": "Exelon Corporation",
}
INITIAL_CASH = 100_000.0
TRANSACTION_COST = 0.001
TRADE_FRACTION = 0.25
REWARD_SCALE = INITIAL_CASH
WINDOW_SIZE = 30

# Model paths
SIGNAL_MODEL_DIR = (
    Path(__file__).resolve().parent.parent / "trained_models" / "lstm_rl_signal_source"
)
SIGNAL_MODEL_PATH = SIGNAL_MODEL_DIR / "rl_signal_source_5d_return.keras"
SIGNAL_SCALERS_PATH = SIGNAL_MODEL_DIR / "rl_signal_source_5d_return.scalers.joblib"
PPO_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "trained_models"
    / "ppo_portfolio_agent"
    / "ppo_portfolio_agent"
)

FEATURE_COLS = list(dict.fromkeys(SELECTED_FEATURE_COLS))
RECENT_TRADING_DAYS = 10
FEATURE_LOOKBACK_DAYS = 320 + RECENT_TRADING_DAYS

_ppo_model = None
_signal_model = None
_signal_scalers = None

ACTION_LABELS = {
    PortfolioEnv.HOLD: "hold",
    PortfolioEnv.BUY: "buy",
    PortfolioEnv.SELL: "sell",
}


def _load_ppo_model():
    global _ppo_model
    if _ppo_model is None:
        if not PPO_MODEL_PATH.with_suffix(".zip").exists():
            raise RuntimeError(
                f"Missing {PPO_MODEL_PATH}.zip. Run backend/training/train_rl_agent.py "
                "to train and save the PPO policy."
            )
        from stable_baselines3 import PPO

        _ppo_model = PPO.load(PPO_MODEL_PATH)
    return _ppo_model


def _load_signal_model():
    global _signal_model, _signal_scalers
    if _signal_model is None:
        if not SIGNAL_MODEL_PATH.exists() or not SIGNAL_SCALERS_PATH.exists():
            raise RuntimeError(
                f"Missing {SIGNAL_MODEL_PATH} or {SIGNAL_SCALERS_PATH}. Run "
                "backend/training/train_rl_signal_lstm.py to train and save the signal-source LSTM."
            )
        from tensorflow import keras

        _signal_model = keras.models.load_model(SIGNAL_MODEL_PATH)
        _signal_scalers = joblib.load(SIGNAL_SCALERS_PATH)
    return _signal_model, _signal_scalers


def _load_recent_prices_and_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Builds a short recent price/signal window for PORTFOLIO_TICKERS, ending today.
    signal_model, (feature_scaler, target_scaler) = _load_signal_model()

    today = pd.Timestamp.today()
    start = (today - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    predicted_returns = {}
    closes = {}

    for ticker in PORTFOLIO_TICKERS:
        df = get_ticker_features(
            ticker=ticker, start=start, end=end, smooth_outliers=True
        )
        if df.empty or len(df) < WINDOW_SIZE + 1:
            raise RuntimeError(
                f"{ticker}: need at least {WINDOW_SIZE + 1} rows but only got {len(df)}"
            )

        predicted_returns[ticker] = predict_return_series(
            signal_model, feature_scaler, target_scaler, df, FEATURE_COLS, WINDOW_SIZE
        )
        closes[ticker] = df["Close"]

    predicted_returns_df = pd.DataFrame(predicted_returns).dropna()
    prices_df = pd.DataFrame(closes).dropna()

    common_dates = predicted_returns_df.index.intersection(prices_df.index)
    # Trim to a short recent window so normalised_prices show a recent trend.
    common_dates = common_dates[-RECENT_TRADING_DAYS:]

    prices_df = prices_df.loc[common_dates, PORTFOLIO_TICKERS].astype("float32")
    signals_df = convert_returns_to_signals(
        predicted_returns_df.loc[common_dates, PORTFOLIO_TICKERS]
    )

    return prices_df, signals_df


def _plan_trade_sizes(
    action: np.ndarray,
    current_prices: np.ndarray,
    cash: float,
    holdings: np.ndarray,
    portfolio_value: float,
) -> dict[int, int]:
    """Work out how many shares each Buy/Sell action would move today.

    Mirrors PortfolioEnv._execute_trades: sells are sized first (they raise cash the
    same-step buys can draw on), then the remaining buy budget is split evenly across
    the tickers still to buy. Runs on copies so the real portfolio is untouched.
    """
    cash = float(cash)
    holdings = holdings.astype("float64").copy()
    planned = {index: 0 for index in range(len(action))}

    for index in np.flatnonzero(action == PortfolioEnv.SELL):
        shares = int(np.ceil(holdings[index] * TRADE_FRACTION))
        shares = min(shares, int(holdings[index]))
        if shares <= 0:
            continue
        gross_proceeds = shares * current_prices[index]
        cash += gross_proceeds - gross_proceeds * TRANSACTION_COST
        holdings[index] -= shares
        planned[index] = shares

    buy_indexes = np.flatnonzero(action == PortfolioEnv.BUY)
    remaining_buys = len(buy_indexes)
    for index in buy_indexes:
        if remaining_buys <= 0 or cash <= 0:
            break
        budget = min(portfolio_value * TRADE_FRACTION, cash / remaining_buys)
        price_with_cost = current_prices[index] * (1 + TRANSACTION_COST)
        shares = int(np.floor(budget / price_with_cost))
        if shares > 0:
            gross_cost = shares * current_prices[index]
            cash -= gross_cost + gross_cost * TRANSACTION_COST
            planned[index] = shares
        remaining_buys -= 1

    return planned


def recommend_trades() -> dict:
    # Asks the trained PPO agent what to do with each tracked ticker today, given the real portfolio.
    ppo_model = _load_ppo_model()
    prices_df, signals_df = _load_recent_prices_and_signals()

    env = PortfolioEnv(
        prices_df,
        signals_df,
        initial_cash=INITIAL_CASH,
        transaction_cost=TRANSACTION_COST,
        trade_fraction=TRADE_FRACTION,
        reward_scale=REWARD_SCALE,
    )
    env.reset()

    # get cash and holding from portfolio
    portfolio = get_portfolio()
    env.cash = float(portfolio["cash"])
    env.holdings = (
        pd.Series(portfolio["holdings"])
        .reindex(PORTFOLIO_TICKERS, fill_value=0.0)
        .to_numpy(dtype="float32")
    )
    # select todays prices
    env.current_step = len(env.prices) - 1

    # make an observation and ask the PPO model what to do with each ticker
    observation = env._get_observation()
    action, _states = ppo_model.predict(observation, deterministic=True)

    as_of_date = env.dates[env.current_step]
    current_prices = env.prices[env.current_step]

    portfolio_value = float(env.cash + np.dot(env.holdings, current_prices))
    planned_shares = _plan_trade_sizes(
        action, current_prices, env.cash, env.holdings, portfolio_value
    )

    recommendations = {}
    for index, ticker in enumerate(PORTFOLIO_TICKERS):
        price = float(current_prices[index])
        shares = planned_shares[index]
        recommendations[ticker] = {
            "action": ACTION_LABELS[int(action[index])],
            "current_price": round(price, 2),
            "current_holding_shares": float(env.holdings[index]),
            "signal_percentile": round(float(env.signals[env.current_step][index]), 4),
            "recommended_shares": shares,
            "estimated_trade_value": round(shares * price, 2),
        }

    return {
        "as_of_date": str(as_of_date.date()),
        "cash": round(env.cash, 2),
        "recommendations": recommendations,
    }
