"""Train the PPO portfolio agent.

Based on guidance here:
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
  (custom Gymnasium environment + check_env)
- https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
  (vec_env/model/train-predict usage example)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from core.portfolio_env import PortfolioEnv, make_buy_and_hold_curve, portfolio_metrics
from core.tickers import TICKER_GROUPS

RANDOM_SEED = 42
PORTFOLIO_TICKERS = TICKER_GROUPS["Utilities"][:5]
PPO_TIMESTEPS = 10_000

INITIAL_CASH = 100_000.0
TRANSACTION_COST = 0.001
TRADE_FRACTION = 0.25
REWARD_SCALE = INITIAL_CASH

# Use 80% of the data for training and 20% for testing
TRAIN_FRACTION = 0.8


def make_synthetic_rl_data(tickers, n_days: int = 500, seed: int = RANDOM_SEED):
    """Generate correlated synthetic prices and noisy momentum-based signals.

    Verifies the environment and PPO integration only; not real market data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    n_assets = len(tickers)

    # Give every asset a shared market move plus its own idiosyncratic noise,
    # so prices are correlated the way real tickers in one sector would be.
    market_returns = rng.normal(0.0003, 0.008, n_days)
    asset_returns = market_returns[:, None] + rng.normal(0, 0.006, (n_days, n_assets))

    starting_prices = rng.uniform(40, 180, n_assets)
    prices = pd.DataFrame(
        starting_prices * np.exp(np.cumsum(asset_returns, axis=0)),
        index=dates,
        columns=tickers,
    )

    momentum = prices.pct_change().rolling(10).mean()
    signals = (1 / (1 + np.exp(-30 * momentum))).fillna(0.5)

    return prices, signals


def split_train_test(prices, signals):
    """Split data into training and testing sets."""
    split_index = int(len(prices) * TRAIN_FRACTION)
    train_prices, test_prices = prices.iloc[:split_index], prices.iloc[split_index:]
    train_signals, test_signals = signals.iloc[:split_index], signals.iloc[split_index:]
    return train_prices, train_signals, test_prices, test_signals


def make_env(prices, signals):
    return PortfolioEnv(
        prices,
        signals,
        initial_cash=INITIAL_CASH,
        transaction_cost=TRANSACTION_COST,
        trade_fraction=TRADE_FRACTION,
        reward_scale=REWARD_SCALE,
    )


def evaluate_policy(model, environment):
    # Run one PPO episode to see how it performs on the test set.
    obs, info = environment.reset(seed=RANDOM_SEED)
    terminated = truncated = False

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = environment.step(action)

    return pd.DataFrame(environment.history).set_index("date")


def evaluate_random_policy(environment, seed=RANDOM_SEED):
    """Run one random-action episode as an evaluation baseline."""
    obs, info = environment.reset(seed=seed)
    environment.action_space.seed(seed)
    terminated = truncated = False

    while not (terminated or truncated):
        action = environment.action_space.sample()
        obs, reward, terminated, truncated, info = environment.step(action)

    return pd.DataFrame(environment.history).set_index("date")


def train() -> pd.DataFrame:
    prices, signals = make_synthetic_rl_data(PORTFOLIO_TICKERS)
    train_prices, train_signals, test_prices, test_signals = split_train_test(prices, signals)

    print(f"Train dates: {train_prices.index[0].date()} to {train_prices.index[-1].date()}")
    print(f"Test dates:  {test_prices.index[0].date()} to {test_prices.index[-1].date()}")

    # check_env verifies Gymnasium API compliance before spending time on training.
    check_env(make_env(train_prices.iloc[:100], train_signals.iloc[:100]), warn=True)

    vec_env = make_vec_env(lambda: make_env(train_prices, train_signals), n_envs=1, seed=RANDOM_SEED)

    model = PPO("MlpPolicy", vec_env, verbose=1, seed=RANDOM_SEED)
    model.learn(total_timesteps=PPO_TIMESTEPS, progress_bar=True)

    ppo_history = evaluate_policy(model, make_env(test_prices, test_signals))
    random_history = evaluate_random_policy(make_env(test_prices, test_signals))

    portfolio_curves = pd.DataFrame(
        {
            "PPO": ppo_history["portfolio_value"],
            "Random policy": random_history["portfolio_value"],
            "Buy and hold": make_buy_and_hold_curve(test_prices, INITIAL_CASH, TRANSACTION_COST),
            "Cash": pd.Series(INITIAL_CASH, index=test_prices.index),
        }
    ).dropna()

    results_df = pd.DataFrame(
        {strategy: portfolio_metrics(portfolio_curves[strategy]) for strategy in portfolio_curves.columns}
    ).T

    metric_cols = [
        "total_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ]
    results_df[metric_cols] = results_df[metric_cols].round(4)
    results_df["final_value"] = results_df["final_value"].round(2)
    return results_df


if __name__ == "__main__":
    print(train().to_string())
