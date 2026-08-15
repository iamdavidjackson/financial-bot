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

from core.portfolio_env import PortfolioEnv
from core.tickers import TICKER_GROUPS

RANDOM_SEED = 42
PORTFOLIO_TICKERS = TICKER_GROUPS["Utilities"][:5]
PPO_TIMESTEPS = 10_000


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


def make_env():
    prices, signals = make_synthetic_rl_data(PORTFOLIO_TICKERS)
    return PortfolioEnv(prices, signals)


def train():
    check_env(make_env(), warn=True)

    vec_env = make_vec_env(make_env, n_envs=1, seed=RANDOM_SEED)

    model = PPO("MlpPolicy", vec_env, verbose=1, seed=RANDOM_SEED)
    model.learn(total_timesteps=PPO_TIMESTEPS, progress_bar=True)

    # Same predict/step shape as the ppo.html usage example.
    obs = vec_env.reset()
    for _ in range(20):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(f"action={action[0]} reward={reward[0]:.2f} portfolio_value={info[0]['portfolio_value']:.2f}")


if __name__ == "__main__":
    train()
