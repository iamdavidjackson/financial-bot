"""Custom Gymnasium environment for the PPO portfolio agent.

Based on guidance here:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
"""

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class PortfolioEnv(Env):
    """Daily Hold/Buy/Sell environment for a single ticker, driven by an LSTM signal."""

    metadata = {"render_modes": ["human"]}

    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, prices: pd.Series, signals: pd.Series, initial_cash: float = 100_000.0):
        super().__init__()

        self.prices = prices.to_numpy(dtype=np.float32)
        self.signals = signals.to_numpy(dtype=np.float32)
        self.dates = prices.index
        self.initial_cash = float(initial_cash)

        self.action_space = spaces.Discrete(3)
        # [signal, price_ratio, cash_ratio]
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(3,), dtype=np.float32)

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def _get_observation(self) -> np.ndarray:
        price = self.prices[self.current_step]
        # Floor the denominator so a near-zero portfolio can't divide by zero.
        portfolio_value = max(self._portfolio_value(price), 1e-8)
        price_ratio = price / self.episode_start_price
        cash_ratio = self.cash / portfolio_value

        return np.array(
            [self.signals[self.current_step], price_ratio, cash_ratio],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.episode_start_price = self.prices[0]
        return self._get_observation(), {}

    def step(self, action):
        price = self.prices[self.current_step]
        previous_value = self._portfolio_value(price)

        if action == self.BUY and self.cash > 0:
            self.shares += self.cash / price
            self.cash = 0.0
        elif action == self.SELL and self.shares > 0:
            self.cash += self.shares * price
            self.shares = 0.0

        self.current_step += 1
        new_price = self.prices[self.current_step]
        new_value = self._portfolio_value(new_price)
        reward = new_value - previous_value

        # The episode ends once there's no next day left to price the portfolio on.
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        info = {"portfolio_value": new_value}

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        price = self.prices[self.current_step]
        print(
            f"{self.dates[self.current_step].date()} | "
            f"value={self._portfolio_value(price):,.2f}"
        )
