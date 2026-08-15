"""Custom Gymnasium environment for the PPO portfolio agent.

Based on guidance here:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
"""

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class PortfolioEnv(Env):
    """Daily Hold/Buy/Sell portfolio environment driven by LSTM directional data."""

    metadata = {"render_modes": ["human"]}

    HOLD = 0
    BUY = 1
    SELL = 2

    # Minimum portfolio value to avoid divide-by-zero in observation calculations.
    _MIN_PORTFOLIO_VALUE = 0.01

    def __init__(self, prices: pd.DataFrame, signals: pd.DataFrame, initial_cash: float = 100_000.0):
        super().__init__()

        self.prices_df = prices.astype(np.float32)
        self.signals_df = signals.astype(np.float32)
        self.prices = self.prices_df.to_numpy()
        self.signals = self.signals_df.to_numpy()
        self.dates = self.prices_df.index
        self.tickers = list(self.prices_df.columns)
        self.n_assets = len(self.tickers)
        self.initial_cash = float(initial_cash)

        # One Hold/Buy/Sell choice per asset each step.
        self.action_space = spaces.MultiDiscrete(np.full(self.n_assets, 3, dtype=np.int64))
        # Matches the observation _get_observation builds: signals, position
        # ratios, normalised prices, cash ratio, value ratio.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(3 * self.n_assets + 2,),
            dtype=np.float32,
        )

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + np.dot(self.holdings, prices))

    def _get_observation(self) -> np.ndarray:
        current_prices = self.prices[self.current_step]
        # Floor the denominator so a near-zero portfolio can't divide by zero.
        portfolio_value = max(self._portfolio_value(current_prices), self._MIN_PORTFOLIO_VALUE)
        position_ratios = self.holdings * current_prices / portfolio_value
        normalised_prices = current_prices / self.episode_start_prices
        cash_ratio = np.array([self.cash / portfolio_value], dtype=np.float32)
        value_ratio = np.array([portfolio_value / self.initial_cash], dtype=np.float32)

        return np.concatenate(
            [
                self.signals[self.current_step],
                position_ratios,
                normalised_prices,
                cash_ratio,
                value_ratio,
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.episode_start_prices = self.prices[0].copy()
        return self._get_observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        current_prices = self.prices[self.current_step]
        previous_value = self._portfolio_value(current_prices)

        # Sell first so a same-step Buy can draw on the proceeds. Still all-in
        # / all-out for now: no trade sizing or transaction costs yet.
        for asset_index in np.flatnonzero(action == self.SELL):
            self.cash += self.holdings[asset_index] * current_prices[asset_index]
            self.holdings[asset_index] = 0.0

        buy_indexes = np.flatnonzero(action == self.BUY)
        if len(buy_indexes) > 0 and self.cash > 0:
            # Split whatever cash is left evenly across every asset flagged
            # to buy this step, so the first ticker doesn't take it all.
            budget_per_asset = self.cash / len(buy_indexes)
            for asset_index in buy_indexes:
                self.holdings[asset_index] += budget_per_asset / current_prices[asset_index]
            self.cash = 0.0

        self.current_step += 1
        new_prices = self.prices[self.current_step]
        new_value = self._portfolio_value(new_prices)
        reward = new_value - previous_value

        # The episode ends once there's no next day left to price the portfolio on.
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        info = {"portfolio_value": new_value}

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        current_prices = self.prices[self.current_step]
        print(
            f"{self.dates[self.current_step].date()} | "
            f"value={self._portfolio_value(current_prices):,.2f} | "
            f"cash={self.cash:,.2f}"
        )
