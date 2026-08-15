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

    def __init__(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        trade_fraction: float = 0.25,
        reward_scale: float = 100_000.0,
    ):
        super().__init__()

        self.prices_df = prices.astype(np.float32)
        self.signals_df = signals.astype(np.float32)
        self.prices = self.prices_df.to_numpy()
        self.signals = self.signals_df.to_numpy()
        self.dates = self.prices_df.index
        self.tickers = list(self.prices_df.columns)
        self.n_assets = len(self.tickers)

        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.trade_fraction = float(trade_fraction)
        self.reward_scale = float(reward_scale)

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

    def _execute_trades(self, action: np.ndarray, current_prices: np.ndarray, portfolio_value: float) -> float:
        # I need to process the Sell actions before any Buy. This raises cash first, so a Buy on
        # the same step can draw on proceeds from a Sell on the same step.
        transaction_costs = 0.0

        for asset_index in np.flatnonzero(action == self.SELL):
            # Sell up to trade_fraction of the position, rounded up, capped at what's held.
            shares_to_sell = int(np.ceil(self.holdings[asset_index] * self.trade_fraction))
            shares_to_sell = min(shares_to_sell, int(self.holdings[asset_index]))
            if shares_to_sell <= 0:
                continue

            gross_proceeds = shares_to_sell * current_prices[asset_index]
            cost = gross_proceeds * self.transaction_cost
            self.cash += gross_proceeds - cost
            self.holdings[asset_index] -= shares_to_sell
            transaction_costs += cost

        buy_indexes = np.flatnonzero(action == self.BUY)
        remaining_buys = len(buy_indexes)

        for asset_index in buy_indexes:
            if remaining_buys <= 0 or self.cash <= 0:
                break

            # I split the remaining cash evenly across the assets still waiting to
            # buy this step, so the first ticker in the list doesn't take it all.
            budget = min(
                portfolio_value * self.trade_fraction,
                self.cash / remaining_buys,
            )
            price_with_cost = current_prices[asset_index] * (1 + self.transaction_cost)
            shares_to_buy = int(np.floor(budget / price_with_cost))

            if shares_to_buy > 0:
                gross_cost = shares_to_buy * current_prices[asset_index]
                cost = gross_cost * self.transaction_cost
                self.cash -= gross_cost + cost
                self.holdings[asset_index] += shares_to_buy
                transaction_costs += cost

            remaining_buys -= 1

        return transaction_costs

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        current_prices = self.prices[self.current_step]
        previous_value = self._portfolio_value(current_prices)

        # Trade at today's prices, then move to tomorrow to see the outcome.
        transaction_costs = self._execute_trades(action, current_prices, previous_value)

        self.current_step += 1
        new_prices = self.prices[self.current_step]
        new_value = self._portfolio_value(new_prices)
        raw_reward = new_value - previous_value
        # I scale the dollar change by reward_scale (initial cash) so PPO trains
        # on rewards in a small, stable range instead of raw dollar amounts.
        reward = raw_reward / self.reward_scale

        # The episode ends once there's no next day left to price the portfolio on.
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        info = {
            "portfolio_value": new_value,
            "transaction_costs": transaction_costs,
            "raw_reward": raw_reward,
        }

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        current_prices = self.prices[self.current_step]
        print(
            f"{self.dates[self.current_step].date()} | "
            f"value={self._portfolio_value(current_prices):,.2f} | "
            f"cash={self.cash:,.2f}"
        )
