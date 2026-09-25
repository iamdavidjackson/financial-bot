"""Custom Gymnasium environment for the PPO portfolio agent.

Based on guidance here:
https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
"""

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


def validate_rl_data(prices: pd.DataFrame, signals: pd.DataFrame, tickers) -> None:
    """Check that price and LSTM-signal inputs are uasble."""
    # Check these values to so we can fail fast if there is a problem.
    if not prices.index.equals(signals.index):
        raise ValueError("index mismatch error")
    if list(prices.columns) != list(signals.columns):
        raise ValueError("column mismatch error")
    if list(prices.columns) != list(tickers):
        raise ValueError("ticker mismatch error")
    if prices.isna().any().any() or signals.isna().any().any():
        raise ValueError("missing values error")
    if (prices <= 0).any().any():
        raise ValueError("negative prices are not allowed")
    if ((signals < 0) | (signals > 1)).any().any():
        raise ValueError("LSTM values must be between 0 and 1")
    if len(prices) < 3:
        raise ValueError("not enough data error")


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
        # validate data so we can fail fast if there is a problem.
        validate_rl_data(prices, signals, prices.columns)

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
        self.action_space = spaces.MultiDiscrete(
            np.full(self.n_assets, 3, dtype=np.int64)
        )
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
        portfolio_value = max(
            self._portfolio_value(current_prices), self._MIN_PORTFOLIO_VALUE
        )
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
        # start with all cash on day 0 of the given price window.
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.episode_start_prices = self.prices[0].copy()
        self.history = [
            {
                "date": self.dates[0],
                "portfolio_value": self.initial_cash,
                "cash": self.cash,
                "transaction_costs": 0.0,
                "raw_reward": 0.0,
                "action": np.full(self.n_assets, self.HOLD, dtype=np.int64),
            }
        ]
        return self._get_observation(), {}

    def _execute_trades(
        self, action: np.ndarray, current_prices: np.ndarray, portfolio_value: float
    ) -> float:
        # process the Sell actions before Buy actions. This raises cash first, so a Buy on
        # the same step can draw on proceeds from a Sell on the same step.
        transaction_costs = 0.0

        for asset_index in np.flatnonzero(action == self.SELL):
            # Sell up to trade_fraction of the position, rounded up, capped at what's held.
            shares_to_sell = int(
                np.ceil(self.holdings[asset_index] * self.trade_fraction)
            )
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

            # split the remaining cash evenly
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
        # the dollar reward is scaled down to keep the PPO loss function in a reasonable range.
        reward = raw_reward / self.reward_scale

        # The episode ends once there's no next day left to price the portfolio on.
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        info = {
            "date": self.dates[self.current_step],
            "portfolio_value": new_value,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "transaction_costs": transaction_costs,
            "raw_reward": raw_reward,
            "action": action.copy(),
        }
        self.history.append(info)

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        current_prices = self.prices[self.current_step]
        print(
            f"{self.dates[self.current_step].date()} | "
            f"value={self._portfolio_value(current_prices):,.2f} | "
            f"cash={self.cash:,.2f}"
        )


def make_buy_and_hold_curve(
    prices: pd.DataFrame, initial_cash: float, transaction_cost: float
) -> pd.Series:
    # I buy an equal dollar amount of each ticker on day 1 and hold to the end.
    # This is the baseline I compare PPO against.
    first_prices = prices.iloc[0].to_numpy()
    allocation = initial_cash / len(prices.columns)
    shares = np.floor(allocation / (first_prices * (1 + transaction_cost)))
    purchase_cost = np.sum(shares * first_prices * (1 + transaction_cost))
    remaining_cash = initial_cash - purchase_cost
    values = remaining_cash + prices.to_numpy() @ shares
    return pd.Series(values, index=prices.index, name="Buy and hold")


def portfolio_metrics(values: pd.Series) -> dict:
    # These are the metrics I use to compare PPO against the baseline.
    returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1
    # There are only 252 trading days in a year instead of 365.
    annualised_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
    annualised_volatility = returns.std() * np.sqrt(252)
    # Sharpe ratio is the mean return divided by the standard deviation of returns, annualised.
    sharpe_ratio = (
        returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    )
    drawdown = values / values.cummax() - 1

    return {
        "final_value": values.iloc[-1],
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": drawdown.min(),
    }
