"""Train the PPO portfolio agent.

Based on guidance here:
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
  (custom Gymnasium environment + check_env)
- https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
  (vec_env/model/train-predict usage example)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean

from core.portfolio_env import PortfolioEnv, make_buy_and_hold_curve, portfolio_metrics, validate_rl_data
from core.tickers import TICKER_GROUPS

# Need to include timestamps with the logs 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
PORTFOLIO_TICKERS = TICKER_GROUPS["Utilities"][:5]
LSTM_SIGNAL_HORIZON = 5

INITIAL_CASH = 100_000.0
TRANSACTION_COST = 0.001
TRADE_FRACTION = 0.25
REWARD_SCALE = INITIAL_CASH

TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

PPO_TIMESTEPS = 25_000

RL_DATA_DIR = Path(__file__).resolve().parent.parent / "training_data" / "rl"
PRICE_PATH = RL_DATA_DIR / "prices.parquet"
SIGNAL_PATH = RL_DATA_DIR / f"lstm_signals_{LSTM_SIGNAL_HORIZON}d.parquet"

MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "ppo_portfolio_agent"
MODEL_PATH = MODEL_DIR / "ppo_portfolio_agent"

TENSORBOARD_LOG_DIR = Path(__file__).resolve().parent.parent / "training_logs" / "ppo"

class IterationLoggingCallback(BaseCallback):
    # Logs a timestamped line after each PPO rollout/update iteration.
    def __init__(self):
        super().__init__()
        self.iteration = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        logger.info(
            "PPO iteration %d | timesteps=%d | ep_rew_mean=%.4f",
            self.iteration,
            self.num_timesteps,
            ep_rew_mean,
        )


def load_rl_data(tickers):
    # Loads aligned prices and LSTM signals, split into train/test.
    prices = pd.read_parquet(PRICE_PATH)[tickers]
    signals = pd.read_parquet(SIGNAL_PATH)[tickers]
    common_dates = prices.index.intersection(signals.index)
    prices = prices.loc[common_dates]
    signals = signals.loc[common_dates]

    validate_rl_data(prices, signals, tickers)

    train_prices = prices.loc[prices.index <= TRAIN_END]
    train_signals = signals.loc[train_prices.index]
    test_prices = prices.loc[prices.index >= TEST_START]
    test_signals = signals.loc[test_prices.index]

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
    # Run one random-action episode as an evaluation baseline.
    obs, info = environment.reset(seed=seed)
    environment.action_space.seed(seed)
    terminated = truncated = False

    while not (terminated or truncated):
        action = environment.action_space.sample()
        obs, reward, terminated, truncated, info = environment.step(action)

    return pd.DataFrame(environment.history).set_index("date")


def train() -> pd.DataFrame:
    logger.info("Starting PPO training run (timesteps=%s)", PPO_TIMESTEPS)

    train_prices, train_signals, test_prices, test_signals = load_rl_data(PORTFOLIO_TICKERS)

    logger.info("Train dates: %s to %s", train_prices.index[0].date(), train_prices.index[-1].date())
    logger.info("Test dates:  %s to %s", test_prices.index[0].date(), test_prices.index[-1].date())

    # check_env verifies Gymnasium API compliance before spending time on training.
    check_env(make_env(train_prices.iloc[:100], train_signals.iloc[:100]), warn=True)

    vec_env = make_vec_env(lambda: make_env(train_prices, train_signals), n_envs=1, seed=RANDOM_SEED)

    policy_kwargs = {
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=RANDOM_SEED,
        tensorboard_log=str(TENSORBOARD_LOG_DIR),
    )
    model.learn(total_timesteps=PPO_TIMESTEPS, callback=IterationLoggingCallback(), progress_bar=True)

    # Save the trained PPO model and evaluate it on the test set.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    logger.info("Saved PPO policy to %s.zip", MODEL_PATH)

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

    logger.info("Finished PPO training run")
    return results_df


if __name__ == "__main__":
    print(train().to_string())
