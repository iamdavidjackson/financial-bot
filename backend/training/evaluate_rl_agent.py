"""Evaluate the saved PPO portfolio agent on the held-out test period.

This does not retrain anything. It loads the already-trained policy from
``trained_models/ppo_portfolio_agent``, runs it deterministically on the test
split, runs the random-policy baseline across many seeds, and writes:

* ``results_table.csv``                         - final metrics per strategy
* ``portfolio_values.csv``                      - daily portfolio value per strategy (chart data)
* ``random_policy_portfolio_values_by_seed.csv``- every random seed's daily curve
* ``random_policy_metrics_by_seed.csv``         - per-seed random-policy metrics
* ``experiment_notes.md``                       - dates, parameters, model info and observations

Run:
    python backend/training/evaluate_rl_agent.py [--seeds N] [--out DIR]
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from stable_baselines3 import PPO

from core.portfolio_env import make_buy_and_hold_curve, portfolio_metrics
from training.train_rl_agent import (
    INITIAL_CASH,
    MODEL_PATH,
    PORTFOLIO_TICKERS,
    RANDOM_SEED,
    REWARD_SCALE,
    TEST_START,
    TRADE_FRACTION,
    TRAIN_END,
    TRANSACTION_COST,
    evaluate_policy,
    evaluate_random_policy,
    load_rl_data,
    make_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_COUNT = 30
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "final-report" / "experiments"

METRIC_COLS = [
    "final_value",
    "total_return",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "max_drawdown",
]
METRIC_HEADERS = [
    "Final value",
    "Total return",
    "Annualised return",
    "Annualised volatility",
    "Sharpe ratio",
    "Max drawdown",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def run_random_seeds(test_prices, test_signals, seeds):
    """Run one random-policy episode per seed; return (daily curves, per-seed metrics)."""
    curves = {}
    metrics = {}
    for seed in seeds:
        history = evaluate_random_policy(make_env(test_prices, test_signals), seed=seed)
        curve = history["portfolio_value"]
        curves[f"seed_{seed}"] = curve
        metrics[seed] = portfolio_metrics(curve)
    curves_df = pd.DataFrame(curves)
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "seed"
    return curves_df, metrics_df


def _fmt(value, metric):
    if pd.isna(value):
        return "n/a"
    if metric == "final_value":
        return f"${value:,.2f}"
    if metric == "sharpe_ratio":
        return f"{value:.2f}"
    return f"{value:.2%}"


def results_to_markdown(df):
    lines = [
        "| " + " | ".join(["Strategy", *METRIC_HEADERS]) + " |",
        "|" + "|".join(["---", *["---:"] * len(METRIC_COLS)]) + "|",
    ]
    for strategy, row in df.iterrows():
        cells = [str(strategy)] + [_fmt(row[m], m) for m in METRIC_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEED_COUNT, help="number of random-policy seeds, run as 0..N-1"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    args = parser.parse_args()

    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")

    model_file = Path(f"{MODEL_PATH}.zip")
    if not model_file.exists():
        raise SystemExit(f"Saved PPO model not found at {model_file}. Run train_rl_agent.py first.")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_prices, test_signals = load_rl_data(PORTFOLIO_TICKERS)
    test_start = test_prices.index[0].date()
    test_end = test_prices.index[-1].date()
    print(f"Test period: {test_start} to {test_end} ({len(test_prices)} trading days)")

    model = PPO.load(MODEL_PATH)
    print(f"Loaded saved PPO model from {model_file}")

    # PPO deterministic evaluation on the held-out set.
    ppo_curve = evaluate_policy(model, make_env(test_prices, test_signals))["portfolio_value"]

    # Random-policy baseline across many seeds.
    seeds = list(range(args.seeds))
    print(f"Running random-policy baseline for {len(seeds)} seeds ({seeds[0]}..{seeds[-1]})")
    random_curves, random_metrics = run_random_seeds(test_prices, test_signals, seeds)

    buy_and_hold_curve = make_buy_and_hold_curve(test_prices, INITIAL_CASH, TRANSACTION_COST)
    cash_curve = pd.Series(INITIAL_CASH, index=test_prices.index)

    # ---- chart data ----
    chart = pd.DataFrame(
        {
            "PPO": ppo_curve,
            "Buy and hold": buy_and_hold_curve,
            "Random policy (mean)": random_curves.mean(axis=1),
            "Random policy (min)": random_curves.min(axis=1),
            "Random policy (max)": random_curves.max(axis=1),
            "Cash": cash_curve,
        }
    ).dropna()
    chart.index.name = "date"
    chart.round(2).to_csv(out_dir / "portfolio_values.csv")
    random_curves.round(2).to_csv(out_dir / "random_policy_portfolio_values_by_seed.csv")

    # ---- results table ----
    best_seed = random_metrics["final_value"].idxmax()
    worst_seed = random_metrics["final_value"].idxmin()
    summary_rows = {
        "PPO": portfolio_metrics(chart["PPO"]),
        "Buy and hold": portfolio_metrics(chart["Buy and hold"]),
        "Random policy (mean of seeds)": random_metrics.mean().to_dict(),
        "Random policy (std of seeds)": random_metrics.std().to_dict(),
        f"Random policy (best seed = {best_seed})": random_metrics.loc[best_seed].to_dict(),
        f"Random policy (worst seed = {worst_seed})": random_metrics.loc[worst_seed].to_dict(),
        "Cash": portfolio_metrics(chart["Cash"]),
    }
    results_df = pd.DataFrame(summary_rows).T[METRIC_COLS]
    results_df["final_value"] = results_df["final_value"].round(2)
    results_df[METRIC_COLS[1:]] = results_df[METRIC_COLS[1:]].round(4)
    results_df.index.name = "strategy"
    results_df.to_csv(out_dir / "results_table.csv")
    random_metrics.round(4).to_csv(out_dir / "random_policy_metrics_by_seed.csv")

    # ---- experiment notes ----
    ppo_final = results_df.loc["PPO", "final_value"]
    bh_final = results_df.loc["Buy and hold", "final_value"]
    ppo_vs_bh_pp = (results_df.loc["PPO", "total_return"] - results_df.loc["Buy and hold", "total_return"]) * 100
    ppo_vs_rand_pp = (results_df.loc["PPO", "total_return"] - random_metrics["total_return"].mean()) * 100
    n_beat_bh = int((random_metrics["final_value"] > bh_final).sum())
    n_beat_ppo = int((random_metrics["final_value"] > ppo_final).sum())
    model_mtime = datetime.fromtimestamp(model_file.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    notes = f"""# PPO evaluation - experiment notes

_Generated {timestamp} by `backend/training/evaluate_rl_agent.py`. Re-run that script to refresh every file in this directory._

## Run metadata

| Item | Value |
|---|---|
| Git commit | `{git_commit()}` |
| Saved PPO model | `{model_file.relative_to(REPO_ROOT)}` |
| Model file last modified | {model_mtime} |
| PPO evaluation | deterministic (`deterministic=True`), env reset seed {RANDOM_SEED} |
| Random-policy seeds | {len(seeds)} (0-{seeds[-1]}) |

## Data and split

| Item | Value |
|---|---|
| Portfolio tickers | {", ".join(PORTFOLIO_TICKERS)} |
| Price / signal source | `training_data/rl/prices.parquet`, `training_data/rl/lstm_signals_5d.parquet` |
| Training data ends | {TRAIN_END} |
| Test start (configured) | {TEST_START} |
| Test period (actual) | {test_start} to {test_end} ({len(chart)} trading days) |

Refresh the underlying price/signal data with `python backend/training/export_rl_signals.py` if a
later test end date is required.

## Environment parameters

| Parameter | Value |
|---|---|
| Initial cash | ${INITIAL_CASH:,.0f} |
| Transaction cost | {TRANSACTION_COST:.3%} |
| Trade fraction | {TRADE_FRACTION:.0%} |
| Reward scale | {REWARD_SCALE:,.0f} |
| Action space | MultiDiscrete - hold / buy / sell per asset |

## Final results

{results_to_markdown(results_df)}

## Random-policy dispersion across {len(seeds)} seeds

| Statistic | Final value | Total return |
|---|---:|---:|
| Mean | ${random_metrics['final_value'].mean():,.2f} | {random_metrics['total_return'].mean():.2%} |
| Std | ${random_metrics['final_value'].std():,.2f} | {random_metrics['total_return'].std():.2%} |
| Min | ${random_metrics['final_value'].min():,.2f} | {random_metrics['total_return'].min():.2%} |
| Max | ${random_metrics['final_value'].max():,.2f} | {random_metrics['total_return'].max():.2%} |

## Observations

- PPO final value ${ppo_final:,.2f} vs buy-and-hold ${bh_final:,.2f}: a {ppo_vs_bh_pp:+.2f} percentage-point total-return difference.
- PPO total return vs the random-policy mean: {ppo_vs_rand_pp:+.2f} percentage points.
- Random seeds beating buy-and-hold: {n_beat_bh}/{len(seeds)}.
- Random seeds beating PPO: {n_beat_ppo}/{len(seeds)}.
- The evaluation reuses one PPO training seed ({RANDOM_SEED}); it measures the saved policy, not PPO training stability.

## Output files

| File | Contents |
|---|---|
| `results_table.csv` | The final results table above. |
| `portfolio_values.csv` | Daily portfolio value per strategy, plus the random-policy mean/min/max envelope (chart data). |
| `random_policy_portfolio_values_by_seed.csv` | Every random seed's daily portfolio-value curve. |
| `random_policy_metrics_by_seed.csv` | Per-seed random-policy metrics. |
"""
    (out_dir / "experiment_notes.md").write_text(notes)

    print()
    print(results_df.to_string())
    print()
    print(f"Wrote 5 files to {out_dir}")


if __name__ == "__main__":
    main()
