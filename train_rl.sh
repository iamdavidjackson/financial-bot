#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$ROOT_DIR/.venv/bin/activate"

echo "== Step 1/3: training LSTM model =="
python "$ROOT_DIR/backend/training/train_rl_signal_lstm.py"

echo "== Step 2/3: exporting LSTM signals =="
python "$ROOT_DIR/backend/training/export_rl_signals.py"

echo "== Step 3/3: training the PPO portfolio model =="
python "$ROOT_DIR/backend/training/train_rl_agent.py"
