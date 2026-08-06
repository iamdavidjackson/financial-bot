#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
  echo "Stopping dev servers..."
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
  wait "$API_PID" "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source "$ROOT_DIR/.venv/bin/activate"
(cd "$ROOT_DIR/backend/api" && uvicorn main:app --reload --port 8000) &
API_PID=$!

(cd "$ROOT_DIR/frontend/nextjs" && npm run dev) &
WEB_PID=$!

wait "$API_PID" "$WEB_PID"
