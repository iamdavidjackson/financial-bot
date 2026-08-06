#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"

cleanup() {
  echo "Stopping dev servers..."
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
  wait "$API_PID" "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if ! curl -s -o /dev/null --max-time 2 http://localhost:11434/api/version; then
  echo "Starting Ollama..."
  if command -v brew >/dev/null 2>&1; then
    brew services start ollama
    sleep 2
  else
    echo "Couldn't start Ollama" >&2
  fi
fi

source "$ROOT_DIR/.venv/bin/activate"
(cd "$ROOT_DIR/backend/api" && uvicorn main:app --reload --port 8000) &
API_PID=$!

(cd "$ROOT_DIR/frontend/nextjs" && npm run dev) &
WEB_PID=$!

wait "$API_PID" "$WEB_PID"
