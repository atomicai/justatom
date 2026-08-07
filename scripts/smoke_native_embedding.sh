#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  printf 'native MPS smoke skipped: Apple Silicon macOS is required\n'
  exit 0
fi

export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
export EMBEDDING_DEVICE=mps
export EMBEDDING_PORT="${EMBEDDING_PORT:-18002}"
LOG_FILE="${TMPDIR:-/tmp}/justatom-native-mps-embedding-$$.log"
SERVER_PID=""
EMBEDDING_PYTHON=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$LOG_FILE"
  exit "$status"
}

fail() {
  printf 'native MPS smoke failure: %s\n' "$1" >&2
  [[ ! -f "$LOG_FILE" ]] || cat "$LOG_FILE" >&2
  exit 1
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 360); do
    if curl --connect-timeout 2 --max-time 5 --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail "$name did not become ready at $url"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v conda >/dev/null || fail "conda is required"
command -v curl >/dev/null || fail "curl is required"
command -v jq >/dev/null || fail "jq is required"

EMBEDDING_PYTHON="$(conda run -n justatom python -c 'import sys; print(sys.executable)')" \
  || fail "the justatom Python environment is required"
"$EMBEDDING_PYTHON" -c 'import torch; assert torch.backends.mps.is_available()' \
  || fail "PyTorch MPS is unavailable"
if ! "$EMBEDDING_PYTHON" - "$EMBEDDING_PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(sys.argv[1])))
PY
then
  fail "port ${EMBEDDING_PORT} is already in use"
fi

"$EMBEDDING_PYTHON" - "$EMBEDDING_PORT" >"$LOG_FILE" 2>&1 <<'PY' &
import asyncio
import os
import sys

from justatom.api.hypercorn_server import serve_app
from justatom.api.serve_embeddings import build_embedding_app

asyncio.run(serve_app(build_embedding_app(), host="127.0.0.1", port=int(sys.argv[1])))
PY
SERVER_PID=$!

wait_http "embedding health" "http://127.0.0.1:${EMBEDDING_PORT}/health"
wait_http "embedding models" "http://127.0.0.1:${EMBEDDING_PORT}/v1/models"

models_response="$(
  curl --connect-timeout 2 --max-time 5 --fail --silent --show-error \
    "http://127.0.0.1:${EMBEDDING_PORT}/v1/models"
)" || fail "embedding model request failed"
printf '%s' "$models_response" | jq -e \
  --arg model "$EMBEDDING_MODEL" \
  '.data | length == 1 and .[0].id == $model' >/dev/null \
  || fail "embedding server reported an unexpected model"

first_request="{\"model\":\"${EMBEDDING_MODEL}\",\"input\":[\"русский запрос\",\"English passage\"],\"encoding_format\":\"float\"}"
if printf '%s' "$first_request" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "first request escaped readable UTF-8 input"
fi
first_response="$(
  curl --connect-timeout 5 --max-time 300 --fail --silent --show-error \
    -H 'Content-Type: application/json; charset=utf-8' \
    -d "$first_request" \
    "http://127.0.0.1:${EMBEDDING_PORT}/v1/embeddings"
)" || fail "first MPS embedding request failed"
printf '%s' "$first_response" | jq -e \
  '.data as $data | ($data | length) == 2 and $data[0].index == 0 and $data[1].index == 1 and ($data[0].embedding | length) > 0 and (($data[0].embedding | length) == ($data[1].embedding | length))' \
  >/dev/null || fail "first MPS response lacks ordered, non-empty equal dimensions"

second_request="{\"model\":\"${EMBEDDING_MODEL}\",\"input\":\"повторный русский запрос\",\"encoding_format\":\"float\"}"
if printf '%s' "$second_request" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "second request escaped readable UTF-8 input"
fi
second_response="$(
  curl --connect-timeout 5 --max-time 300 --fail --silent --show-error \
    -H 'Content-Type: application/json; charset=utf-8' \
    -d "$second_request" \
    "http://127.0.0.1:${EMBEDDING_PORT}/v1/embeddings"
)" || fail "second MPS embedding request failed"
printf '%s' "$second_response" | jq -e \
  '.data | length == 1 and .[0].index == 0 and (.[0].embedding | length) > 0' \
  >/dev/null || fail "second MPS response lacks ordered non-empty dimensions"

model_loads="$(grep -E -c 'Loading from huggingface hub via|Model found locally at' "$LOG_FILE" || true)"
[[ "$model_loads" == "1" ]] || fail "expected one model load, observed $model_loads"

printf 'native MPS embedding smoke passed: model=%s port=%s model_loads=%s\n' \
  "$EMBEDDING_MODEL" "$EMBEDDING_PORT" "$model_loads"
