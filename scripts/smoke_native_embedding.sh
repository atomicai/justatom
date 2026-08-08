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

check_embedding_port_is_free() {
  [[ -n "$EMBEDDING_PYTHON" ]] || return 1
  "$EMBEDDING_PYTHON" - "$EMBEDDING_PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(sys.argv[1])))
PY
}

terminate_server() {
  local attempt
  local wait_status

  if [[ -z "$SERVER_PID" ]]; then
    return 0
  fi
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    if ! kill -TERM "$SERVER_PID" >/dev/null 2>&1 && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      return 1
    fi
    for attempt in $(seq 1 10); do
      if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      if ! kill -KILL "$SERVER_PID" >/dev/null 2>&1 && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        return 1
      fi
    fi
  fi

  if wait "$SERVER_PID" >/dev/null 2>&1; then
    wait_status=0
  else
    wait_status=$?
  fi
  case "$wait_status" in
    0|137|143) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup() {
  local main_status=$?
  local cleanup_failed=0

  trap - EXIT INT TERM

  if ! terminate_server; then
    printf 'cleanup failure: embedding server could not be terminated and reaped\n' >&2
    cleanup_failed=1
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    printf 'cleanup failure: embedding server remains alive\n' >&2
    cleanup_failed=1
  else
    printf 'cleanup evidence: embedding server stopped and reaped\n'
  fi
  if ! check_embedding_port_is_free; then
    printf 'cleanup failure: embedding port remains occupied: %s\n' "$EMBEDDING_PORT" >&2
    cleanup_failed=1
  else
    printf 'cleanup evidence: port free=%s\n' "$EMBEDDING_PORT"
  fi
  if ! rm -f "$LOG_FILE"; then
    printf 'cleanup failure: embedding log could not be removed\n' >&2
    cleanup_failed=1
  else
    printf 'cleanup evidence: embedding log removed=%s\n' "$LOG_FILE"
  fi

  if (( main_status == 0 && cleanup_failed )); then
    exit 1
  fi
  exit "$main_status"
}

fail() {
  printf 'native MPS smoke failure: %s\n' "$1" >&2
  [[ ! -f "$LOG_FILE" ]] || cat "$LOG_FILE" >&2
  exit 1
}

ensure_server_is_alive() {
  if [[ -z "$SERVER_PID" ]] || ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    fail "embedding server exited before smoke completion"
  fi
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
if ! check_embedding_port_is_free; then
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
if printf '%s' "$first_response" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "first response escaped readable UTF-8 output"
fi
first_dimension="$(
  printf '%s' "$first_response" | jq -er --arg model "$EMBEDDING_MODEL" \
    'select(.model == $model) | .data as $data | select(($data | length) == 2 and $data[0].index == 0 and $data[1].index == 1 and ($data[0].embedding | length) > 0 and (($data[0].embedding | length) == ($data[1].embedding | length))) | ($data[0].embedding | length)'
)" || fail "first MPS response lacks model identity, ordered non-empty equal dimensions"

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
if printf '%s' "$second_response" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "second response escaped readable UTF-8 output"
fi
printf '%s' "$second_response" | jq -e \
  --arg model "$EMBEDDING_MODEL" \
  --argjson dimension "$first_dimension" \
  'select(.model == $model) | .data as $data | select(($data | length) == 1 and $data[0].index == 0 and ($data[0].embedding | length) > 0 and (($data[0].embedding | length) == $dimension))' \
  >/dev/null || fail "second MPS response lacks model identity, ordered non-empty dimensions, or first-call dimension"

model_loads="$(grep -E -c 'Loading from huggingface hub via|Model found locally at' "$LOG_FILE" || true)"
[[ "$model_loads" == "1" ]] || fail "expected one model load, observed $model_loads"

ensure_server_is_alive
printf 'native MPS embedding smoke passed: model=%s port=%s model_loads=%s\n' \
  "$EMBEDDING_MODEL" "$EMBEDDING_PORT" "$model_loads"
