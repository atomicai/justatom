#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT="justatom-api-smoke-$(date +%s)-$$"
export COMPOSE_PROJECT_NAME="$PROJECT"
export JUSTATOM_API_PORT="${JUSTATOM_API_PORT:-15556}"
export WEAVIATE_HTTP_PORT="${WEAVIATE_HTTP_PORT:-13212}"
export WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-15052}"
export FAKE_EMBEDDING_PORT="${FAKE_EMBEDDING_PORT:-18001}"
export EMBEDDING_MODEL="fixture-embedding-model"
export EMBEDDING_BASE_URL="http://host.docker.internal:${FAKE_EMBEDDING_PORT}/v1"

FAKE_LOG="${TMPDIR:-/tmp}/${PROJECT}-embedder.log"
FAKE_PID=""

list_compose_projects() {
  {
    docker ps -a --format '{{.Label "com.docker.compose.project"}}'
    docker volume ls -q --filter label=com.docker.compose.project | while read -r volume; do
      docker volume inspect --format '{{index .Labels "com.docker.compose.project"}}' "$volume"
    done
    docker network ls -q --filter label=com.docker.compose.project | while read -r network; do
      docker network inspect --format '{{index .Labels "com.docker.compose.project"}}' "$network"
    done
  } | sed '/^$/d' | sort -u
}

project_resources() {
  docker ps -aq --filter "label=com.docker.compose.project=${PROJECT}"
  docker volume ls -q --filter "label=com.docker.compose.project=${PROJECT}"
  docker network ls -q --filter "label=com.docker.compose.project=${PROJECT}"
}

check_ports_are_free() {
  python - "$JUSTATOM_API_PORT" "$WEAVIATE_HTTP_PORT" "$WEAVIATE_GRPC_PORT" "$FAKE_EMBEDDING_PORT" <<'PY'
import socket
import sys

for value in sys.argv[1:]:
    port = int(value)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
PY
}

cleanup() {
  scripts/services.sh external down -v --remove-orphans >/dev/null 2>&1 || true
  if [[ -n "$FAKE_PID" ]]; then
    kill "$FAKE_PID" >/dev/null 2>&1 || true
    wait "$FAKE_PID" >/dev/null 2>&1 || true
  fi
  rm -f "$FAKE_LOG"

  local remaining
  remaining="$(project_resources)"
  if [[ -n "$remaining" ]]; then
    printf 'cleanup warning: smoke project resources remain: %s\n' "$remaining" >&2
  else
    printf 'cleanup evidence: project=%s containers/volumes/networks=none\n' "$PROJECT"
  fi
  if check_ports_are_free; then
    printf 'cleanup evidence: ports free=%s,%s,%s,%s\n' \
      "$JUSTATOM_API_PORT" "$WEAVIATE_HTTP_PORT" "$WEAVIATE_GRPC_PORT" "$FAKE_EMBEDDING_PORT"
  else
    printf 'cleanup warning: one or more smoke ports remain occupied\n' >&2
  fi
}

fail() {
  printf 'external-backend smoke failure: %s\n' "$1" >&2
  scripts/services.sh external logs --no-color >&2 || true
  [[ ! -f "$FAKE_LOG" ]] || cat "$FAKE_LOG" >&2
  exit 1
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 150); do
    if curl --connect-timeout 2 --max-time 5 --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail "$name did not become ready at $url"
}

wait_fake_embedding() {
  local url="http://127.0.0.1:${FAKE_EMBEDDING_PORT}/health"
  local attempt
  for attempt in $(seq 1 150); do
    if ! kill -0 "$FAKE_PID" >/dev/null 2>&1; then
      fail "fake embedding endpoint exited before becoming ready at $url"
    fi
    if curl --connect-timeout 2 --max-time 5 --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  fail "fake embedding endpoint did not become ready at $url"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v python >/dev/null || fail "python is required"
command -v docker >/dev/null || fail "docker is required"
command -v curl >/dev/null || fail "curl is required"
command -v jq >/dev/null || fail "jq is required"
python -c 'import hypercorn, quart' || fail "activate the justatom Python environment"

before_projects="$(list_compose_projects)"
printf 'isolation evidence before: compose projects=%s\n' "${before_projects:-none}"
if ! check_ports_are_free; then
  fail "smoke ports must be free: ${JUSTATOM_API_PORT},${WEAVIATE_HTTP_PORT},${WEAVIATE_GRPC_PORT},${FAKE_EMBEDDING_PORT}"
fi
printf 'isolation evidence before: ports free=%s,%s,%s,%s\n' \
  "$JUSTATOM_API_PORT" "$WEAVIATE_HTTP_PORT" "$WEAVIATE_GRPC_PORT" "$FAKE_EMBEDDING_PORT"

FAKE_EMBEDDING_MODEL="$EMBEDDING_MODEL" \
  PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python tests/fixtures/openai_embedding_stub.py >"$FAKE_LOG" 2>&1 &
FAKE_PID=$!
wait_fake_embedding

scripts/services.sh external up -d --build weaviate api \
  || fail "external-backend services failed to start"
wait_http "retrieval API" "http://127.0.0.1:${JUSTATOM_API_PORT}/"

index_response="$(
  curl --connect-timeout 5 --max-time 60 --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"dataset_name_or_docs":[{"content":"банк негативов расширяет множество негативных примеров.","meta":{"topic":"retrieval"}},{"content":"Qwen создаёт эмбеддинги документов.","meta":{"topic":"embeddings"}},{"content":"Weaviate хранит векторы документов.","meta":{"topic":"storage"}}]}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/indexing"
)" || fail "indexing through external endpoint failed"
printf '%s' "$index_response" | jq -e '.total_docs == 3' >/dev/null \
  || fail "expected three indexed documents"

search_response="$(
  curl --connect-timeout 5 --max-time 60 --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Зачем нужен банк негативов?","top_k":1}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "retrieval search through external endpoint failed"
printf '%s' "$search_response" | jq -e '.docs[0].meta.topic == "retrieval"' >/dev/null \
  || fail "retrieval document was not ranked first"
printf '%s' "$search_response" | grep -Fq 'банк негативов' \
  || fail "Russian result text is missing"
if printf '%s' "$search_response" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "API response escaped readable UTF-8 text"
fi

storage_response="$(
  curl --connect-timeout 5 --max-time 60 --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Где хранятся векторы документов?","top_k":1}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "storage search through external endpoint failed"
printf '%s' "$storage_response" | jq -e '.docs[0].meta.topic == "storage"' >/dev/null \
  || fail "storage document was not ranked first"

api_container="$(
  docker ps -q \
    --filter "label=com.docker.compose.project=${PROJECT}" \
    --filter 'label=com.docker.compose.service=api'
)"
[[ -n "$api_container" ]] || fail "could not identify the smoke API container"
docker exec "$api_container" python -c \
  'import importlib.util; assert importlib.util.find_spec("torch") is None' \
  || fail "Torch is installed in the model-free API image"

printf 'model-free API smoke passed: project=%s\n' "$PROJECT"
