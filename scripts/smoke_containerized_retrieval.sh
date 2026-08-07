#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT="justatom-smoke-$(date +%s)-$$"
export COMPOSE_PROJECT_NAME="$PROJECT"
export JUSTATOM_API_PORT="${JUSTATOM_API_PORT:-15555}"
export EMBEDDING_PORT="${EMBEDDING_PORT:-18000}"
export WEAVIATE_HTTP_PORT="${WEAVIATE_HTTP_PORT:-13211}"
export WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-15051}"

cleanup() {
  scripts/services.sh cpu down -v --remove-orphans >/dev/null 2>&1 || true
}

fail() {
  printf 'smoke failure: %s\n' "$1" >&2
  scripts/services.sh cpu logs --no-color >&2 || true
  exit 1
}

wait_http() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in $(seq 1 300); do
    if curl --connect-timeout 2 --max-time 5 --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  fail "$name did not become ready at $url"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v docker >/dev/null || fail "docker is required"
command -v curl >/dev/null || fail "curl is required"
command -v jq >/dev/null || fail "jq is required"

scripts/services.sh cpu up -d --build weaviate embedder-cpu api \
  || fail "CPU services failed to start"
wait_http "embedding health" "http://127.0.0.1:${EMBEDDING_PORT}/health"
wait_http "embedding models" "http://127.0.0.1:${EMBEDDING_PORT}/v1/models"
wait_http "retrieval API" "http://127.0.0.1:${JUSTATOM_API_PORT}/"

index_response="$(
  curl --connect-timeout 5 --max-time 300 --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"dataset_name_or_docs":[{"content":"банк негативов расширяет множество негативных примеров при контрастном обучении информационного поиска.","meta":{"topic":"retrieval"}},{"content":"Qwen3 Embedding преобразует запросы и документы в плотные векторные представления.","meta":{"topic":"embeddings"}},{"content":"Weaviate хранит векторы документов и выполняет поиск ближайших соседей.","meta":{"topic":"storage"}}]}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/indexing"
)" || fail "indexing request failed"
printf '%s' "$index_response" | jq -e '.total_docs == 3' >/dev/null \
  || fail "expected three indexed documents"

search_response="$(
  curl --connect-timeout 5 --max-time 300 --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Что даёт банк негативов при обучении поиска?","top_k":2}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "first search request failed"
printf '%s' "$search_response" | jq -e '.docs[0].meta.topic == "retrieval"' >/dev/null \
  || fail "retrieval document was not ranked first"
if printf '%s' "$search_response" | grep -Eq '\\u[0-9a-fA-F]{4}'; then
  fail "API response escaped readable UTF-8 text"
fi
printf '%s' "$search_response" | grep -Fq 'банк негативов' \
  || fail "Russian result text is missing"

second_response="$(
  curl --connect-timeout 5 --max-time 300 --fail --silent --show-error \
    -H 'Content-Type: application/json' \
    -d '{"text":"Где хранятся векторы документов?","top_k":2}' \
    "http://127.0.0.1:${JUSTATOM_API_PORT}/searching"
)" || fail "second search request failed"
printf '%s' "$second_response" | jq -e '.docs[0].meta.topic == "storage"' >/dev/null \
  || fail "storage document was not ranked first"

model_loads="$(
  scripts/services.sh cpu logs --no-color embedder-cpu \
    | grep -F -c 'Loading from huggingface hub via' || true
)"
[[ "$model_loads" == "1" ]] || fail "expected one model load, observed $model_loads"

printf 'containerized retrieval smoke passed: project=%s\n' "$PROJECT"
