#!/usr/bin/env bash
set -euo pipefail

# Evaluate one encoder model across multiple dataset presets.
# The script is config-first: it uses `configs/evaluate.yaml` as the source of
# truth and only applies a small set of launch-time overrides.
# Example:
#   bash scripts/run_eval_model_on_datasets.sh --embedding-model intfloat/multilingual-e5-small
#
# Common overrides:
#   DATASET_IDS="demo-eval justatom miracl-ru" \
#   CUDA_VISIBLE_DEVICES=0 \
#   ENABLE_WANDB=1 \
#   WANDB_PROJECT=justatom-evals \
#   bash scripts/run_eval_model_on_datasets.sh --embedding-model intfloat/multilingual-e5-small

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-intfloat/multilingual-e5-small}"
CONFIG_PATH="${CONFIG_PATH:-configs/evaluate.yaml}"
DATASET_IDS_RAW="${DATASET_IDS:-demo-eval justatom miracl-ru}"
COLLECTION_PREFIX="${COLLECTION_PREFIX:-EvalBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/eval_runs}"
SEARCH_MODE="${SEARCH_MODE:-}"
TOP_K="${TOP_K:-}"
SEARCH_BATCH_SIZE="${SEARCH_BATCH_SIZE:-}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-}"
WEAVIATE_URL="${WEAVIATE_URL:-http://${WEAVIATE_HOST:-localhost}:${WEAVIATE_PORT:-2211}}"
WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-50051}"
INDEX_FLUSH_COLLECTION="${INDEX_FLUSH_COLLECTION:-}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-justatom-evals}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-eval-bench}"
WANDB_TAGS="${WANDB_TAGS:-}"
QUERY_PREFIX="${QUERY_PREFIX:-}"
DOCUMENT_PREFIX="${DOCUMENT_PREFIX:-}"
AUTO_E5_PREFIXES="${AUTO_E5_PREFIXES:-0}"
RETRIEVAL_ALPHA="${RETRIEVAL_ALPHA:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_eval_model_on_datasets.sh [options]

Options:
  --embedding-model MODEL
  --search-mode keyword|vector|hybrid
  --query-prefix TEXT
  --document-prefix TEXT
  --weaviate-url URL
  --weaviate-grpc-port PORT
  --alpha VALUE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding-model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --search-mode) SEARCH_MODE="$2"; shift 2 ;;
    --query-prefix) QUERY_PREFIX="$2"; shift 2 ;;
    --document-prefix) DOCUMENT_PREFIX="$2"; shift 2 ;;
    --weaviate-url) WEAVIATE_URL="$2"; shift 2 ;;
    --weaviate-grpc-port) WEAVIATE_GRPC_PORT="$2"; shift 2 ;;
    --alpha) RETRIEVAL_ALPHA="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/$TIMESTAMP"
mkdir -p "$RUN_DIR"

sanitize_name() {
  printf '%s' "$1" | tr '/: ?&=' '_' | tr -s '_'
}

weaviate_class_name() {
  local raw="$1"
  raw="$(printf '%s' "$raw" | sed -E 's/[^[:alnum:]]+/ /g')"
  local out=""
  local part=""
  for part in $raw; do
    local lower="$(printf '%s' "$part" | tr '[:upper:]' '[:lower:]')"
    out+="$(printf '%s' "${lower:0:1}" | tr '[:lower:]' '[:upper:]')${lower:1}"
  done
  if [[ -z "$out" ]]; then
    out="EvalBench"
  fi
  if [[ ! "$out" =~ ^[A-Z] ]]; then
    out="Eval${out}"
  fi
  printf '%s' "$out"
}

trim() {
  local value="$1"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"
  printf '%s' "$value"
}

should_use_e5_prefixes() {
  local model_lower
  model_lower="$(printf '%s' "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')"
  [[ "$model_lower" == *"/e5"* || "$model_lower" == *"-e5-"* || "$model_lower" == e5* ]]
}

if [[ "$AUTO_E5_PREFIXES" == "1" ]] && [[ -z "$QUERY_PREFIX" ]] && should_use_e5_prefixes; then
  QUERY_PREFIX="query: "
fi

if [[ "$AUTO_E5_PREFIXES" == "1" ]] && [[ -z "$DOCUMENT_PREFIX" ]] && should_use_e5_prefixes; then
  DOCUMENT_PREFIX="passage: "
fi

case "$SEARCH_MODE" in ""|keyword|vector|hybrid) ;; *) echo "invalid search mode: $SEARCH_MODE; expected keyword, vector, or hybrid" >&2; exit 2 ;; esac

printf 'Repo root: %s\n' "$REPO_ROOT"
printf 'Config: %s\n' "$CONFIG_PATH"
printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Model: %s\n' "$MODEL_NAME_OR_PATH"
printf 'Datasets: %s\n' "$DATASET_IDS_RAW"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "$CUDA_VISIBLE_DEVICES"
if [[ -n "$SEARCH_MODE" ]]; then
  printf 'Search mode override: %s\n' "$SEARCH_MODE"
fi
if [[ -n "$QUERY_PREFIX" || -n "$DOCUMENT_PREFIX" ]]; then
  printf 'Prefixes: query=%q document=%q\n' "$QUERY_PREFIX" "$DOCUMENT_PREFIX"
fi

upload_csv_to_wandb() {
  local csv_path="$1"
  local dataset_id="$2"
  local run_name="$3"

  if [[ "$ENABLE_WANDB" != "1" ]]; then
    return 0
  fi

  if ! command -v python >/dev/null 2>&1; then
    printf 'Skipping wandb upload for %s: python not found\n' "$dataset_id" >&2
    return 0
  fi

  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("wandb") else 1)
PY
  then
    printf 'Skipping wandb upload for %s: wandb is not installed in the active environment\n' "$dataset_id" >&2
    return 0
  fi

  CSV_PATH="$csv_path" \
  DATASET_ID="$dataset_id" \
  MODEL_NAME="$MODEL_NAME_OR_PATH" \
  SEARCH_MODE="$SEARCH_MODE" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  WANDB_RUN_NAME="$run_name" \
  WANDB_TAGS="$WANDB_TAGS" \
  python - <<'PY'
import csv
import os
from pathlib import Path

import wandb

csv_path = Path(os.environ["CSV_PATH"])
dataset_id = os.environ["DATASET_ID"]
model_name = os.environ["MODEL_NAME"]
search_mode = os.environ["SEARCH_MODE"]
project = os.environ["WANDB_PROJECT"]
entity = os.environ.get("WANDB_ENTITY") or None
run_name = os.environ["WANDB_RUN_NAME"]
raw_tags = os.environ.get("WANDB_TAGS", "")
tags = [tag for tag in raw_tags.split(",") if tag]
tags.extend([dataset_id, search_mode, "eval"])

run = wandb.init(
    project=project,
    entity=entity,
    name=run_name,
    tags=tags,
    config={
        "model_name_or_path": model_name,
        "dataset_id": dataset_id,
        "search_mode": search_mode,
        "csv_path": str(csv_path),
    },
)

with csv_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

summary_payload = {}
for row in rows:
    metric_name = row.get("name")
    if not metric_name:
        continue
    try:
        summary_payload[f"metrics/{metric_name}/mean"] = float(row["mean"])
    except (TypeError, ValueError, KeyError):
        pass
    try:
        summary_payload[f"metrics/{metric_name}/std"] = float(row["std"])
    except (TypeError, ValueError, KeyError):
        pass

if summary_payload:
    wandb.log(summary_payload)

artifact = wandb.Artifact(
    name=f"eval-results-{dataset_id}",
    type="evaluation",
    metadata={
        "dataset_id": dataset_id,
        "model_name_or_path": model_name,
        "search_mode": search_mode,
    },
)
artifact.add_file(str(csv_path))
run.log_artifact(artifact)
run.finish()
PY
}

IFS=' ' read -r -a DATASET_IDS <<< "$DATASET_IDS_RAW"

for dataset_id in "${DATASET_IDS[@]}"; do
  dataset_id="$(trim "$dataset_id")"
  if [[ -z "$dataset_id" ]]; then
    continue
  fi

  dataset_slug="$(sanitize_name "$dataset_id")"
  model_slug="$(sanitize_name "$MODEL_NAME_OR_PATH")"
  collection_name="$(weaviate_class_name "${COLLECTION_PREFIX}_${dataset_slug}_${TIMESTAMP}")"
  dataset_out_dir="$RUN_DIR/$dataset_slug"
  mkdir -p "$dataset_out_dir"

  cmd=(
    python -m justatom.api.eval
    --config "$CONFIG_PATH"
    --dataset.id "$dataset_id"
    --embedding-model "$MODEL_NAME_OR_PATH"
    --collection-name "$collection_name"
    --save-results-to-dir "$dataset_out_dir"
    --weaviate-url "$WEAVIATE_URL"
    --weaviate-grpc-port "$WEAVIATE_GRPC_PORT"
  )

  if [[ -n "$SEARCH_MODE" ]]; then
    cmd+=(--search-mode "$SEARCH_MODE")
  fi
  if [[ -n "$TOP_K" ]]; then
    cmd+=(--top-k "$TOP_K")
  fi
  if [[ -n "$SEARCH_BATCH_SIZE" ]]; then
    cmd+=(--search-batch-size "$SEARCH_BATCH_SIZE")
  fi
  if [[ -n "$INDEX_BATCH_SIZE" ]]; then
    cmd+=(--index-batch-size "$INDEX_BATCH_SIZE")
  fi
  if [[ -n "$INDEX_FLUSH_COLLECTION" ]]; then
    cmd+=(--flush-collection)
  fi

  if [[ -n "$QUERY_PREFIX" ]]; then
    cmd+=(--query-prefix "$QUERY_PREFIX")
  fi
  if [[ -n "$DOCUMENT_PREFIX" ]]; then
    cmd+=(--document-prefix "$DOCUMENT_PREFIX")
  fi
  if [[ -n "$RETRIEVAL_ALPHA" ]]; then
    cmd+=(--alpha "$RETRIEVAL_ALPHA")
  fi

  printf '\n[%s] Evaluating dataset.id=%s\n' "$(date +%H:%M:%S)" "$dataset_id"
  printf '[%s] Collection: %s\n' "$(date +%H:%M:%S)" "$collection_name"
  printf '[%s] Output dir: %s\n' "$(date +%H:%M:%S)" "$dataset_out_dir"
  printf '[%s] Command:' "$(date +%H:%M:%S)"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}"

  latest_csv="$(ls -1t "$dataset_out_dir"/*.csv 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_csv" ]]; then
    printf 'No CSV metrics file was produced for dataset.id=%s\n' "$dataset_id" >&2
    exit 1
  fi

  cp "$latest_csv" "$RUN_DIR/${dataset_slug}__${model_slug}.csv"
  printf '[%s] Metrics CSV: %s\n' "$(date +%H:%M:%S)" "$latest_csv"

  if [[ "$ENABLE_WANDB" == "1" ]]; then
    wandb_run_name="${WANDB_RUN_PREFIX}-${dataset_slug}-${model_slug}-${TIMESTAMP}"
    printf '[%s] Uploading to wandb: %s\n' "$(date +%H:%M:%S)" "$wandb_run_name"
    upload_csv_to_wandb "$latest_csv" "$dataset_id" "$wandb_run_name"
  fi
done

printf '\nAll evaluations finished. Consolidated outputs are in %s\n' "$RUN_DIR"
