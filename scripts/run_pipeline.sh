#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
METHOD="${METHOD:-vanilla}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-intfloat/multilingual-e5-small}"
DATASET_IDS_RAW="${DATASET_IDS:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/pipeline_runs}"
TABLE_RESULTS_PATH=""
RUN_MODE="full"
RUN_BASELINE=1
WANDB_MODE_VALUE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-justatom}"
WEAVIATE_HOST_VALUE="${WEAVIATE_HOST:-localhost}"
WEAVIATE_PORT_VALUE="${WEAVIATE_PORT:-2211}"
WEAVIATE_URL="${WEAVIATE_URL:-http://${WEAVIATE_HOST_VALUE}:${WEAVIATE_PORT_VALUE}}"
WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-50051}"
SEARCH_MODE="${SEARCH_MODE:-}"
SEARCH_TOP_K="${TOP_K:-}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
AUTO_E5_PREFIXES=0
QUERY_PREFIX=""
DOCUMENT_PREFIX=""
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"
RETRIEVAL_ALPHA="${RETRIEVAL_ALPHA:-}"
NSAMPLES=""

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
TEMPERATURE="${TEMPERATURE:-0.05}"
LR_ENCODER="${LR_ENCODER:-2e-5}"
LR_HEADS="${LR_HEADS:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_QUERY_SEQ_LEN=""
EXPERIMENT_ROLE="canonical"

EXPLICIT_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline.sh --dataset-ids justatom,meme-russian-ir [options]

Core options:
  --method vanilla|atom_gate|atomic
  --model MODEL
  --dataset-ids IDS
  --batch-size N
  --epochs N
  --grad-acc-steps N
  --temperature VALUE
  --experiment-role canonical|ablation
  --nsamples N
  --output-root DIR
  --table-results PATH
  --eval-only
  --no-baseline

Component overrides:
  --alpha-gate-layers N
  --alpha-gate-hidden-dim N
  --alpha-gate-dropout P
  --memory-bank-size N
  --memory-bank-warmup-steps N
  --memory-bank-mining all|random|hard|mixed
  --memory-bank-hard-negatives N
  --memory-bank-random-negatives N
  --memory-bank-hard-warmup-steps N
  --memory-bank-hard-ramp-steps N
  --memory-bank-collision-threshold VALUE
  --memory-bank-collision-beta VALUE
  --memory-bank-margin-mode off|constant|query
  --memory-bank-margin-base VALUE
  --memory-bank-margin-scale VALUE
  --memory-bank-margin-min VALUE
  --memory-bank-margin-max VALUE
  --memory-bank-admission-beta VALUE
  --memory-bank-margin-reg-weight VALUE

Evaluation options:
  --wandb-mode disabled|offline|online
  --wandb-project NAME
  --weaviate-url URL
  --weaviate-grpc-port PORT
  --eval-batch-size N
  --search-mode keyword|vector|hybrid
  --embedding-backend local|openai-compatible
  --embedding-base-url URL
  --embedding-api-key KEY
  --top-k N
  --index-batch-size N
  --query-prefix TEXT
  --document-prefix TEXT
  --alpha VALUE
  --auto-e5-prefixes
EOF
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

slugify() {
  printf '%s' "$1" | tr '/:@ ?&=,' '________' | tr -cd '[:alnum:]_.-'
}

weaviate_class_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import re
import sys

parts = re.findall(r"[A-Za-z0-9]+", sys.argv[1])
name = "".join(part[:1].upper() + part[1:].lower() for part in parts) or "PipelineRun"
print(name if name[0].isalpha() and name[0].isupper() else f"Pipeline{name}", end="")
PY
}

resolve_dataset_id() {
  case "$1" in
    boolq) printf 'boolq-ru' ;;
    electrical|electrical-engineering) printf 'electrical-engineering-ru' ;;
    miracl) printf 'miracl-ru' ;;
    meme) printf 'meme-russian-ir' ;;
    *) printf '%s' "$1" ;;
  esac
}

resolve_eval_dataset_id() {
  local dataset_id="$1"
  if [[ -f "$REPO_ROOT/configs/dataset/${dataset_id}-dev.yaml" ]]; then
    printf '%s-dev' "$dataset_id"
  else
    printf '%s' "$dataset_id"
  fi
}

should_use_e5_prefixes() {
  local model_lower
  model_lower="$(printf '%s' "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')"
  [[ "$model_lower" == *"e5"* ]]
}

check_weaviate() {
  local base_url
  base_url="$(trim "$WEAVIATE_URL")"
  while [[ "$base_url" == */ ]]; do
    base_url="${base_url%/}"
  done
  case "$base_url" in
    http://*|https://*) ;;
    *) return 1 ;;
  esac
  curl --silent --fail --max-time 3 \
    "${base_url}/v1/.well-known/ready" >/dev/null 2>&1
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$SUMMARY_LOG"
}

log_rss() {
  "$PYTHON_BIN" -m justatom.tooling.resources \
    --label "$1" \
    --pid "$$" \
    --top 8 | tee -a "$SUMMARY_LOG"
}

run_cmd() {
  local label="$1"
  local logfile="$2"
  shift 2
  log "START $label"
  log_rss "before $label"
  set +e
  "$@" 2>&1 | tee "$logfile"
  local code=${PIPESTATUS[0]}
  set -e
  log_rss "after $label"
  log "END $label exit=$code"
  return "$code"
}

latest_csv() {
  local directory="$1"
  ls -1t "$directory"/*.csv 2>/dev/null | head -n 1 || true
}

extract_metric() {
  local path="$1"
  local metric="$2"
  [[ -f "$path" ]] || { printf 'NA'; return; }
  awk -F',' -v metric="$metric" '$1 == metric { print $2; found=1; exit } END { if (!found) print "NA" }' "$path"
}

EVAL_LAST_CSV=""
evaluate_model() {
  local label="$1" logfile="$2" model_ref="$3" collection="$4" output_dir="$5" dataset_id="$6"
  mkdir -p "$output_dir"
  local command=(
    "$PYTHON_BIN" -m justatom.api.eval
    --config configs/evaluate.yaml
    --embedding-model "$model_ref"
    --collection-name "$collection"
    --save-results-to-dir "$output_dir"
    --flush-collection
    --search-batch-size "$EVAL_BATCH_SIZE"
    --weaviate-url "$WEAVIATE_URL"
    --weaviate-grpc-port "$WEAVIATE_GRPC_PORT"
    --dataset.id "$dataset_id"
  )
  [[ -z "$SEARCH_MODE" ]] || command+=(--search-mode "$SEARCH_MODE")
  [[ -z "$EMBEDDING_BACKEND" ]] || command+=(--embedding-backend "$EMBEDDING_BACKEND")
  [[ -z "$EMBEDDING_BASE_URL" ]] || command+=(--embedding-base-url "$EMBEDDING_BASE_URL")
  [[ -z "$EMBEDDING_API_KEY" ]] || command+=(--embedding-api-key "$EMBEDDING_API_KEY")
  [[ -z "$SEARCH_TOP_K" ]] || command+=(--top-k "$SEARCH_TOP_K")
  [[ -z "$INDEX_BATCH_SIZE" ]] || command+=(--index-batch-size "$INDEX_BATCH_SIZE")
  [[ -z "$QUERY_PREFIX" ]] || command+=(--query-prefix "$QUERY_PREFIX")
  [[ -z "$DOCUMENT_PREFIX" ]] || command+=(--document-prefix "$DOCUMENT_PREFIX")
  [[ -z "$RETRIEVAL_ALPHA" ]] || command+=(--alpha "$RETRIEVAL_ALPHA")
  [[ -z "$NSAMPLES" ]] || command+=(--dataset-limit "$NSAMPLES")
  EVAL_LAST_CSV=""
  run_cmd "$label" "$logfile" "${command[@]}" || return 1
  EVAL_LAST_CSV="$(latest_csv "$output_dir")"
  [[ -n "$EVAL_LAST_CSV" ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --dataset-ids|--datasets) DATASET_IDS_RAW="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --grad-acc-steps) GRAD_ACC_STEPS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --lr-encoder) LR_ENCODER="$2"; shift 2 ;;
    --lr-heads) LR_HEADS="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --max-query-seq-len) MAX_QUERY_SEQ_LEN="$2"; shift 2 ;;
    --experiment-role) EXPERIMENT_ROLE="$2"; shift 2 ;;
    --nsamples) NSAMPLES="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --table-results) TABLE_RESULTS_PATH="$2"; shift 2 ;;
    --eval-only) RUN_MODE="eval-only"; shift ;;
    --no-baseline) RUN_BASELINE=0; shift ;;
    --wandb-mode) WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --weaviate-url) WEAVIATE_URL="$2"; shift 2 ;;
    --weaviate-grpc-port) WEAVIATE_GRPC_PORT="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --search-mode) SEARCH_MODE="$2"; shift 2 ;;
    --embedding-backend) EMBEDDING_BACKEND="$2"; shift 2 ;;
    --embedding-base-url) EMBEDDING_BASE_URL="$2"; shift 2 ;;
    --embedding-api-key) EMBEDDING_API_KEY="$2"; shift 2 ;;
    --top-k) SEARCH_TOP_K="$2"; shift 2 ;;
    --index-batch-size) INDEX_BATCH_SIZE="$2"; shift 2 ;;
    --query-prefix) QUERY_PREFIX="$2"; shift 2 ;;
    --document-prefix) DOCUMENT_PREFIX="$2"; shift 2 ;;
    --alpha) RETRIEVAL_ALPHA="$2"; shift 2 ;;
    --auto-e5-prefixes) AUTO_E5_PREFIXES=1; shift ;;
    --alpha-gate-layers) EXPLICIT_OVERRIDES+=(--alpha-gate.head.layers "$2"); shift 2 ;;
    --alpha-gate-hidden-dim) EXPLICIT_OVERRIDES+=(--alpha-gate.head.hidden-dim "$2"); shift 2 ;;
    --alpha-gate-dropout) EXPLICIT_OVERRIDES+=(--alpha-gate.head.dropout "$2"); shift 2 ;;
    --memory-bank-size) EXPLICIT_OVERRIDES+=(--memory-bank.size "$2"); shift 2 ;;
    --memory-bank-warmup-steps) EXPLICIT_OVERRIDES+=(--memory-bank.warmup-steps "$2"); shift 2 ;;
    --memory-bank-mining) EXPLICIT_OVERRIDES+=(--memory-bank.mining "$2"); shift 2 ;;
    --memory-bank-hard-negatives) EXPLICIT_OVERRIDES+=(--memory-bank.hard-negatives "$2"); shift 2 ;;
    --memory-bank-random-negatives) EXPLICIT_OVERRIDES+=(--memory-bank.random-negatives "$2"); shift 2 ;;
    --memory-bank-hard-warmup-steps) EXPLICIT_OVERRIDES+=(--memory-bank.hard-warmup-steps "$2"); shift 2 ;;
    --memory-bank-hard-ramp-steps) EXPLICIT_OVERRIDES+=(--memory-bank.hard-ramp-steps "$2"); shift 2 ;;
    --memory-bank-collision-threshold) EXPLICIT_OVERRIDES+=(--memory-bank.adaptive.collision-threshold "$2"); shift 2 ;;
    --memory-bank-collision-beta) EXPLICIT_OVERRIDES+=(--memory-bank.adaptive.collision-beta "$2"); shift 2 ;;
    --memory-bank-margin-mode) EXPLICIT_OVERRIDES+=(--memory-bank.margin.mode "$2"); shift 2 ;;
    --memory-bank-margin-base) EXPLICIT_OVERRIDES+=(--memory-bank.margin.base "$2"); shift 2 ;;
    --memory-bank-margin-scale) EXPLICIT_OVERRIDES+=(--memory-bank.margin.scale "$2"); shift 2 ;;
    --memory-bank-margin-min) EXPLICIT_OVERRIDES+=(--memory-bank.margin.minimum "$2"); shift 2 ;;
    --memory-bank-margin-max) EXPLICIT_OVERRIDES+=(--memory-bank.margin.maximum "$2"); shift 2 ;;
    --memory-bank-admission-beta) EXPLICIT_OVERRIDES+=(--memory-bank.margin.admission-beta "$2"); shift 2 ;;
    --memory-bank-margin-reg-weight) EXPLICIT_OVERRIDES+=(--memory-bank.margin.regularization-weight "$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATASET_IDS_RAW" ]] || { echo "--dataset-ids is required" >&2; exit 2; }
case "$METHOD" in vanilla|atom_gate|atomic) ;; *) echo "invalid method: $METHOD" >&2; exit 2 ;; esac
case "$EXPERIMENT_ROLE" in canonical|ablation) ;; *) echo "invalid experiment role: $EXPERIMENT_ROLE" >&2; exit 2 ;; esac
case "$WANDB_MODE_VALUE" in disabled|offline|online) ;; *) echo "invalid wandb mode: $WANDB_MODE_VALUE" >&2; exit 2 ;; esac
case "$SEARCH_MODE" in ""|keyword|vector|hybrid) ;; *) echo "invalid search mode: $SEARCH_MODE; expected keyword, vector, or hybrid" >&2; exit 2 ;; esac
case "$EMBEDDING_BACKEND" in ""|local|openai-compatible) ;; *) echo "invalid embedding backend: $EMBEDDING_BACKEND" >&2; exit 2 ;; esac

if [[ "$AUTO_E5_PREFIXES" == "1" ]] && should_use_e5_prefixes; then
  [[ -n "$QUERY_PREFIX" ]] || QUERY_PREFIX="query: "
  [[ -n "$DOCUMENT_PREFIX" ]] || DOCUMENT_PREFIX="passage: "
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$(slugify "$METHOD")_$(slugify "$MODEL_NAME_OR_PATH")"
[[ "$RUN_MODE" == "full" ]] || RUN_STAMP="$(date +%Y%m%d_%H%M%S)_eval_only_$(slugify "$MODEL_NAME_OR_PATH")"
RUN_ROOT="$OUTPUT_ROOT/$RUN_STAMP"
mkdir -p "$RUN_ROOT"
TABLE_RESULTS_PATH="${TABLE_RESULTS_PATH:-$RUN_ROOT/TABLE_RESULTS.md}"
SUMMARY_LOG="$RUN_ROOT/summary.log"
: > "$SUMMARY_LOG"

cat > "$TABLE_RESULTS_PATH" <<EOF
# Table Results

- Method: $METHOD
- Model: $MODEL_NAME_OR_PATH
- Run root: $RUN_ROOT

| Dataset | Method | Baseline Status | Base HR@1 | Base HR@10 | Tuned Status | Tuned HR@1 | Tuned HR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
EOF

WEAVIATE_READY=1
check_weaviate || WEAVIATE_READY=0
[[ "$WEAVIATE_READY" == "1" ]] || log "WARNING Weaviate unavailable; evaluation stages will be skipped"

IFS=',' read -r -a DATASET_IDS <<< "$DATASET_IDS_RAW"
for raw_id in "${DATASET_IDS[@]}"; do
  dataset_id="$(resolve_dataset_id "$(trim "$raw_id")")"
  [[ -n "$dataset_id" ]] || continue
  eval_config_id="$(resolve_eval_dataset_id "$dataset_id")"
  dataset_dir="$RUN_ROOT/$(slugify "$dataset_id")"
  mkdir -p "$dataset_dir"
  baseline_status="SKIPPED"
  tuned_status="SKIPPED"
  baseline_csv=""
  tuned_csv=""

  if [[ "$RUN_BASELINE" == "1" && "$WEAVIATE_READY" == "1" ]]; then
    baseline_collection="$(weaviate_class_name "justatom_${eval_config_id}_baseline_${RUN_STAMP}")"
    if evaluate_model "$dataset_id baseline eval" "$dataset_dir/baseline_eval.log" "$MODEL_NAME_OR_PATH" "$baseline_collection" "$dataset_dir/baseline_eval" "$eval_config_id"; then
      baseline_csv="$EVAL_LAST_CSV"
      baseline_status="OK"
    else
      baseline_status="FAILED"
      baseline_csv=""
    fi
  fi

  if [[ "$RUN_MODE" == "full" ]]; then
    train_args=(
      "$PYTHON_BIN" -m justatom.api.train
      --config configs/train.yaml
      --method "$METHOD"
      --dataset.id "$dataset_id"
      --model.name-or-path "$MODEL_NAME_OR_PATH"
      --model.max-seq-len "$MAX_SEQ_LEN"
      --optimization.batch-size "$BATCH_SIZE"
      --optimization.grad-acc-steps "$GRAD_ACC_STEPS"
      --optimization.epochs "$EPOCHS"
      --optimization.lr-encoder "$LR_ENCODER"
      --optimization.lr-heads "$LR_HEADS"
      --optimization.weight-decay "$WEIGHT_DECAY"
      --optimization.num-samples "${NSAMPLES:--1}"
      --objective.temperature "$TEMPERATURE"
      --experiment.role "$EXPERIMENT_ROLE"
      --artifacts.save-dir "$dataset_dir/tuned"
      --artifacts.collection-name "$(weaviate_class_name "justatom_${eval_config_id}_tuned_${RUN_STAMP}")"
      --telemetry.metrics-path "$dataset_dir/tuned_metrics.csv"
    )
    [[ -z "$MAX_QUERY_SEQ_LEN" ]] || train_args+=(--model.max-query-seq-len "$MAX_QUERY_SEQ_LEN")
    [[ -z "$QUERY_PREFIX" ]] || train_args+=(--model.query-prefix "$QUERY_PREFIX")
    [[ -z "$DOCUMENT_PREFIX" ]] || train_args+=(--model.content-prefix "$DOCUMENT_PREFIX")
    [[ -z "$NSAMPLES" ]] || train_args+=(--dataset.limit "$NSAMPLES")
    train_args+=("${EXPLICIT_OVERRIDES[@]}")

    train_env=(env)
    if [[ "$WANDB_MODE_VALUE" == "disabled" ]]; then
      train_args+=(--telemetry.backend csv)
    else
      train_env+=(WANDB_MODE="$WANDB_MODE_VALUE")
      train_args+=(--telemetry.backend wandb --telemetry.wandb-project "$WANDB_PROJECT" --telemetry.run-name "${dataset_id}-${METHOD}")
    fi

    if run_cmd "$dataset_id tune" "$dataset_dir/train.log" "${train_env[@]}" "${train_args[@]}"; then
      if [[ "$WEAVIATE_READY" == "1" ]]; then
        tuned_collection="$(weaviate_class_name "justatom_${eval_config_id}_tuned_${RUN_STAMP}")"
        if evaluate_model "$dataset_id tuned eval" "$dataset_dir/tuned_eval.log" "$dataset_dir/tuned/encoder" "$tuned_collection" "$dataset_dir/tuned_eval" "$eval_config_id"; then
          tuned_csv="$EVAL_LAST_CSV"
          tuned_status="OK"
        else
          tuned_status="FAILED"
          tuned_csv=""
        fi
      else
        tuned_status="SKIPPED_NO_WEAVIATE"
      fi
    else
      tuned_status="FAILED"
    fi
  elif [[ "$WEAVIATE_READY" == "1" ]]; then
    tuned_collection="$(weaviate_class_name "justatom_${eval_config_id}_eval_${RUN_STAMP}")"
    if evaluate_model "$dataset_id eval" "$dataset_dir/eval.log" "$MODEL_NAME_OR_PATH" "$tuned_collection" "$dataset_dir/eval" "$eval_config_id"; then
      tuned_csv="$EVAL_LAST_CSV"
      tuned_status="OK"
    else
      tuned_status="FAILED"
      tuned_csv=""
    fi
  fi

  printf '| `%s` | `%s` | `%s` | %s | %s | `%s` | %s | %s |\n' \
    "$dataset_id" "$METHOD" "$baseline_status" \
    "$(extract_metric "$baseline_csv" "HitRate@1")" "$(extract_metric "$baseline_csv" "HitRate@10")" \
    "$tuned_status" "$(extract_metric "$tuned_csv" "HitRate@1")" "$(extract_metric "$tuned_csv" "HitRate@10")" \
    >> "$TABLE_RESULTS_PATH"
done

log "PIPELINE FINISHED"
log "Results table: $TABLE_RESULTS_PATH"
