#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-intfloat/multilingual-e5-small}"
RECIPE="${RECIPE:-}"
DATASET_IDS_RAW="${DATASET_IDS:-}"
RUN_MODE="${RUN_MODE:-full}"
WANDB_MODE_VALUE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-justatom-pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/pipeline_runs}"
DEFAULT_TABLE_RESULTS_PATH="$REPO_ROOT/TABLE_RESULTS.md"
TABLE_RESULTS_PATH="${TABLE_RESULTS_PATH:-$DEFAULT_TABLE_RESULTS_PATH}"
WEAVIATE_HOST_VALUE="${WEAVIATE_HOST:-localhost}"
WEAVIATE_PORT_VALUE="${WEAVIATE_PORT:-2211}"
COLLECTION_PREFIX="${COLLECTION_PREFIX:-Pipeline}"
RUN_BASELINE=1
BASELINE_CACHE_DIR="${BASELINE_CACHE_DIR:-$REPO_ROOT/.tmp_runs/baseline_cache}"
BASELINE_REFRESH="${BASELINE_REFRESH:-0}"
NSAMPLES="${NSAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
EPOCHS="${EPOCHS:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-6}"
LR_ENCODER="${LR_ENCODER:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-0}"
MEMORY_BANK_WARMUP_STEPS="${MEMORY_BANK_WARMUP_STEPS:-0}"
MEMORY_BANK_MINING_MODE="${MEMORY_BANK_MINING_MODE:-all}"
MEMORY_BANK_HARD_NEGATIVES="${MEMORY_BANK_HARD_NEGATIVES:-0}"
MEMORY_BANK_RANDOM_NEGATIVES="${MEMORY_BANK_RANDOM_NEGATIVES:-0}"
MEMORY_BANK_HARD_WARMUP_STEPS="${MEMORY_BANK_HARD_WARMUP_STEPS:-0}"
MEMORY_BANK_HARD_RAMP_STEPS="${MEMORY_BANK_HARD_RAMP_STEPS:-1}"
MEMORY_BANK_TOO_HARD_MARGIN="${MEMORY_BANK_TOO_HARD_MARGIN:-}"
ADD_ALPHA_GATE="${ADD_ALPHA_GATE:-0}"
ALPHA_GATE_LAYERS="${ALPHA_GATE_LAYERS:-}"
ALPHA_GATE_HIDDEN_DIM="${ALPHA_GATE_HIDDEN_DIM:-}"
ALPHA_GATE_DROPOUT="${ALPHA_GATE_DROPOUT:-}"
ALPHA_GATE_INPUT="${ALPHA_GATE_INPUT:-}"
LOSS_NAME="${LOSS_NAME:-contrastive}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-}"
# Capture which recipe-controlled knobs the user explicitly passed via env
# BEFORE we apply any defaults. Recipe blocks (e.g. `atom_gate`) below
# only set their canonical values when the user did NOT override.
__USER_SET_TEMPERATURE=${TEMPERATURE+1}
__USER_SET_ALPHA_MIX_WEIGHT=${ALPHA_MIX_WEIGHT+1}
__USER_SET_CONTRASTIVE_LEARNABLE_TAU=${CONTRASTIVE_LEARNABLE_TAU+1}
__USER_SET_CONTRASTIVE_DECOUPLED=${CONTRASTIVE_DECOUPLED+1}
__USER_SET_CONTRASTIVE_SIMCSE_WEIGHT=${CONTRASTIVE_SIMCSE_WEIGHT+1}
__USER_SET_CONTRASTIVE_LOSS_ALPHA_GATE=${CONTRASTIVE_LOSS_ALPHA_GATE+1}
__USER_SET_CONTRASTIVE_LOSS_ALPHA_GATE_MODE=${CONTRASTIVE_LOSS_ALPHA_GATE_MODE+1}
__USER_SET_LOSS_NAME=${LOSS_NAME+1}
__USER_SET_OPTIMIZER_NAME=${OPTIMIZER_NAME+1}
__USER_SET_INCLUDE_ALPHA=${INCLUDE_ALPHA+1}
__USER_SET_ALPHA_MODE=${ALPHA_MODE+1}
TEMPERATURE="${TEMPERATURE:-0.03}"
INCLUDE_ALPHA="${INCLUDE_ALPHA:-0}"
ALPHA_MODE="${ALPHA_MODE:-}"
ALPHA_MIX_WEIGHT="${ALPHA_MIX_WEIGHT:-0.4}"
# --- Atom Gate recipe knobs ---
# These ship with safe defaults that match a vanilla contrastive run; the
# `--recipe atom_gate` preset below flips them to the canonical JustAtom values.
# Everything in this block is a *training-only* stabilizer: at eval time only
# the encoder weights are loaded by justatom.api.eval, the gamma-mixer and the
# learnable temperature are discarded.
CONTRASTIVE_LEARNABLE_TAU="${CONTRASTIVE_LEARNABLE_TAU:-}"
CONTRASTIVE_DECOUPLED="${CONTRASTIVE_DECOUPLED:-}"
CONTRASTIVE_SIMCSE_WEIGHT="${CONTRASTIVE_SIMCSE_WEIGHT:-}"
CONTRASTIVE_LOSS_ALPHA_GATE="${CONTRASTIVE_LOSS_ALPHA_GATE:-}"
CONTRASTIVE_LOSS_ALPHA_GATE_MODE="${CONTRASTIVE_LOSS_ALPHA_GATE_MODE:-}"
SEARCH_PIPELINE="${SEARCH_PIPELINE:-}"
SEARCH_TOP_K="${TOP_K:-}"
INDEX_BATCH_SIZE_OVERRIDE="${INDEX_BATCH_SIZE:-}"
QUERY_PREFIX="${QUERY_PREFIX:-}"
CONTENT_PREFIX="${CONTENT_PREFIX:-}"
AUTO_E5_PREFIXES="${AUTO_E5_PREFIXES:-0}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
SUMMARY_LOG=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline.sh --dataset-ids boolq-ru,electrical-engineering-ru,meme-russian-ir [options]

Required:
  --dataset-ids IDS          Comma-separated dataset preset ids from configs/dataset
                             Alias: --datasets

Main options:
  --eval-only              Run only evaluation on the provided model and datasets
  --model MODEL              Base model or checkpoint to tune/evaluate
  --loss NAME                contrastive | soft-contrastive | focal-contrastive
  --temperature VALUE        Main temperature for the selected loss
                             contrastive       -> contrastive_temperature
                             soft-contrastive  -> soft_contrastive_temperature
  --include-alpha            Enable alpha branch with train-only alpha mixing
  --mix-weight VALUE         Alpha mix weight when --include-alpha is enabled
  --optimizer NAME           adafactor | adamw
  --tune-method METHOD       Optional compatibility preset:
                             legacy/classic -> contrastive + no alpha + adafactor + t=0.1
                             new/modern     -> contrastive + alpha + adamw + t=0.03 + mix=0.4
                             atom_gate      -> JustAtom recipe: DCL + learnable tau +
                                               SimCSE 0.1 + alpha-gate(augment) + adamw
                                               (see BEST_TRAINING_RECIPE.md)
                             Alias: --recipe
  --wandb-mode MODE          offline | online | disabled
  --wandb-project NAME       W&B project when wandb is enabled
  --weaviate-host HOST       Weaviate host for justatom.api.eval, default: localhost
  --weaviate-port PORT       Weaviate port for justatom.api.eval, default: 2211
  --search-pipeline NAME     embedding | hybrid | keywords | atomicai for eval stages
  --top-k N                  Retrieval top-k for eval stages
  --output-root DIR          Run artifacts root directory
  --table-results PATH       Markdown summary path, default TABLE_RESULTS.md inside RUN_ROOT
  --nsamples N               Limit both train and eval to the same first N examples
  --no-baseline              Skip baseline eval

Advanced options:
  --alpha-mode MODE          off | train-only | joint
  --alpha-mix-weight VALUE   Same as --mix-weight
  --add-alpha-gate           Enable structured alpha(q) gate config
  --alpha-gate-layers N      Hidden layers in alpha(q), default from config
  --alpha-gate-hidden-dim N  Hidden dim in alpha(q), or auto
  --alpha-gate-dropout P     Dropout inside alpha(q)
  --alpha-gate-input MODE    query | query_doc
  --batch-size N             Train batch size
  --eval-batch-size N        Eval batch size
  --max-seq-len N            Max sequence length for eval
  --epochs N                 Number of epochs
  --grad-acc-steps N         Gradient accumulation steps
  --lr-encoder VALUE         Encoder learning rate
  --weight-decay VALUE       Weight decay for AdamW/custom
  --memory-bank-size N       FIFO document embedding bank for extra InfoNCE negatives
  --memory-bank-warmup-steps N
                             Training steps before bank negatives are used
  --memory-bank-mining MODE  all | random | hard | mixed
  --memory-bank-hard-negatives N
                             Max hard bank negatives per query after ramp
  --memory-bank-random-negatives N
                             Random safe bank negatives per query
  --memory-bank-hard-warmup-steps N
                             Steps before hard mining starts
  --memory-bank-hard-ramp-steps N
                             Steps to ramp hard negatives to max
  --memory-bank-too-hard-margin VALUE
                             Drop bank negatives with sim > pos_sim - VALUE
  --index-batch-size N       Index batch size for eval stages
  --query-prefix TEXT        Query prefix override for eval stages
  --content-prefix TEXT      Content prefix override for eval stages
  --auto-e5-prefixes         Auto-set e5-compatible query/content prefixes for eval stages

Examples:
  bash scripts/run_pipeline.sh \
    --eval-only \
    --dataset-ids boolq-ru,electrical-engineering-ru,justatom,meme-russian-ir \
    --model intfloat/multilingual-e5-small \
    --wandb-mode offline

  bash scripts/run_pipeline.sh \
    --dataset-ids boolq-ru,electrical-engineering-ru,meme-russian-ir \
    --model intfloat/multilingual-e5-small \
    --loss contrastive \
    --temperature 0.03 \
    --include-alpha \
    --mix-weight 0.4 \
    --wandb-mode offline

  bash scripts/run_pipeline.sh \
    --dataset-ids justatom \
    --model intfloat/multilingual-e5-small \
    --loss contrastive \
    --temperature 0.03 \
    --optimizer adamw \
    --wandb-mode offline

  bash scripts/run_pipeline.sh \
    --dataset-ids justatom \
    --model intfloat/multilingual-e5-small \
    --loss soft-contrastive \
    --temperature 10.0 \
    --include-alpha \
    --mix-weight 0.4 \
    --optimizer adamw \
    --wandb-mode offline
EOF
}

trim() {
  local value="$1"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"
  printf '%s' "$value"
}

slugify() {
  printf '%s' "$1" | tr '/: ?&=,' '_' | tr -s '_'
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
    out="PipelineRun"
  fi
  if [[ ! "$out" =~ ^[A-Z] ]]; then
    out="Pipeline${out}"
  fi
  printf '%s' "$out"
}

canonicalize_recipe() {
  case "$1" in
    classic|encoder-only|legacy)
      printf 'classic'
      ;;
    modern|alpha-gamma|new|recommended)
      printf 'modern'
      ;;
    atom_gate|atom-gate|atom|justatom_gate|justatom-gate)
      printf 'atom_gate'
      ;;
    custom)
      printf 'custom'
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

loss_display_name() {
  case "$1" in
    contrastive)
      printf 'contrastive'
      ;;
    soft-contrastive)
      printf 'soft-contrastive'
      ;;
    focal-contrastive)
      printf 'focal-contrastive'
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

should_use_e5_prefixes() {
  local model_lower
  model_lower="$(printf '%s' "$MODEL_NAME_OR_PATH" | tr '[:upper:]' '[:lower:]')"
  [[ "$model_lower" == *"/e5"* || "$model_lower" == *"-e5-"* || "$model_lower" == e5* ]]
}

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message" | tee -a "$SUMMARY_LOG"
}

run_cmd() {
  local label="$1"
  local logfile="$2"
  shift 2
  log "START $label"
  set +e
  if [[ "${PIPELINE_TEE_OUTPUT:-1}" == "1" ]]; then
    "$@" 2>&1 | tee "$logfile"
    local code=${PIPESTATUS[0]}
  else
    "$@" > "$logfile" 2>&1
    local code=$?
  fi
  set -e
  log "END $label exit=$code"
  return $code
}

check_weaviate() {
  local host="$1"
  local port="$2"
  local ready_url="http://${host}:${port}/v1/.well-known/ready"
  local meta_url="http://${host}:${port}/v1/meta"

  if command -v curl >/dev/null 2>&1; then
    curl --silent --fail --max-time 3 "$ready_url" >/dev/null 2>&1 && return 0
    curl --silent --fail --max-time 3 "$meta_url" >/dev/null 2>&1 && return 0
    return 1
  fi

  return 0
}

resolve_dataset_id() {
  case "$1" in
    boolq)
      printf 'boolq-ru'
      ;;
    electrical|electrical-engineering|electrical-engineering-ru)
      printf 'electrical-engineering-ru'
      ;;
    miracl|miracl-ru)
      printf 'miracl-ru'
      ;;
    meme|meme-russian-ir)
      printf 'meme-russian-ir'
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

extract_metric() {
  local csv_path="$1"
  local metric_name="$2"
  if [[ ! -f "$csv_path" ]]; then
  printf 'NA'
  return
  fi
  awk -F',' -v metric="$metric_name" '
  $1 == metric { print $2; found = 1; exit }
  END { if (!found) print "NA" }
  ' "$csv_path"
}

latest_csv() {
  local dir="$1"
  local latest=""
  if [[ -d "$dir" ]]; then
  latest="$(ls -1t "$dir"/*.csv 2>/dev/null | head -n 1 || true)"
  fi
  printf '%s' "$latest"
}

EVAL_LAST_CSV=""
evaluate_model() {
  local label="$1"
  local logfile="$2"
  local model_ref="$3"
  local collection_name="$4"
  local output_dir="$5"
  local dataset_id="$6"

  mkdir -p "$output_dir"

  local cmd=(
    "$PYTHON_BIN" -m justatom.api.eval
    --config configs/evaluate.yaml
    --model-name-or-path "$model_ref"
    --collection-name "$collection_name"
    --save-results-to-dir "$output_dir"
    --flush-collection
    --search-batch-size "$EVAL_BATCH_SIZE"
    --weaviate-host "$WEAVIATE_HOST_VALUE"
    --weaviate-port "$WEAVIATE_PORT_VALUE"
    --dataset.id "$dataset_id"
  )

  if [[ -n "$SEARCH_PIPELINE" ]]; then
    cmd+=(--search-pipeline "$SEARCH_PIPELINE")
  fi
  if [[ -n "$SEARCH_TOP_K" ]]; then
    cmd+=(--top-k "$SEARCH_TOP_K")
  fi
  if [[ -n "$INDEX_BATCH_SIZE_OVERRIDE" ]]; then
    cmd+=(--index-batch-size "$INDEX_BATCH_SIZE_OVERRIDE")
  fi
  if [[ -n "$QUERY_PREFIX" ]]; then
    cmd+=(--query-prefix "$QUERY_PREFIX")
  fi
  if [[ -n "$CONTENT_PREFIX" ]]; then
    cmd+=(--content-prefix "$CONTENT_PREFIX")
  fi
  if [[ ${#eval_limit_args[@]} -gt 0 ]]; then
    cmd+=("${eval_limit_args[@]}")
  fi
  if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $EXTRA_EVAL_ARGS )
    cmd+=("${extra_args[@]}")
  fi

  EVAL_LAST_CSV=""
  if run_cmd "$label" "$logfile" "${cmd[@]}"; then
    EVAL_LAST_CSV="$(latest_csv "$output_dir")"
    [[ -n "$EVAL_LAST_CSV" ]]
    return
  fi
  return 1
}

append_table_row() {
  local dataset_id="$1"
  local method_label="$2"
  local baseline_status="$3"
  local tuned_status="$4"
  local baseline_csv="$5"
  local tuned_csv="$6"

  local b_hr1 b_hr5 b_hr10 b_mrr10 b_ndcg10
  local t_hr1 t_hr5 t_hr10 t_mrr10 t_ndcg10

  b_hr1="$(extract_metric "$baseline_csv" "HitRate@1")"
  b_hr5="$(extract_metric "$baseline_csv" "HitRate@5")"
  b_hr10="$(extract_metric "$baseline_csv" "HitRate@10")"
  b_mrr10="$(extract_metric "$baseline_csv" "mrr@10")"
  b_ndcg10="$(extract_metric "$baseline_csv" "ndcg@10")"

  t_hr1="$(extract_metric "$tuned_csv" "HitRate@1")"
  t_hr5="$(extract_metric "$tuned_csv" "HitRate@5")"
  t_hr10="$(extract_metric "$tuned_csv" "HitRate@10")"
  t_mrr10="$(extract_metric "$tuned_csv" "mrr@10")"
  t_ndcg10="$(extract_metric "$tuned_csv" "ndcg@10")"

  printf '| `%s` | `%s` | `%s` | %s | %s | %s | %s | %s | `%s` | %s | %s | %s | %s | %s |\n' \
    "$dataset_id" "$method_label" "$baseline_status" \
    "$b_hr1" "$b_hr5" "$b_hr10" "$b_mrr10" "$b_ndcg10" \
    "$tuned_status" "$t_hr1" "$t_hr5" "$t_hr10" "$t_mrr10" "$t_ndcg10" \
    >> "$TABLE_RESULTS_PATH"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loss)
      LOSS_NAME="$2"
      shift 2
      ;;
    --eval-only)
      RUN_MODE="eval-only"
      shift
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --include-alpha)
      INCLUDE_ALPHA=1
      shift
      ;;
    --mix-weight)
      ALPHA_MIX_WEIGHT="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER_NAME="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --recipe|--tune-method)
      RECIPE="$2"
      shift 2
      ;;
    --dataset-ids|--datasets)
      DATASET_IDS_RAW="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE_VALUE="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --weaviate-host)
      WEAVIATE_HOST_VALUE="$2"
      shift 2
      ;;
    --weaviate-port)
      WEAVIATE_PORT_VALUE="$2"
      shift 2
      ;;
    --search-pipeline)
      SEARCH_PIPELINE="$2"
      shift 2
      ;;
    --top-k)
      SEARCH_TOP_K="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --nsamples)
      NSAMPLES="$2"
      shift 2
      ;;
    --table-results)
      TABLE_RESULTS_PATH="$2"
      shift 2
      ;;
    --alpha-mode)
      ALPHA_MODE="$2"
      shift 2
      ;;
    --alpha-mix-weight)
      ALPHA_MIX_WEIGHT="$2"
      shift 2
      ;;
    --add-alpha-gate)
      ADD_ALPHA_GATE=1
      shift
      ;;
    --alpha-gate-layers)
      ALPHA_GATE_LAYERS="$2"
      shift 2
      ;;
    --alpha-gate-hidden-dim)
      ALPHA_GATE_HIDDEN_DIM="$2"
      shift 2
      ;;
    --alpha-gate-dropout)
      ALPHA_GATE_DROPOUT="$2"
      shift 2
      ;;
    --alpha-gate-input)
      ALPHA_GATE_INPUT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --eval-batch-size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --grad-acc-steps)
      GRAD_ACC_STEPS="$2"
      shift 2
      ;;
    --lr-encoder)
      LR_ENCODER="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --memory-bank-size)
      MEMORY_BANK_SIZE="$2"
      shift 2
      ;;
    --memory-bank-warmup-steps)
      MEMORY_BANK_WARMUP_STEPS="$2"
      shift 2
      ;;
    --memory-bank-mining)
      MEMORY_BANK_MINING_MODE="$2"
      shift 2
      ;;
    --memory-bank-hard-negatives)
      MEMORY_BANK_HARD_NEGATIVES="$2"
      shift 2
      ;;
    --memory-bank-random-negatives)
      MEMORY_BANK_RANDOM_NEGATIVES="$2"
      shift 2
      ;;
    --memory-bank-hard-warmup-steps)
      MEMORY_BANK_HARD_WARMUP_STEPS="$2"
      shift 2
      ;;
    --memory-bank-hard-ramp-steps)
      MEMORY_BANK_HARD_RAMP_STEPS="$2"
      shift 2
      ;;
    --memory-bank-too-hard-margin)
      MEMORY_BANK_TOO_HARD_MARGIN="$2"
      shift 2
      ;;
    --index-batch-size)
      INDEX_BATCH_SIZE_OVERRIDE="$2"
      shift 2
      ;;
    --query-prefix)
      QUERY_PREFIX="$2"
      shift 2
      ;;
    --content-prefix)
      CONTENT_PREFIX="$2"
      shift 2
      ;;
    --auto-e5-prefixes)
      AUTO_E5_PREFIXES=1
      shift
      ;;
    --no-baseline)
      RUN_BASELINE=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_IDS_RAW" ]]; then
  echo "--dataset-ids is required" >&2
  usage >&2
  exit 1
fi

case "$RUN_MODE" in
  full|eval-only)
    ;;
  *)
    echo "Unsupported run mode: $RUN_MODE" >&2
    exit 1
    ;;
esac

if [[ "$AUTO_E5_PREFIXES" == "1" ]] && [[ -z "$QUERY_PREFIX" ]] && should_use_e5_prefixes; then
  QUERY_PREFIX="query: "
fi

if [[ "$AUTO_E5_PREFIXES" == "1" ]] && [[ -z "$CONTENT_PREFIX" ]] && should_use_e5_prefixes; then
  CONTENT_PREFIX="passage: "
fi

if [[ -n "$RECIPE" ]]; then
  RECIPE="$(canonicalize_recipe "$RECIPE")"
  case "$RECIPE" in
    classic)
      LOSS_NAME="contrastive"
      OPTIMIZER_NAME="adafactor"
      TEMPERATURE="0.1"
      INCLUDE_ALPHA=0
      ALPHA_MODE="off"
      ALPHA_MIX_WEIGHT="0.0"
      ;;
    modern)
      LOSS_NAME="contrastive"
      OPTIMIZER_NAME="adamw"
      TEMPERATURE="0.03"
      INCLUDE_ALPHA=1
      ALPHA_MODE="train-only"
      ALPHA_MIX_WEIGHT="0.4"
      ;;
    atom_gate)
      # Atom Gate recipe: JustAtom's alpha-gated contrastive tune.
      # Main = decoupled InfoNCE with learnable tau, aux = SimCSE dropout-view
      # alignment whose strength is gated per-query by (1 - alpha(q)) from the
      # gamma-mixer. None of the gating components survive at eval time -- the
      # eval path (justatom.api.eval) only loads the encoder weights.
      # NOTE: each knob below is applied ONLY when the user did not pass it
      # explicitly via env (see __USER_SET_* sentinels at top of script).
      [[ -z "$__USER_SET_LOSS_NAME"                       ]] && LOSS_NAME="contrastive"
      [[ -z "$__USER_SET_OPTIMIZER_NAME"                  ]] && OPTIMIZER_NAME="adamw"
      [[ -z "$__USER_SET_TEMPERATURE"                     ]] && TEMPERATURE="0.05"
      [[ -z "$__USER_SET_INCLUDE_ALPHA"                   ]] && INCLUDE_ALPHA=1
      [[ -z "$__USER_SET_ALPHA_MODE"                      ]] && ALPHA_MODE="train-only"
      [[ -z "$__USER_SET_ALPHA_MIX_WEIGHT"                ]] && ALPHA_MIX_WEIGHT="0.3"
      ADD_ALPHA_GATE=1
      [[ -z "$__USER_SET_CONTRASTIVE_LEARNABLE_TAU"       ]] && CONTRASTIVE_LEARNABLE_TAU="true"
      [[ -z "$__USER_SET_CONTRASTIVE_DECOUPLED"           ]] && CONTRASTIVE_DECOUPLED="true"
      [[ -z "$__USER_SET_CONTRASTIVE_SIMCSE_WEIGHT"       ]] && CONTRASTIVE_SIMCSE_WEIGHT="0.1"
      [[ -z "$__USER_SET_CONTRASTIVE_LOSS_ALPHA_GATE"     ]] && CONTRASTIVE_LOSS_ALPHA_GATE="true"
      [[ -z "$__USER_SET_CONTRASTIVE_LOSS_ALPHA_GATE_MODE" ]] && CONTRASTIVE_LOSS_ALPHA_GATE_MODE="augment"
      ;;
    custom)
      ;;
    *)
      echo "Unsupported tune method: $RECIPE" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "$OPTIMIZER_NAME" ]]; then
  OPTIMIZER_NAME="adamw"
fi

if [[ -z "$ALPHA_MODE" ]]; then
  if [[ "$INCLUDE_ALPHA" -eq 1 ]]; then
    ALPHA_MODE="train-only"
  else
    ALPHA_MODE="off"
  fi
fi

case "$LOSS_NAME" in
  contrastive|soft-contrastive|focal-contrastive)
    ;;
  *)
    echo "Unsupported --loss: $LOSS_NAME" >&2
    exit 1
    ;;
esac

case "$ALPHA_MODE" in
  off|train-only|joint)
    ;;
  *)
    echo "Unsupported --alpha-mode: $ALPHA_MODE" >&2
    exit 1
    ;;
esac

case "$WANDB_MODE_VALUE" in
  offline|online|disabled)
    ;;
  *)
    echo "Unsupported --wandb-mode: $WANDB_MODE_VALUE" >&2
    exit 1
    ;;
esac

if ! [[ "$WEAVIATE_PORT_VALUE" =~ ^[0-9]+$ ]]; then
  echo "Unsupported --weaviate-port: $WEAVIATE_PORT_VALUE" >&2
  exit 1
fi

if ! [[ "$MEMORY_BANK_SIZE" =~ ^[0-9]+$ ]]; then
  echo "Unsupported --memory-bank-size: $MEMORY_BANK_SIZE" >&2
  exit 1
fi

for numeric_arg in \
  MEMORY_BANK_WARMUP_STEPS \
  MEMORY_BANK_HARD_NEGATIVES \
  MEMORY_BANK_RANDOM_NEGATIVES \
  MEMORY_BANK_HARD_WARMUP_STEPS \
  MEMORY_BANK_HARD_RAMP_STEPS; do
  numeric_value="${!numeric_arg}"
  if ! [[ "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "Unsupported ${numeric_arg}: $numeric_value" >&2
    exit 1
  fi
done

case "$MEMORY_BANK_MINING_MODE" in
  all|random|hard|mixed)
    ;;
  *)
    echo "Unsupported --memory-bank-mining: $MEMORY_BANK_MINING_MODE" >&2
    exit 1
    ;;
esac

if [[ "$RUN_MODE" == "eval-only" ]]; then
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_eval_only_$(slugify "$MODEL_NAME_OR_PATH")"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_$(slugify "$RECIPE")_$(slugify "$MODEL_NAME_OR_PATH")"
fi
RUN_ROOT="$OUTPUT_ROOT/$RUN_STAMP"
mkdir -p "$RUN_ROOT"
if [[ "$TABLE_RESULTS_PATH" == "$DEFAULT_TABLE_RESULTS_PATH" ]]; then
  TABLE_RESULTS_PATH="$RUN_ROOT/TABLE_RESULTS.md"
fi
DATA_CACHE_DIR="$RUN_ROOT/data_cache"
mkdir -p "$DATA_CACHE_DIR"
SUMMARY_LOG="$RUN_ROOT/summary.log"
: > "$SUMMARY_LOG"

LOSS_LABEL="$(loss_display_name "$LOSS_NAME")"
ALPHA_LABEL="disabled"
METHOD_LABEL=""
METHOD_SUMMARY=""
RESOLVED_CONFIG=""
if [[ "$RUN_MODE" == "eval-only" ]]; then
  LOSS_LABEL="n/a"
  ALPHA_LABEL="n/a"
  METHOD_LABEL="eval-only"
  METHOD_SUMMARY="search_pipeline=${SEARCH_PIPELINE:-embedding}, eval_batch_size=$EVAL_BATCH_SIZE"
  RESOLVED_CONFIG="mode=eval-only"
  if [[ -n "$SEARCH_TOP_K" ]]; then
    RESOLVED_CONFIG="$RESOLVED_CONFIG,top_k=$SEARCH_TOP_K"
  fi
  if [[ -n "$NSAMPLES" ]]; then
    RESOLVED_CONFIG="$RESOLVED_CONFIG,nsamples=$NSAMPLES"
  fi
else
  if [[ "$ALPHA_MODE" != "off" ]]; then
    ALPHA_LABEL="enabled(mode=$ALPHA_MODE,mix=$ALPHA_MIX_WEIGHT)"
  fi
  METHOD_LABEL="$LOSS_LABEL + alpha=$ALPHA_LABEL"
  METHOD_SUMMARY="optimizer=$OPTIMIZER_NAME, temperature=$TEMPERATURE, memory_bank=$MEMORY_BANK_SIZE/$MEMORY_BANK_MINING_MODE"
  if [[ -n "$RECIPE" ]]; then
    METHOD_SUMMARY="$METHOD_SUMMARY, preset=$RECIPE"
  fi
  RESOLVED_CONFIG="loss=$LOSS_NAME,opt=$OPTIMIZER_NAME,temp=$TEMPERATURE,alpha=$ALPHA_MODE,mix=$ALPHA_MIX_WEIGHT,memory_bank=$MEMORY_BANK_SIZE,bank_mode=$MEMORY_BANK_MINING_MODE,bank_warmup=$MEMORY_BANK_WARMUP_STEPS,bank_hard=$MEMORY_BANK_HARD_NEGATIVES,bank_random=$MEMORY_BANK_RANDOM_NEGATIVES"
  if [[ -n "$NSAMPLES" ]]; then
    RESOLVED_CONFIG="$RESOLVED_CONFIG,nsamples=$NSAMPLES"
  fi
fi

cat > "$TABLE_RESULTS_PATH" <<EOF
# Table Results

- Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Model: $MODEL_NAME_OR_PATH
- Loss: $LOSS_LABEL
- Temperature: $TEMPERATURE
- Alpha: $ALPHA_LABEL
- Memory bank size: $MEMORY_BANK_SIZE
- Memory bank mining: $MEMORY_BANK_MINING_MODE
- N samples: ${NSAMPLES:-all}
- Method summary: $METHOD_SUMMARY
- Resolved config: $RESOLVED_CONFIG
- Wandb mode: $WANDB_MODE_VALUE
- Run root: $RUN_ROOT

| Dataset | Tuning Method | Baseline Status | Base HR@1 | Base HR@5 | Base HR@10 | Base MRR@10 | Base NDCG@10 | Tuned Status | Tuned HR@1 | Tuned HR@5 | Tuned HR@10 | Tuned MRR@10 | Tuned NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
EOF

log "Run root: $RUN_ROOT"
log "Model: $MODEL_NAME_OR_PATH"
log "Loss: $LOSS_LABEL"
log "Temperature: $TEMPERATURE"
log "Alpha: $ALPHA_LABEL"
log "Memory bank size: $MEMORY_BANK_SIZE"
log "Memory bank mining: mode=$MEMORY_BANK_MINING_MODE warmup=$MEMORY_BANK_WARMUP_STEPS hard=$MEMORY_BANK_HARD_NEGATIVES random=$MEMORY_BANK_RANDOM_NEGATIVES hard_warmup=$MEMORY_BANK_HARD_WARMUP_STEPS hard_ramp=$MEMORY_BANK_HARD_RAMP_STEPS too_hard_margin=${MEMORY_BANK_TOO_HARD_MARGIN:-none}"
log "Run mode: $RUN_MODE"
log "N samples: ${NSAMPLES:-all}"
log "Weaviate: ${WEAVIATE_HOST_VALUE}:${WEAVIATE_PORT_VALUE}"
log "Method summary: $METHOD_SUMMARY"
log "Resolved config: $RESOLVED_CONFIG"
log "Datasets: $DATASET_IDS_RAW"

WEAVIATE_READY=1
if ! check_weaviate "$WEAVIATE_HOST_VALUE" "$WEAVIATE_PORT_VALUE"; then
  WEAVIATE_READY=0
  log "WARNING Weaviate is unavailable at ${WEAVIATE_HOST_VALUE}:${WEAVIATE_PORT_VALUE}; eval stages will be skipped"
fi

IFS=',' read -r -a RAW_DATASET_IDS <<< "$DATASET_IDS_RAW"
for raw_id in "${RAW_DATASET_IDS[@]}"; do
  dataset_id="$(trim "$raw_id")"
  [[ -n "$dataset_id" ]] || continue
  config_id="$(resolve_dataset_id "$dataset_id")"
  dataset_dir="$RUN_ROOT/$(slugify "$config_id")"
  baseline_collection="$(weaviate_class_name "${COLLECTION_PREFIX}_${config_id}_baseline_${RUN_STAMP}")"
  tuned_collection="$(weaviate_class_name "${COLLECTION_PREFIX}_${config_id}_tuned_${RUN_STAMP}")"
  mkdir -p "$dataset_dir"

  eval_limit_args=()
  train_sample_count="1000000000"
  if [[ -n "$NSAMPLES" ]]; then
    eval_limit_args+=(--dataset-limit "$NSAMPLES")
    train_sample_count="$NSAMPLES"
  fi

  if [[ "$RUN_MODE" == "eval-only" ]]; then
    eval_status="FAILED"
    eval_log="$dataset_dir/eval_only.log"
    eval_out_dir="$dataset_dir/eval_only"
    eval_collection="$(weaviate_class_name "${COLLECTION_PREFIX}_${config_id}_${RUN_STAMP}")"
    eval_csv=""

    if [[ "$WEAVIATE_READY" -eq 0 ]]; then
      eval_status="SKIPPED_NO_WEAVIATE"
      log "SKIP $config_id eval-only because Weaviate is unavailable"
    else
      if evaluate_model "$config_id eval-only" "$eval_log" "$MODEL_NAME_OR_PATH" "$eval_collection" "$eval_out_dir" "$config_id"; then
        eval_csv="$EVAL_LAST_CSV"
        cp "$eval_csv" "$RUN_ROOT/$(slugify "$config_id")__$(slugify "$MODEL_NAME_OR_PATH").csv"
        eval_status="OK"
      fi
    fi

    append_table_row "$config_id" "$METHOD_LABEL" "$eval_status" "SKIPPED" "$eval_csv" ""
    continue
  fi

  baseline_status="SKIPPED"
  baseline_log="$dataset_dir/baseline.log"
  baseline_csv=""
  if [[ "$RUN_BASELINE" -eq 1 ]]; then
    if [[ "$WEAVIATE_READY" -eq 0 ]]; then
      baseline_status="SKIPPED_NO_WEAVIATE"
      log "SKIP $config_id baseline because Weaviate is unavailable"
    else
      baseline_eval_dir="$dataset_dir/baseline_eval"
      mkdir -p "$baseline_eval_dir"
      baseline_cache_key="$(slugify "$MODEL_NAME_OR_PATH")__$(slugify "$config_id").csv"
      baseline_cache_path="$BASELINE_CACHE_DIR/$baseline_cache_key"
      if [[ "$BASELINE_REFRESH" -ne 1 && -s "$baseline_cache_path" ]]; then
        baseline_csv="$baseline_eval_dir/$(basename "$baseline_cache_path")"
        cp "$baseline_cache_path" "$baseline_csv"
        baseline_status="CACHED"
        log "CACHE-HIT $config_id baseline reused from $baseline_cache_path"
      else
        if evaluate_model "$config_id baseline" "$baseline_log" "$MODEL_NAME_OR_PATH" "$baseline_collection" "$baseline_eval_dir" "$config_id"; then
          baseline_csv="$EVAL_LAST_CSV"
          baseline_status="OK"
          mkdir -p "$BASELINE_CACHE_DIR"
          cp "$baseline_csv" "$baseline_cache_path"
          log "CACHE-STORE $config_id baseline saved to $baseline_cache_path"
        else
          baseline_status="FAILED"
        fi
      fi
    fi
  fi

  train_log="$dataset_dir/train.log"
  train_env=(env)
  train_args=(
    "$PYTHON_BIN" -m justatom.api.train
    --config configs/train.yaml
    --model.name "$MODEL_NAME_OR_PATH"
    --dataset.id "$config_id"
    --training.loss "$LOSS_NAME"
    --training.freeze_encoder false
    --training.optimizer "$OPTIMIZER_NAME"
    --training.lr_encoder "$LR_ENCODER"
    --training.grad_acc_steps "$GRAD_ACC_STEPS"
    --training.batch_size "$BATCH_SIZE"
    --training.max_seq_len "$MAX_SEQ_LEN"
    --training.num_samples "$train_sample_count"
    --training.n_epochs "$EPOCHS"
    --training.weight_decay "$WEIGHT_DECAY"
    --training.memory_bank_size "$MEMORY_BANK_SIZE"
    --training.memory_bank_warmup_steps "$MEMORY_BANK_WARMUP_STEPS"
    --training.memory_bank_mining_mode "$MEMORY_BANK_MINING_MODE"
    --training.memory_bank_hard_negatives "$MEMORY_BANK_HARD_NEGATIVES"
    --training.memory_bank_random_negatives "$MEMORY_BANK_RANDOM_NEGATIVES"
    --training.memory_bank_hard_warmup_steps "$MEMORY_BANK_HARD_WARMUP_STEPS"
    --training.memory_bank_hard_ramp_steps "$MEMORY_BANK_HARD_RAMP_STEPS"
    --output.save_dir "$dataset_dir/tuned"
    --output.metrics_path "$dataset_dir/tuned_metrics.csv"
  )
  if [[ "$RECIPE" == "atom_gate" ]]; then
    train_args+=(--recipe atom_gate)
    train_args+=(--atom-gate.temperature "$TEMPERATURE")
  fi
  if [[ "$ADD_ALPHA_GATE" == "1" ]]; then
    train_args+=(--add-alpha-gate)
  fi
  if [[ -n "$ALPHA_GATE_LAYERS" ]]; then
    train_args+=(--alpha-gate.alpha-query.layers "$ALPHA_GATE_LAYERS")
  fi
  if [[ -n "$ALPHA_GATE_HIDDEN_DIM" ]]; then
    train_args+=(--alpha-gate.alpha-query.hidden-dim "$ALPHA_GATE_HIDDEN_DIM")
  fi
  if [[ -n "$ALPHA_GATE_DROPOUT" ]]; then
    train_args+=(--alpha-gate.alpha-query.dropout "$ALPHA_GATE_DROPOUT")
  fi
  if [[ -n "$ALPHA_GATE_INPUT" ]]; then
    train_args+=(--alpha-gate.alpha-query.input "$ALPHA_GATE_INPUT")
  fi
  if [[ -n "$MEMORY_BANK_TOO_HARD_MARGIN" ]]; then
    train_args+=(--training.memory_bank_too_hard_margin "$MEMORY_BANK_TOO_HARD_MARGIN")
  fi
  if [[ -n "$NSAMPLES" ]]; then
    train_args+=(--dataset.limit "$NSAMPLES")
  fi

  case "$LOSS_NAME" in
    contrastive)
      train_args+=(--training.contrastive_temperature "$TEMPERATURE")
      ;;
    soft-contrastive)
      train_args+=(--training.soft_contrastive_temperature "$TEMPERATURE")
      ;;
  esac

  # Atom Gate training-time stabilizers. Each one is forwarded only
  # if the recipe / env explicitly set it, so legacy `classic` / `modern`
  # presets keep their previous behavior.
  if [[ -n "$CONTRASTIVE_LEARNABLE_TAU" ]]; then
    train_args+=(--training.contrastive_learnable_temperature "$CONTRASTIVE_LEARNABLE_TAU")
  fi
  if [[ -n "$CONTRASTIVE_DECOUPLED" ]]; then
    train_args+=(--training.contrastive_decoupled "$CONTRASTIVE_DECOUPLED")
  fi
  if [[ -n "$CONTRASTIVE_SIMCSE_WEIGHT" ]]; then
    train_args+=(--training.contrastive_simcse_dropout_weight "$CONTRASTIVE_SIMCSE_WEIGHT")
  fi
  if [[ -n "$CONTRASTIVE_LOSS_ALPHA_GATE" ]]; then
    train_args+=(--training.contrastive_loss_alpha_gate "$CONTRASTIVE_LOSS_ALPHA_GATE")
  fi
  if [[ -n "$CONTRASTIVE_LOSS_ALPHA_GATE_MODE" ]]; then
    train_args+=(--training.contrastive_loss_alpha_gate_mode "$CONTRASTIVE_LOSS_ALPHA_GATE_MODE")
  fi

  checkpoint_rel="BiGamma/epoch1"
  case "$ALPHA_MODE" in
    off)
      checkpoint_rel="Encoder/epoch1"
      train_args+=(
        --training.gamma_joint false
        --training.include_semantic_gamma false
        --training.include_keywords_gamma false
        --training.alpha_train_only false
        --training.alpha_mix_weight 0.0
      )
      ;;
    train-only)
      train_args+=(
        --training.gamma_joint true
        --training.alpha_train_only true
        --training.alpha_mix_weight "$ALPHA_MIX_WEIGHT"
      )
      ;;
    joint)
      train_args+=(
        --training.gamma_joint true
        --training.alpha_train_only false
        --training.alpha_mix_weight "$ALPHA_MIX_WEIGHT"
      )
      ;;
  esac
  if [[ "$ADD_ALPHA_GATE" == "1" ]]; then
    checkpoint_rel="BiGamma/epoch1"
  fi

  if [[ "$WANDB_MODE_VALUE" == "disabled" ]]; then
    train_args+=(--logging.backend csv)
  else
    train_env+=(WANDB_MODE="$WANDB_MODE_VALUE")
    train_args+=(
      --logging.backend wandb
      --logging.wandb_project "$WANDB_PROJECT"
      --logging.wandb_run_name "$(slugify "$config_id")-$(slugify "$RECIPE")"
    )
  fi

  tuned_status="FAILED"
  tuned_eval_log="$dataset_dir/tuned_eval.log"
  tuned_csv=""
  if run_cmd "$config_id tune" "$train_log" "${train_env[@]}" "${train_args[@]}"; then
    if [[ "$WEAVIATE_READY" -eq 0 ]]; then
      tuned_status="SKIPPED_NO_WEAVIATE"
      log "SKIP $config_id tuned eval because Weaviate is unavailable"
    else
      tuned_eval_dir="$dataset_dir/tuned_eval"
      mkdir -p "$tuned_eval_dir"
      if evaluate_model "$config_id tuned eval" "$tuned_eval_log" "$dataset_dir/tuned/$checkpoint_rel" "$tuned_collection" "$tuned_eval_dir" "$config_id"; then
        tuned_csv="$EVAL_LAST_CSV"
        tuned_status="OK"
      fi
    fi
  fi

  append_table_row "$config_id" "$METHOD_LABEL" "$baseline_status" "$tuned_status" "$baseline_csv" "$tuned_csv"
done

log "PIPELINE FINISHED"
log "Results table: $TABLE_RESULTS_PATH"
