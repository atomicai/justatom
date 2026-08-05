#!/usr/bin/env bash
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
  for candidate in "${JUSTATOM_BASH:-}" /opt/homebrew/bin/bash /usr/local/bin/bash; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      exec "$candidate" "$0" "$@"
    fi
  done
  echo "scripts/run_benchmark.sh requires Bash >= 4. Install Bash 4+ or run with /opt/homebrew/bin/bash." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PIPELINE_SHELL="${PIPELINE_SHELL:-${BASH:-bash}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-intfloat/multilingual-e5-small}"
DATASET_IDS_RAW="${DATASET_IDS:-justatom}"
WANDB_MODE_VALUE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-justatom-benchmark}"
WEAVIATE_HOST_VALUE="${WEAVIATE_HOST:-localhost}"
WEAVIATE_PORT_VALUE="${WEAVIATE_PORT:-2211}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/benchmark_runs}"
BENCH_ROOT="${BENCH_ROOT:-}"
VARIANTS_RAW="${VARIANTS:-vanilla,atom_gate,atomic}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
EPOCHS="${EPOCHS:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
LR_ENCODER="${LR_ENCODER:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TEMPERATURE="${TEMPERATURE:-0.05}"
NSAMPLES="${NSAMPLES:-}"
SEARCH_PIPELINE="${SEARCH_PIPELINE:-}"
SEARCH_TOP_K="${TOP_K:-}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-}"
AUTO_E5_PREFIXES="${AUTO_E5_PREFIXES:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
BENCHMARK_RESOURCE_LOG="${BENCHMARK_RESOURCE_LOG:-1}"
BENCHMARK_RESOURCE_TOP_N="${BENCHMARK_RESOURCE_TOP_N:-8}"
DRY_RUN=0

MEMORY_BANK_SIZE="${MEMORY_BANK_SIZE:-512}"
MEMORY_BANK_MINING_MODE="${MEMORY_BANK_MINING_MODE:-mixed}"
MEMORY_BANK_WARMUP_STEPS="${MEMORY_BANK_WARMUP_STEPS:-50}"
MEMORY_BANK_HARD_NEGATIVES="${MEMORY_BANK_HARD_NEGATIVES:-4}"
MEMORY_BANK_RANDOM_NEGATIVES="${MEMORY_BANK_RANDOM_NEGATIVES:-12}"
MEMORY_BANK_HARD_WARMUP_STEPS="${MEMORY_BANK_HARD_WARMUP_STEPS:-120}"
MEMORY_BANK_HARD_RAMP_STEPS="${MEMORY_BANK_HARD_RAMP_STEPS:-200}"
MEMORY_BANK_TOO_HARD_MARGIN="${MEMORY_BANK_TOO_HARD_MARGIN:-0.05}"
MEMORY_BANK_HARD_SIMILARITY_CAP="${MEMORY_BANK_HARD_SIMILARITY_CAP:-}"
MEMORY_BANK_ADAPTIVE_HARD="${MEMORY_BANK_ADAPTIVE_HARD:-1}"
MEMORY_BANK_ADAPTIVE_HARD_MODE="${MEMORY_BANK_ADAPTIVE_HARD_MODE:-soft}"
MEMORY_BANK_HARD_COLLISION_THRESHOLD="${MEMORY_BANK_HARD_COLLISION_THRESHOLD:-0.0}"
MEMORY_BANK_HARD_COLLISION_BETA="${MEMORY_BANK_HARD_COLLISION_BETA:-0.05}"
MEMORY_BANK_SOFT_MODE="${MEMORY_BANK_SOFT_MODE:-soft}"
MEMORY_BANK_SOFT_BETA="${MEMORY_BANK_SOFT_BETA:-0.05}"
MEMORY_BANK_MARGIN_HEAD="${MEMORY_BANK_MARGIN_HEAD:-1}"
MEMORY_BANK_MARGIN_REG_WEIGHT="${MEMORY_BANK_MARGIN_REG_WEIGHT:-50}"

ALPHA_GATE_LAYERS="${ALPHA_GATE_LAYERS:-}"
ALPHA_GATE_HIDDEN_DIM="${ALPHA_GATE_HIDDEN_DIM:-}"
ALPHA_GATE_DROPOUT="${ALPHA_GATE_DROPOUT:-}"
ALPHA_GATE_INPUT="${ALPHA_GATE_INPUT:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_benchmark.sh [options]

Runs the core JustAtom ablation matrix through scripts/run_pipeline.sh and
collects retrieval + training-geometry summaries in one benchmark directory.

Default variants:
  vanilla          contrastive InfoNCE, no alpha, no memory bank
  atom_gate        alpha(q), no memory bank
  atomic          alpha(q) + dynamic/adaptive memory bank + m(q)

Main options:
  --dataset-ids IDS          Comma-separated dataset preset ids
  --model MODEL              Base model or checkpoint
  --variants IDS             Comma-separated variants, or "all"
  --batch-size N             Train batch size, default 32
  --epochs N                 Number of epochs, default 2
  --grad-acc-steps N         Gradient accumulation steps, default 1
  --temperature VALUE        Shared contrastive temperature, default 0.05
  --nsamples N               Limit train/eval samples
  --output-root DIR          Benchmark artifacts root
  --bench-root DIR           Exact benchmark directory
  --wandb-mode MODE          offline | online | disabled
  --no-baseline              Skip baseline eval in every pipeline run
  --dry-run                  Print pipeline commands without running them

Memory bank options:
  --memory-bank-size N
  --memory-bank-mining MODE  all | random | hard | mixed
  --memory-bank-hard-negatives N
  --memory-bank-random-negatives N
  --memory-bank-warmup-steps N
  --memory-bank-hard-warmup-steps N
  --memory-bank-hard-ramp-steps N
  --memory-bank-too-hard-margin VALUE
  --memory-bank-hard-similarity-cap VALUE
  --memory-bank-adaptive-hard
  --memory-bank-adaptive-hard-mode MODE
  --memory-bank-hard-collision-threshold VALUE
  --memory-bank-hard-collision-beta VALUE
  --memory-bank-soft-mode MODE  hard | soft-const | soft
  --memory-bank-soft-beta VALUE
  --memory-bank-margin-head
  --memory-bank-margin-reg-weight VALUE

Alpha gate options:
  --alpha-gate-layers N
  --alpha-gate-hidden-dim N
  --alpha-gate-dropout P
  --alpha-gate-input MODE    query | query_doc

Eval options:
  --weaviate-host HOST
  --weaviate-port PORT
  --search-pipeline NAME
  --top-k N
  --index-batch-size N
  --auto-e5-prefixes
  --no-auto-e5-prefixes

Example:
  scripts/run_benchmark.sh \
    --dataset-ids justatom,boolq-ru,electrical-engineering-ru,meme-russian-ir \
    --model intfloat/multilingual-e5-small \
    --batch-size 32 \
    --epochs 2
EOF
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

slugify() {
  printf '%s' "$1" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-'
}

normalize_variant() {
  case "$1" in
    vanilla|baseline|contrastive|infonce)
      printf 'vanilla'
      ;;
    atom_gate|atom-gate|atom|gate|justatom_gate|justatom-gate)
      printf 'atom_gate'
      ;;
    bank_only|bank-only|memory_bank|memory-bank|bank|atom_gate_bank|atom-gate-bank|gate_bank|gate-bank|atom_bank|atom-bank|atom_gate_dynamic|atom-gate-dynamic|dynamic)
      echo "retired benchmark variant: $1. Public variants are: vanilla, atom_gate, atomic. Use atomic for alpha(q) + dynamic/adaptive bank + m(q)." >&2
      exit 1
      ;;
    atomic)
      printf 'atomic'
      ;;
    *)
      echo "Unsupported benchmark variant: $1" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-ids|--datasets)
      DATASET_IDS_RAW="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --variants)
      VARIANTS_RAW="$2"
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
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --nsamples)
      NSAMPLES="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --bench-root)
      BENCH_ROOT="$2"
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
    --index-batch-size)
      INDEX_BATCH_SIZE="$2"
      shift 2
      ;;
    --memory-bank-size)
      MEMORY_BANK_SIZE="$2"
      shift 2
      ;;
    --memory-bank-mining)
      MEMORY_BANK_MINING_MODE="$2"
      shift 2
      ;;
    --memory-bank-warmup-steps)
      MEMORY_BANK_WARMUP_STEPS="$2"
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
    --memory-bank-hard-similarity-cap)
      MEMORY_BANK_HARD_SIMILARITY_CAP="$2"
      shift 2
      ;;
    --memory-bank-adaptive-hard)
      MEMORY_BANK_ADAPTIVE_HARD=1
      shift
      ;;
    --memory-bank-adaptive-hard-mode)
      MEMORY_BANK_ADAPTIVE_HARD_MODE="$2"
      shift 2
      ;;
    --memory-bank-hard-collision-threshold)
      MEMORY_BANK_HARD_COLLISION_THRESHOLD="$2"
      shift 2
      ;;
    --memory-bank-hard-collision-beta)
      MEMORY_BANK_HARD_COLLISION_BETA="$2"
      shift 2
      ;;
    --memory-bank-soft-mode)
      MEMORY_BANK_SOFT_MODE="$2"
      shift 2
      ;;
    --memory-bank-soft-beta)
      MEMORY_BANK_SOFT_BETA="$2"
      shift 2
      ;;
    --memory-bank-margin-head)
      MEMORY_BANK_MARGIN_HEAD=1
      shift
      ;;
    --memory-bank-margin-reg-weight|--memory-bank-margin-regularization-weight)
      MEMORY_BANK_MARGIN_REG_WEIGHT="$2"
      shift 2
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
    --auto-e5-prefixes)
      AUTO_E5_PREFIXES=1
      shift
      ;;
    --no-auto-e5-prefixes)
      AUTO_E5_PREFIXES=0
      shift
      ;;
    --no-baseline)
      RUN_BASELINE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
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
  exit 1
fi

if [[ "$VARIANTS_RAW" == "all" ]]; then
  VARIANTS_RAW="vanilla,atom_gate,atomic"
fi

IFS=',' read -r -a RAW_VARIANTS <<< "$VARIANTS_RAW"
VARIANTS=()
declare -A SEEN_VARIANTS=()
for raw_variant in "${RAW_VARIANTS[@]}"; do
  variant="$(trim "$raw_variant")"
  [[ -n "$variant" ]] || continue
  variant="$(normalize_variant "$variant")"
  if [[ -z "${SEEN_VARIANTS[$variant]:-}" ]]; then
    VARIANTS+=("$variant")
    SEEN_VARIANTS[$variant]=1
  fi
done

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
  echo "No benchmark variants selected" >&2
  exit 1
fi

case "$MEMORY_BANK_SOFT_MODE" in
  hard|soft-const|soft)
    ;;
  *)
    echo "Unsupported --memory-bank-soft-mode: $MEMORY_BANK_SOFT_MODE" >&2
    exit 1
    ;;
esac

case "$MEMORY_BANK_MARGIN_HEAD" in
  0|1)
    ;;
  *)
    echo "Unsupported --memory-bank-margin-head value: $MEMORY_BANK_MARGIN_HEAD" >&2
    exit 1
    ;;
esac

case "$MEMORY_BANK_ADAPTIVE_HARD" in
  0|1|true|false|yes|no|on|off)
    ;;
  *)
    echo "Unsupported --memory-bank-adaptive-hard value: $MEMORY_BANK_ADAPTIVE_HARD" >&2
    exit 1
    ;;
esac

case "$MEMORY_BANK_ADAPTIVE_HARD_MODE" in
  hard|soft)
    ;;
  *)
    echo "Unsupported --memory-bank-adaptive-hard-mode: $MEMORY_BANK_ADAPTIVE_HARD_MODE" >&2
    exit 1
    ;;
esac

if [[ "$MEMORY_BANK_MARGIN_HEAD" == "1" && "$MEMORY_BANK_SOFT_MODE" == "soft-const" ]]; then
  echo "--memory-bank-margin-head requires --memory-bank-soft-mode soft" >&2
  exit 1
fi
if [[ "$MEMORY_BANK_MARGIN_HEAD" == "1" && "$MEMORY_BANK_SOFT_MODE" == "hard" ]]; then
  MEMORY_BANK_SOFT_MODE="soft"
fi
if [[ "$MEMORY_BANK_SOFT_MODE" == "soft" ]]; then
  MEMORY_BANK_MARGIN_HEAD=1
fi

if [[ -z "$BENCH_ROOT" ]]; then
  BENCH_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$(slugify "$MODEL_NAME_OR_PATH")"
fi

mkdir -p "$BENCH_ROOT"/{logs,pipeline_runs,tables}

SUMMARY_PATH="$BENCH_ROOT/BENCHMARK_RESULTS.md"
GEOMETRY_PATH="$BENCH_ROOT/GEOMETRY_RESULTS.md"
COMMANDS_PATH="$BENCH_ROOT/COMMANDS.md"

cat > "$SUMMARY_PATH" <<EOF
# Benchmark Results

- Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Model: $MODEL_NAME_OR_PATH
- Datasets: $DATASET_IDS_RAW
- Variants: ${VARIANTS[*]}
- Batch size: $BATCH_SIZE
- Grad accumulation: $GRAD_ACC_STEPS
- Epochs: $EPOCHS
- Temperature: $TEMPERATURE
- Memory bank: size=$MEMORY_BANK_SIZE, mining=$MEMORY_BANK_MINING_MODE, hard=$MEMORY_BANK_HARD_NEGATIVES, random=$MEMORY_BANK_RANDOM_NEGATIVES, warmup=$MEMORY_BANK_WARMUP_STEPS, hard_warmup=$MEMORY_BANK_HARD_WARMUP_STEPS, hard_ramp=$MEMORY_BANK_HARD_RAMP_STEPS, hard_similarity_cap=${MEMORY_BANK_HARD_SIMILARITY_CAP:-none}, adaptive_hard=$MEMORY_BANK_ADAPTIVE_HARD, adaptive_hard_mode=$MEMORY_BANK_ADAPTIVE_HARD_MODE, hard_collision_threshold=$MEMORY_BANK_HARD_COLLISION_THRESHOLD, hard_collision_beta=$MEMORY_BANK_HARD_COLLISION_BETA, soft=$MEMORY_BANK_SOFT_MODE, margin_reg=$MEMORY_BANK_MARGIN_REG_WEIGHT
- N samples: ${NSAMPLES:-all}

| Variant | Dataset | Tuning Method | Baseline Status | Base HR@1 | Base HR@5 | Base HR@10 | Base MRR@10 | Base NDCG@10 | Tuned Status | Tuned HR@1 | Tuned HR@5 | Tuned HR@10 | Tuned MRR@10 | Tuned NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
EOF

cat > "$COMMANDS_PATH" <<EOF
# Benchmark Commands

EOF

append_variant_rows() {
  local variant="$1"
  local table_path="$2"

  if [[ ! -s "$table_path" ]]; then
    printf '| `%s` | `_missing_table_` |  |  |  |  |  |  |  |  |  |  |  |  |  |\n' "$variant" >> "$SUMMARY_PATH"
    return
  fi

  awk -v variant="$variant" '
    /^\| `/ {
      row = $0
      sub(/^\| /, "", row)
      printf("| `%s` | %s\n", variant, row)
    }
  ' "$table_path" >> "$SUMMARY_PATH"
}

append_command_markdown() {
  local variant="$1"
  shift

  {
    printf '## %s\n\n' "$variant"
    printf '```bash\n'
    printf '%q ' "$@"
    printf '\n```\n\n'
  } >> "$COMMANDS_PATH"
}

log_rss() {
  local label="$1"
  local snapshot=""
  if [[ "$BENCHMARK_RESOURCE_LOG" != "1" ]]; then
    return 0
  fi
  if snapshot="$("$PYTHON_BIN" -m justatom.tooling.resources --label "$label" --pid "$$" --top "$BENCHMARK_RESOURCE_TOP_N" 2>/dev/null)"; then
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$snapshot" | tee -a "$BENCH_ROOT/logs/resource.log"
  else
    printf '[%s] RSS %s: unavailable\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" | tee -a "$BENCH_ROOT/logs/resource.log"
  fi
}

add_common_pipeline_args() {
  command+=(
    --dataset-ids "$DATASET_IDS_RAW"
    --model "$MODEL_NAME_OR_PATH"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --max-seq-len "$MAX_SEQ_LEN"
    --epochs "$EPOCHS"
    --grad-acc-steps "$GRAD_ACC_STEPS"
    --lr-encoder "$LR_ENCODER"
    --weight-decay "$WEIGHT_DECAY"
    --temperature "$TEMPERATURE"
    --wandb-mode "$WANDB_MODE_VALUE"
    --wandb-project "$WANDB_PROJECT"
    --weaviate-host "$WEAVIATE_HOST_VALUE"
    --weaviate-port "$WEAVIATE_PORT_VALUE"
  )

  if [[ "$AUTO_E5_PREFIXES" == "1" ]]; then
    command+=(--auto-e5-prefixes)
  fi
  if [[ "$RUN_BASELINE" == "0" ]]; then
    command+=(--no-baseline)
  fi
  if [[ -n "$NSAMPLES" ]]; then
    command+=(--nsamples "$NSAMPLES")
  fi
  if [[ -n "$SEARCH_PIPELINE" ]]; then
    command+=(--search-pipeline "$SEARCH_PIPELINE")
  fi
  if [[ -n "$SEARCH_TOP_K" ]]; then
    command+=(--top-k "$SEARCH_TOP_K")
  fi
  if [[ -n "$INDEX_BATCH_SIZE" ]]; then
    command+=(--index-batch-size "$INDEX_BATCH_SIZE")
  fi
}

add_memory_bank_args() {
  command+=(
    --memory-bank-size "$MEMORY_BANK_SIZE"
    --memory-bank-mining "$MEMORY_BANK_MINING_MODE"
    --memory-bank-warmup-steps "$MEMORY_BANK_WARMUP_STEPS"
    --memory-bank-hard-negatives "$MEMORY_BANK_HARD_NEGATIVES"
    --memory-bank-random-negatives "$MEMORY_BANK_RANDOM_NEGATIVES"
    --memory-bank-hard-warmup-steps "$MEMORY_BANK_HARD_WARMUP_STEPS"
    --memory-bank-hard-ramp-steps "$MEMORY_BANK_HARD_RAMP_STEPS"
    --memory-bank-adaptive-hard-mode "$MEMORY_BANK_ADAPTIVE_HARD_MODE"
    --memory-bank-hard-collision-threshold "$MEMORY_BANK_HARD_COLLISION_THRESHOLD"
    --memory-bank-hard-collision-beta "$MEMORY_BANK_HARD_COLLISION_BETA"
    --memory-bank-soft-mode "$MEMORY_BANK_SOFT_MODE"
    --memory-bank-margin-reg-weight "$MEMORY_BANK_MARGIN_REG_WEIGHT"
  )

  if [[ -n "$MEMORY_BANK_TOO_HARD_MARGIN" ]]; then
    command+=(--memory-bank-too-hard-margin "$MEMORY_BANK_TOO_HARD_MARGIN")
  fi
  if [[ -n "$MEMORY_BANK_HARD_SIMILARITY_CAP" ]]; then
    command+=(--memory-bank-hard-similarity-cap "$MEMORY_BANK_HARD_SIMILARITY_CAP")
  fi
  if [[ "$MEMORY_BANK_ADAPTIVE_HARD" == "1" || "$MEMORY_BANK_ADAPTIVE_HARD" == "true" || "$MEMORY_BANK_ADAPTIVE_HARD" == "yes" || "$MEMORY_BANK_ADAPTIVE_HARD" == "on" ]]; then
    command+=(--memory-bank-adaptive-hard)
  fi
  if [[ -n "$MEMORY_BANK_SOFT_BETA" ]]; then
    command+=(--memory-bank-soft-beta "$MEMORY_BANK_SOFT_BETA")
  fi
  if [[ "$MEMORY_BANK_MARGIN_HEAD" == "1" ]]; then
    command+=(--memory-bank-margin-head)
  fi
}

add_alpha_gate_args() {
  if [[ -n "$ALPHA_GATE_LAYERS" ]]; then
    command+=(--alpha-gate-layers "$ALPHA_GATE_LAYERS")
  fi
  if [[ -n "$ALPHA_GATE_HIDDEN_DIM" ]]; then
    command+=(--alpha-gate-hidden-dim "$ALPHA_GATE_HIDDEN_DIM")
  fi
  if [[ -n "$ALPHA_GATE_DROPOUT" ]]; then
    command+=(--alpha-gate-dropout "$ALPHA_GATE_DROPOUT")
  fi
  if [[ -n "$ALPHA_GATE_INPUT" ]]; then
    command+=(--alpha-gate-input "$ALPHA_GATE_INPUT")
  fi
}

build_variant_command() {
  local variant="$1"
  local output_root="$2"
  local table_path="$3"

  command=("$PIPELINE_SHELL" "$REPO_ROOT/scripts/run_pipeline.sh")
  add_common_pipeline_args
  command+=(--output-root "$output_root" --table-results "$table_path")

  case "$variant" in
    vanilla)
      command+=(--loss contrastive --optimizer adamw --alpha-mode off)
      ;;
    atom_gate)
      command+=(--recipe atom_gate)
      add_alpha_gate_args
      ;;
    atomic)
      command+=(--recipe atom_gate)
      add_memory_bank_args
      add_alpha_gate_args
      ;;
    *)
      echo "Unsupported benchmark variant: $variant" >&2
      exit 1
      ;;
  esac
}

write_geometry_summary() {
  "$PYTHON_BIN" - "$BENCH_ROOT" "$GEOMETRY_PATH" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_path = Path(sys.argv[2])

columns = [
    ("Align", "Geom/Alignment"),
    ("UnifQ", "Geom/UniformityQ"),
    ("UnifD", "Geom/UniformityD"),
    ("PosSim", "Geom/PositiveSimMean"),
    ("NegSim", "Geom/NegativeSimMean"),
    ("NegMax", "Geom/NegativeSimMax"),
    ("Gap", "Geom/SimGap"),
    ("RankQ", "Geom/EffectiveRankQ"),
    ("RankD", "Geom/EffectiveRankD"),
    ("AnisoQ", "Geom/AnisotropyQ"),
    ("AnisoD", "Geom/AnisotropyD"),
    ("EffNeg", "ContrastiveEffectiveNegativesMean"),
    ("AlphaMean", "ContrastiveLossAlphaGateAlphaMean"),
    ("AlphaStd", "ContrastiveLossAlphaGateAlphaStd"),
    ("AlphaP05", "ContrastiveLossAlphaGateAlphaP05"),
    ("AlphaP50", "ContrastiveLossAlphaGateAlphaP50"),
    ("AlphaP95", "ContrastiveLossAlphaGateAlphaP95"),
    ("BankActive", "MemoryBankActiveNegativesMean"),
    ("BankValid", "MemoryBankValidNegativesMean"),
    ("BankSimP95", "MemoryBankActiveSimilarityP95"),
    ("BankSimMax", "MemoryBankActiveSimilarityMax"),
    ("BankGapP05", "MemoryBankActivePositiveGapP05"),
    ("BankGapMin", "MemoryBankActivePositiveGapMin"),
    ("HardSimP95", "MemoryBankActiveHardSimilarityP95"),
    ("HardSimMax", "MemoryBankActiveHardSimilarityMax"),
    ("HardGapMin", "MemoryBankActiveHardPositiveGapMin"),
    ("HardCand", "MemoryBankHardCandidateNegativesMean"),
    ("HardActive", "MemoryBankActiveHardNegativesMean"),
    ("HardAllow", "MemoryBankAdaptiveHardAllowedMean"),
    ("HardSupp", "MemoryBankAdaptiveHardSuppressedMean"),
    ("HardSuppRows", "MemoryBankAdaptiveHardSuppressedRows"),
    ("HardWMean", "MemoryBankAdaptiveHardWeightMean"),
    ("HardWP05", "MemoryBankAdaptiveHardWeightP05"),
    ("HardWP95", "MemoryBankAdaptiveHardWeightP95"),
    ("RandActive", "MemoryBankActiveRandomNegativesMean"),
    ("ValidSimMax", "MemoryBankValidSimilarityMax"),
    ("CollGMean", "MemoryBankCollisionGMean"),
    ("CollGP95", "MemoryBankCollisionGP95"),
    ("CollGMax", "MemoryBankCollisionGMax"),
    ("CollMax", "MemoryBankCollisionBankMaxSimilarityMax"),
    ("MMean", "ContrastiveMemoryMarginMean"),
    ("MStd", "ContrastiveMemoryMarginStd"),
    ("MP05", "ContrastiveMemoryMarginP05"),
    ("MP95", "ContrastiveMemoryMarginP95"),
    ("MRawMean", "ContrastiveMemoryMarginRawMean"),
    ("MRawStd", "ContrastiveMemoryMarginRawStd"),
    ("MReg", "ContrastiveMemoryMarginRegLoss"),
    ("MRegW", "ContrastiveMemoryMarginRegWeight"),
    ("GradM", "Grad_norm_margin_head"),
    ("SoftMode", "ContrastiveMemorySoftMode"),
    ("HardK", "MemoryBankActiveHardK"),
    ("RandomK", "MemoryBankActiveRandomK"),
]


def fmt(value):
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.4f}"


def last_row(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row]
    return rows[-1] if rows else {}


rows = []
for metrics_path in sorted(root.glob("pipeline_runs/*/*/*/tuned_metrics.csv")):
    rel_parts = metrics_path.relative_to(root).parts
    if len(rel_parts) < 5:
        continue
    variant = rel_parts[1]
    run_id = rel_parts[2]
    dataset = rel_parts[3]
    row = last_row(metrics_path)
    rows.append((variant, dataset, run_id, row))

with out_path.open("w") as out:
    out.write("# Geometry Results\n\n")
    out.write(
        "Last logged training row per variant/dataset. These diagnostics are for the training dynamics, "
        "not direct retrieval metrics.\n\n"
    )
    header = ["Variant", "Dataset", "Run"] + [label for label, _ in columns]
    out.write("| " + " | ".join(header) + " |\n")
    out.write("| " + " | ".join(["---"] * len(header)) + " |\n")
    for variant, dataset, run_id, row in rows:
        values = [f"`{variant}`", f"`{dataset}`", f"`{run_id}`"]
        values.extend(fmt(row.get(key)) for _, key in columns)
        out.write("| " + " | ".join(values) + " |\n")
PY
}

echo "Benchmark root: $BENCH_ROOT"
echo "Results summary: $SUMMARY_PATH"
echo "Geometry summary: $GEOMETRY_PATH"
echo "Variants: ${VARIANTS[*]}"
if [[ "$DRY_RUN" != "1" ]]; then
  log_rss "benchmark start"
fi

for variant in "${VARIANTS[@]}"; do
  variant_root="$BENCH_ROOT/pipeline_runs/$variant"
  variant_table="$BENCH_ROOT/tables/$variant.md"
  variant_log="$BENCH_ROOT/logs/$variant.log"
  command=()
  build_variant_command "$variant" "$variant_root" "$variant_table"
  append_command_markdown "$variant" "${command[@]}"

  echo
  echo "=== $variant ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi

  log_rss "before variant $variant"
  "${command[@]}" 2>&1 | tee "$variant_log"
  log_rss "after variant $variant"
  append_variant_rows "$variant" "$variant_table"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run complete. Commands were saved to $COMMANDS_PATH"
  exit 0
fi

write_geometry_summary
log_rss "benchmark end"

echo
echo "Benchmark finished."
echo "Results summary: $SUMMARY_PATH"
echo "Geometry summary: $GEOMETRY_PATH"
echo "Commands: $COMMANDS_PATH"
