#!/usr/bin/env bash
set -euo pipefail

if (( BASH_VERSINFO[0] < 4 )); then
  echo "scripts/run_benchmark.sh requires Bash >= 4" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PIPELINE_SHELL="${PIPELINE_SHELL:-${BASH:-bash}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-intfloat/multilingual-e5-small}"
DATASET_IDS_RAW="${DATASET_IDS:-justatom}"
VARIANTS_RAW="${VARIANTS:-vanilla,atom_gate,atomic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/benchmark_runs}"
BENCH_ROOT="${BENCH_ROOT:-}"
DRY_RUN=0

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-1}"
TEMPERATURE="${TEMPERATURE:-0.05}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WANDB_MODE_VALUE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-justatom-benchmark}"
WEAVIATE_URL="${WEAVIATE_URL:-http://${WEAVIATE_HOST:-localhost}:${WEAVIATE_PORT:-2211}}"
WEAVIATE_GRPC_PORT="${WEAVIATE_GRPC_PORT:-50051}"
NSAMPLES=""
AUTO_E5_PREFIXES=0
RUN_BASELINE=1
SEARCH_MODE="${SEARCH_MODE:-}"
TRAIN_CONFIG=""
PIPELINE_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_benchmark.sh [options]

Runs exactly three public methods: vanilla, atom_gate, atomic.

Options:
  --dataset-ids IDS
  --model MODEL
  --variants vanilla,atom_gate,atomic|all
  --batch-size N
  --epochs N
  --grad-acc-steps N
  --temperature VALUE
  --train-config PATH
  --aux-gradient-mode off|observe|safe
  --aux-gradient-max-norm-ratio VALUE
  --aux-gradient-eps VALUE
  --memory-bank-mass-ratio VALUE
  --memory-bank-mass-ramp-steps N
  --nsamples N
  --bench-root DIR
  --output-root DIR
  --wandb-mode disabled|offline|online
  --weaviate-url URL
  --weaviate-grpc-port PORT
  --search-mode keyword|vector|hybrid
  --no-baseline
  --auto-e5-prefixes
  --dry-run

Any component override accepted by run_pipeline.sh can also be supplied here.
EOF
}

slugify() {
  printf '%s' "$1" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-'
}

log_rss() {
  "${PYTHON_BIN:-python}" -m justatom.tooling.resources \
    --label "$1" \
    --pid "$$" \
    --top 8 | tee -a "$RESOURCE_LOG"
}

normalize_variant() {
  case "$1" in
    vanilla|atom_gate|atomic) printf '%s' "$1" ;;
    *) echo "invalid variant: $1; expected vanilla,atom_gate,atomic" >&2; exit 2 ;;
  esac
}

normalize_search_mode() {
  case "$1" in
    keyword|vector|hybrid) printf '%s' "$1" ;;
    *) echo "invalid search mode: $1; expected keyword, vector, or hybrid" >&2; exit 2 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-ids|--datasets) DATASET_IDS_RAW="$2"; shift 2 ;;
    --model) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --variants) VARIANTS_RAW="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --grad-acc-steps) GRAD_ACC_STEPS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --nsamples) NSAMPLES="$2"; shift 2 ;;
    --bench-root) BENCH_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE_VALUE="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --weaviate-url) WEAVIATE_URL="$2"; shift 2 ;;
    --weaviate-grpc-port) WEAVIATE_GRPC_PORT="$2"; shift 2 ;;
    --search-mode) SEARCH_MODE="$2"; shift 2 ;;
    --auto-e5-prefixes) AUTO_E5_PREFIXES=1; shift ;;
    --no-auto-e5-prefixes) AUTO_E5_PREFIXES=0; shift ;;
    --no-baseline) RUN_BASELINE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --memory-bank-size|--memory-bank-warmup-steps|--memory-bank-mass-ratio|--memory-bank-mass-ramp-steps|--memory-bank-mining|--memory-bank-hard-negatives|--memory-bank-random-negatives|--memory-bank-hard-warmup-steps|--memory-bank-hard-ramp-steps|--memory-bank-collision-threshold|--memory-bank-collision-beta|--memory-bank-margin-mode|--memory-bank-margin-base|--memory-bank-margin-scale|--memory-bank-margin-min|--memory-bank-margin-max|--memory-bank-admission-beta|--memory-bank-margin-reg-weight|--alpha-gate-layers|--alpha-gate-hidden-dim|--alpha-gate-dropout|--aux-gradient-mode|--aux-gradient-max-norm-ratio|--aux-gradient-eps|--experiment-role|--lr-encoder|--lr-heads|--weight-decay|--max-query-seq-len|--top-k|--index-batch-size)
      PIPELINE_OVERRIDES+=("$1" "$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$VARIANTS_RAW" == "all" ]]; then
  VARIANTS_RAW="vanilla,atom_gate,atomic"
fi
IFS=',' read -r -a RAW_VARIANTS <<< "$VARIANTS_RAW"
VARIANTS=()
for raw in "${RAW_VARIANTS[@]}"; do
  VARIANTS+=("$(normalize_variant "$raw")")
done
if [[ -n "$SEARCH_MODE" ]]; then
  SEARCH_MODE="$(normalize_search_mode "$SEARCH_MODE")"
fi

if [[ -z "$BENCH_ROOT" ]]; then
  BENCH_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$(slugify "$MODEL_NAME_OR_PATH")"
fi
mkdir -p "$BENCH_ROOT/tables" "$BENCH_ROOT/pipeline_runs"
COMMANDS_PATH="$BENCH_ROOT/COMMANDS.md"
RESULTS_PATH="$BENCH_ROOT/BENCHMARK_RESULTS.md"
GEOMETRY_PATH="$BENCH_ROOT/GEOMETRY_RESULTS.md"
RESOURCE_LOG="$BENCH_ROOT/RESOURCES.log"

printf '# Benchmark Commands\n\n' > "$COMMANDS_PATH"
printf 'Benchmark root: %s\n' "$BENCH_ROOT"
printf 'Results summary: %s\n' "$RESULTS_PATH"
printf 'Geometry summary: %s\n' "$GEOMETRY_PATH"
printf 'Variants: %s\n' "${VARIANTS[*]}"
if [[ "$DRY_RUN" == "0" ]]; then
  log_rss "benchmark start"
fi

build_variant_command() {
  local variant="$1" output_root="$2" table_path="$3"
  command=("$PIPELINE_SHELL" "$REPO_ROOT/scripts/run_pipeline.sh")
  command+=(
    --method "$variant"
    --dataset-ids "$DATASET_IDS_RAW"
    --model "$MODEL_NAME_OR_PATH"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --grad-acc-steps "$GRAD_ACC_STEPS"
    --temperature "$TEMPERATURE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --wandb-mode "$WANDB_MODE_VALUE"
    --wandb-project "$WANDB_PROJECT"
    --weaviate-url "$WEAVIATE_URL"
    --weaviate-grpc-port "$WEAVIATE_GRPC_PORT"
    --output-root "$output_root"
    --table-results "$table_path"
  )
  [[ -z "$TRAIN_CONFIG" ]] || command+=(--train-config "$TRAIN_CONFIG")
  [[ -z "$NSAMPLES" ]] || command+=(--nsamples "$NSAMPLES")
  [[ -z "$SEARCH_MODE" ]] || command+=(--search-mode "$SEARCH_MODE")
  [[ "$AUTO_E5_PREFIXES" == "0" ]] || command+=(--auto-e5-prefixes)
  [[ "$RUN_BASELINE" == "1" ]] || command+=(--no-baseline)
  command+=("${PIPELINE_OVERRIDES[@]}")
}

for variant in "${VARIANTS[@]}"; do
  output_root="$BENCH_ROOT/pipeline_runs/$variant"
  table_path="$BENCH_ROOT/tables/$variant.md"
  build_variant_command "$variant" "$output_root" "$table_path"

  {
    printf '## %s\n\n```bash\n' "$variant"
    printf '%q ' "${command[@]}"
    printf '\n```\n\n'
  } >> "$COMMANDS_PATH"

  printf '\n=== %s ===\n' "$variant"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done

{
  printf '# Benchmark Results\n\n'
  printf -- '- Model: `%s`\n' "$MODEL_NAME_OR_PATH"
  printf -- '- Datasets: `%s`\n\n' "$DATASET_IDS_RAW"
  for variant in "${VARIANTS[@]}"; do
    printf '## %s\n\n' "$variant"
    if [[ -f "$BENCH_ROOT/tables/$variant.md" ]]; then
      sed -n '/^| Dataset |/,$p' "$BENCH_ROOT/tables/$variant.md"
    else
      printf '_No results: dry run or failed pipeline._\n'
    fi
    printf '\n'
  done
} > "$RESULTS_PATH"

"${PYTHON_BIN:-python}" - "$BENCH_ROOT" "$GEOMETRY_PATH" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
destination = Path(sys.argv[2])
fields = [
    "temperature",
    "batch/hit_rate_at_1",
    "batch/mrr",
    "alpha/mean",
    "memory/active_negatives_mean",
    "margin/bounded/mean",
    "margin/bounded/std",
]
rows = []
for path in root.glob("pipeline_runs/*/**/tuned_metrics.csv"):
    with path.open(encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    if values:
        rows.append((path.parts[-4], path.parts[-2], values[-1]))
with destination.open("w", encoding="utf-8") as handle:
    handle.write("# Geometry Results\n\n")
    handle.write("| Variant | Dataset | " + " | ".join(fields) + " |\n")
    handle.write("| --- | --- | " + " | ".join("---" for _ in fields) + " |\n")
    for variant, dataset, row in rows:
        handle.write(f"| {variant} | {dataset} | " + " | ".join(row.get(field, "NA") for field in fields) + " |\n")
PY

printf '\nBenchmark finished.\n'
printf 'Results summary: %s\n' "$RESULTS_PATH"
printf 'Geometry summary: %s\n' "$GEOMETRY_PATH"
printf 'Commands: %s\n' "$COMMANDS_PATH"
if [[ "$DRY_RUN" == "0" ]]; then
  log_rss "benchmark end"
fi
