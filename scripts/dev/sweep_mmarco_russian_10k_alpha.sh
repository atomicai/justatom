#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/thebat/minicondaenvs/envs/justatom/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/mmarco_russian_10k_alpha_sweep}"
WEAVIATE_HOST="${WEAVIATE_HOST:-localhost}"
WEAVIATE_PORT="${WEAVIATE_PORT:-2211}"
DATASET_ID="mmarco-russian"
DATASET_LIMIT="10000"
MODEL_NAME="intfloat/multilingual-e5-small"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"

mkdir -p "$OUTPUT_ROOT"
SUMMARY_CSV="$OUTPUT_ROOT/summary.csv"

metric_from_csv() {
  local csv_path="$1"
  local metric_name="$2"
  awk -F',' -v metric="$metric_name" '$1 == metric { print $2; exit }' "$csv_path"
}

latest_metrics_csv() {
  local dir="$1"
  ls -1t "$dir"/*.csv 2>/dev/null | head -n 1 || true
}

run_candidate() {
  local label="$1"
  local temperature="$2"
  local mix_weight="$3"

  local train_dir="$OUTPUT_ROOT/$label/train"
  local eval_dir="$OUTPUT_ROOT/$label/eval"
  local checkpoint_dir="$train_dir/BiGamma/epoch1"
  local train_metrics="$train_dir/train_metrics.csv"

  mkdir -p "$train_dir" "$eval_dir"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "[train] $label temp=$temperature mix=$mix_weight"
    WANDB_MODE=offline "$PYTHON_BIN" -m justatom.api.train \
      --config configs/train.yaml \
      --dataset.id "$DATASET_ID" \
      --dataset.limit "$DATASET_LIMIT" \
      --model.name "$MODEL_NAME" \
      --training.loss contrastive \
      --training.num_samples "$NUM_SAMPLES" \
      --training.batch_size "$TRAIN_BATCH_SIZE" \
      --training.n_epochs "$EPOCHS" \
      --training.freeze_encoder false \
      --training.optimizer adamw \
      --training.contrastive_temperature "$temperature" \
      --training.gamma_joint true \
      --training.include_semantic_gamma true \
      --training.include_keywords_gamma true \
      --training.alpha_train_only true \
      --training.alpha_mix_weight "$mix_weight" \
      --training.alpha_entropy_weight 0.0 \
      --output.save_dir "$train_dir" \
      --output.metrics_path "$train_metrics" \
      --logging.backend csv
  else
    echo "[skip-train] $label checkpoint already exists"
  fi

  local eval_csv
  eval_csv="$(latest_metrics_csv "$eval_dir")"
  if [[ -z "$eval_csv" ]]; then
    echo "[eval] $label"
    WANDB_MODE=offline "$PYTHON_BIN" -m justatom.api.eval \
      --config configs/evaluate.yaml \
      --dataset.id "$DATASET_ID" \
      --dataset.limit "$DATASET_LIMIT" \
      --model.name "$checkpoint_dir" \
      --collection.name "MMarcoRussian10k_${label}" \
      --output.save_results_to_dir "$eval_dir" \
      --search.batch_size "$EVAL_BATCH_SIZE" \
      --index.batch_size 32 \
      --index.flush_collection true \
      --weaviate.host "$WEAVIATE_HOST" \
      --weaviate.port "$WEAVIATE_PORT"
    eval_csv="$(latest_metrics_csv "$eval_dir")"
  else
    echo "[skip-eval] $label metrics already exist"
  fi

  if [[ -z "$eval_csv" ]]; then
    echo "Missing eval metrics for $label" >&2
    return 1
  fi

  local hr1 hr5 hr10 mrr10 ndcg10
  hr1="$(metric_from_csv "$eval_csv" "HitRate@1")"
  hr5="$(metric_from_csv "$eval_csv" "HitRate@5")"
  hr10="$(metric_from_csv "$eval_csv" "HitRate@10")"
  mrr10="$(metric_from_csv "$eval_csv" "mrr@10")"
  ndcg10="$(metric_from_csv "$eval_csv" "ndcg@10")"

  echo "$label,$temperature,$mix_weight,$hr1,$hr5,$hr10,$mrr10,$ndcg10,$eval_csv" >> "$SUMMARY_CSV"
}

cat > "$SUMMARY_CSV" <<'EOF'
label,contrastive_temperature,alpha_mix_weight,HitRate@1,HitRate@5,HitRate@10,mrr@10,ndcg@10,metrics_csv
EOF

run_candidate "alpha_t003_m04" "0.03" "0.4"
run_candidate "alpha_t003_m01" "0.03" "0.1"
run_candidate "alpha_t0025_m01" "0.025" "0.1"
run_candidate "alpha_t0025_m02" "0.025" "0.2"
run_candidate "alpha_t002_m02" "0.02" "0.2"

echo
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo
printf 'Summary saved to %s\n' "$SUMMARY_CSV"
