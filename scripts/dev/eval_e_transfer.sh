#!/usr/bin/env bash
# Transfer / OOD evaluation of the InfoNCE stabilizer-family checkpoints
# trained by scripts/dev/sweep_justatom_contrastive_v2.sh.
#
# Question: is the in-domain win of E_alpha_gate_augment (+2.49pp HR@1 vs A,
# +1.20pp vs B on `justatom`) a recipe-level improvement, or only a
# source-fit win? We probe by running the same checkpoints against datasets
# the encoder has never seen.
#
# Models under test (existing checkpoints, no retraining):
#   A_baseline             encoder-only DCL + learnable τ
#   B_simcse               + constant SimCSE 0.1
#   E_alpha_gate_augment   + per-query augment alpha-gate over SimCSE  (winner)
#
# OOD datasets (preset configs in configs/dataset/):
#   boolq-ru
#   electrical-engineering-ru
#   mmarco-russian   (large; first 10k rows by default)
#
# Outputs:
#   $OUTPUT_ROOT/<dataset>/<model_label>/  per-eval CSV from justatom.api.eval
#   $OUTPUT_ROOT/transfer_summary.csv      one row per (dataset, model) pair
#
# Usage:
#   bash scripts/dev/eval_e_transfer.sh
#   DATASETS="boolq-ru electrical-engineering-ru" bash scripts/dev/eval_e_transfer.sh
#   MODELS="A_baseline E_alpha_gate_augment" bash scripts/dev/eval_e_transfer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/thebat/minicondaenvs/envs/justatom/bin/python}"
SWEEP_ROOT="${SWEEP_ROOT:-.tmp_runs/justatom_contrastive_v2_sweep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/justatom_contrastive_v2_transfer}"
WEAVIATE_HOST="${WEAVIATE_HOST:-localhost}"
WEAVIATE_PORT="${WEAVIATE_PORT:-2211}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-32}"
DATASET_LIMIT="${DATASET_LIMIT:-10000}"   # per-dataset cap; set to 0 / unset for full

DATASETS="${DATASETS:-boolq-ru electrical-engineering-ru mmarco-russian}"
MODELS="${MODELS:-A_baseline B_simcse E_alpha_gate_augment}"

mkdir -p "$OUTPUT_ROOT"
SUMMARY_CSV="$OUTPUT_ROOT/transfer_summary.csv"
cat > "$SUMMARY_CSV" <<'EOF'
dataset,model,HitRate@1,HitRate@5,HitRate@10,mrr@10,ndcg@10,metrics_csv
EOF

metric_from_csv() {
  local csv_path="$1" metric_name="$2"
  awk -F',' -v m="$metric_name" '$1 == m { print $2; exit }' "$csv_path"
}

latest_metrics_csv() {
  local dir="$1"
  ls -1t "$dir"/*.csv 2>/dev/null | head -n 1 || true
}

checkpoint_for() {
  local label="$1"
  case "$label" in
    A_baseline|B_simcse|C_softfn) echo "$SWEEP_ROOT/$label/train/Encoder/epoch1" ;;
    D_alpha_gate_convex|E_alpha_gate_augment) echo "$SWEEP_ROOT/$label/train/BiGamma/epoch1" ;;
    *) echo "" ;;
  esac
}

run_eval() {
  local dataset_id="$1" model_label="$2"
  local ckpt
  ckpt="$(checkpoint_for "$model_label")"
  if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
    echo "[skip] missing checkpoint for $model_label at $ckpt" >&2
    return 0
  fi

  local eval_dir="$OUTPUT_ROOT/$dataset_id/$model_label"
  mkdir -p "$eval_dir"

  local eval_csv
  eval_csv="$(latest_metrics_csv "$eval_dir")"
  if [[ -z "$eval_csv" ]]; then
    echo "[eval] $dataset_id :: $model_label"
    local -a limit_args=()
    if [[ -n "$DATASET_LIMIT" && "$DATASET_LIMIT" != "0" ]]; then
      limit_args=(--dataset.limit "$DATASET_LIMIT")
    fi
    WANDB_MODE=offline "$PYTHON_BIN" -m justatom.api.eval \
      --config configs/evaluate.yaml \
      --dataset.id "$dataset_id" \
      --model.name "$ckpt" \
      --collection.name "JATOMv2Tx_${dataset_id//-/_}_${model_label}" \
      --output.save_results_to_dir "$eval_dir" \
      --search.batch_size "$EVAL_BATCH_SIZE" \
      --index.batch_size "$INDEX_BATCH_SIZE" \
      --index.flush_collection true \
      --weaviate.host "$WEAVIATE_HOST" \
      --weaviate.port "$WEAVIATE_PORT" \
      "${limit_args[@]}"
    eval_csv="$(latest_metrics_csv "$eval_dir")"
  else
    echo "[skip-eval] $dataset_id :: $model_label exists: $eval_csv"
  fi

  if [[ -z "$eval_csv" ]]; then
    echo "Missing eval metrics for $dataset_id / $model_label" >&2
    return 1
  fi

  local hr1 hr5 hr10 mrr10 ndcg10
  hr1="$(metric_from_csv "$eval_csv" "HitRate@1")"
  hr5="$(metric_from_csv "$eval_csv" "HitRate@5")"
  hr10="$(metric_from_csv "$eval_csv" "HitRate@10")"
  mrr10="$(metric_from_csv "$eval_csv" "mrr@10")"
  ndcg10="$(metric_from_csv "$eval_csv" "ndcg@10")"

  echo "$dataset_id,$model_label,$hr1,$hr5,$hr10,$mrr10,$ndcg10,$eval_csv" >> "$SUMMARY_CSV"
}

for ds in $DATASETS; do
  for m in $MODELS; do
    run_eval "$ds" "$m"
  done
done

echo
echo "=== Transfer summary ==="
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo
printf 'Saved to %s\n' "$SUMMARY_CSV"
