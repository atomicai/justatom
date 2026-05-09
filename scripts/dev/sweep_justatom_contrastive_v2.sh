#!/usr/bin/env bash
# A/B/C/D sweep for the new InfoNCE stabilizer family on the `justatom` dataset.
#
# Variants (all share learnable_temperature=true + decoupled=true defaults):
#   A  baseline                  encoder-only InfoNCE (DCL + learnable τ)
#   B  +SimCSE                   adds dropout-view stabilizer
#   C  +SoftFN                   adds top-K mass attraction (Huynh-style)
#   D  +AlphaGate (joint γ)      α(q)·main + (1-α(q))·SimCSE under joint γ
#
# Logs go to .tmp_runs/justatom_contrastive_v2_sweep/<label>/{train,eval}.
# Eval uses the local Weaviate at localhost:2211 (override via env).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/Users/thebat/minicondaenvs/envs/justatom/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.tmp_runs/justatom_contrastive_v2_sweep}"
WEAVIATE_HOST="${WEAVIATE_HOST:-localhost}"
WEAVIATE_PORT="${WEAVIATE_PORT:-2211}"

DATASET_ID="justatom"
CONTENT_FIELD="content"
LABELS_FIELD="queries"
CHUNK_ID_COL="chunk_id"
MODEL_NAME="${MODEL_NAME:-intfloat/multilingual-e5-small}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-4992}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.05}"
SIMCSE_WEIGHT="${SIMCSE_WEIGHT:-0.1}"
SOFT_FN_WEIGHT="${SOFT_FN_WEIGHT:-0.1}"
SOFT_FN_TOPK="${SOFT_FN_TOPK:-1}"
ALPHA_MIX_WEIGHT="${ALPHA_MIX_WEIGHT:-0.3}"

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

# Common training args shared by all variants.
common_train_args() {
  cat <<EOF
--config configs/train.yaml
--dataset.id $DATASET_ID
--dataset.content_field $CONTENT_FIELD
--dataset.labels_field $LABELS_FIELD
--dataset.chunk_id_col $CHUNK_ID_COL
--model.name $MODEL_NAME
--training.loss contrastive
--training.num_samples $NUM_SAMPLES
--training.batch_size $TRAIN_BATCH_SIZE
--training.n_epochs $EPOCHS
--training.optimizer adamw
--training.contrastive_temperature $CONTRASTIVE_TEMPERATURE
--training.contrastive_learnable_temperature true
--training.contrastive_decoupled true
EOF
}

run_candidate() {
  local label="$1"; shift
  local -a extra_args=("$@")

  local train_dir="$OUTPUT_ROOT/$label/train"
  local eval_dir="$OUTPUT_ROOT/$label/eval"
  local checkpoint_dir
  if [[ " ${extra_args[*]} " == *" --training.gamma_joint true "* ]]; then
    checkpoint_dir="$train_dir/BiGamma/epoch1"
  else
    checkpoint_dir="$train_dir/Encoder/epoch1"
  fi
  local train_metrics="$train_dir/train_metrics.csv"

  mkdir -p "$train_dir" "$eval_dir"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "[train] $label"
    # shellcheck disable=SC2046
    WANDB_MODE=offline "$PYTHON_BIN" -m justatom.api.train \
      $(common_train_args) \
      "${extra_args[@]}" \
      --output.save_dir "$train_dir" \
      --output.metrics_path "$train_metrics" \
      --logging.backend csv
  else
    echo "[skip-train] $label checkpoint already exists at $checkpoint_dir"
  fi

  local eval_csv
  eval_csv="$(latest_metrics_csv "$eval_dir")"
  if [[ -z "$eval_csv" ]]; then
    echo "[eval] $label"
    WANDB_MODE=offline "$PYTHON_BIN" -m justatom.api.eval \
      --config configs/evaluate.yaml \
      --dataset.id "$DATASET_ID" \
      --dataset.content_field "$CONTENT_FIELD" \
      --dataset.labels_field "$LABELS_FIELD" \
      --dataset.chunk_id_col "$CHUNK_ID_COL" \
      --model.name "$checkpoint_dir" \
      --collection.name "JustAtomV2_${label}" \
      --output.save_results_to_dir "$eval_dir" \
      --search.batch_size "$EVAL_BATCH_SIZE" \
      --index.batch_size 32 \
      --index.flush_collection true \
      --weaviate.host "$WEAVIATE_HOST" \
      --weaviate.port "$WEAVIATE_PORT"
    eval_csv="$(latest_metrics_csv "$eval_dir")"
  else
    echo "[skip-eval] $label metrics already exist: $eval_csv"
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

  echo "$label,$hr1,$hr5,$hr10,$mrr10,$ndcg10,$eval_csv" >> "$SUMMARY_CSV"
}

cat > "$SUMMARY_CSV" <<'EOF'
label,HitRate@1,HitRate@5,HitRate@10,mrr@10,ndcg@10,metrics_csv
EOF

# A — encoder-only baseline (no SimCSE / no soft-FN / no alpha-gate).
run_candidate "A_baseline" \
  --training.freeze_encoder false \
  --training.gamma_joint false \
  --training.include_semantic_gamma false \
  --training.include_keywords_gamma false \
  --training.contrastive_simcse_dropout_weight 0.0 \
  --training.contrastive_soft_fn_attract_weight 0.0 \
  --training.contrastive_loss_alpha_gate false

# B — +SimCSE dropout view.
run_candidate "B_simcse" \
  --training.freeze_encoder false \
  --training.gamma_joint false \
  --training.include_semantic_gamma false \
  --training.include_keywords_gamma false \
  --training.contrastive_simcse_dropout_weight "$SIMCSE_WEIGHT" \
  --training.contrastive_soft_fn_attract_weight 0.0 \
  --training.contrastive_loss_alpha_gate false

# C — +Soft false-negative attraction (top-K).
run_candidate "C_softfn" \
  --training.freeze_encoder false \
  --training.gamma_joint false \
  --training.include_semantic_gamma false \
  --training.include_keywords_gamma false \
  --training.contrastive_simcse_dropout_weight 0.0 \
  --training.contrastive_soft_fn_attract_weight "$SOFT_FN_WEIGHT" \
  --training.contrastive_soft_fn_topk "$SOFT_FN_TOPK" \
  --training.contrastive_loss_alpha_gate false

# D — joint γ + alpha-gate convex mixing of main and SimCSE per query.
# NOTE: requires alpha_train_only=true so that semantic_loss (the alpha-gated InfoNCE)
# is actually included in the optimization objective. With alpha_train_only=false the
# trainer minimizes only mix_loss and the contrastive head is silently ignored.
run_candidate "D_alpha_gate_convex" \
  --training.freeze_encoder false \
  --training.gamma_joint true \
  --training.include_semantic_gamma true \
  --training.include_keywords_gamma true \
  --training.alpha_train_only true \
  --training.alpha_mix_weight "$ALPHA_MIX_WEIGHT" \
  --training.contrastive_simcse_dropout_weight "$SIMCSE_WEIGHT" \
  --training.contrastive_soft_fn_attract_weight 0.0 \
  --training.contrastive_loss_alpha_gate true \
  --training.contrastive_loss_alpha_gate_mode convex

# E — joint γ + alpha-gate "augment" mode: L = main + (1-alpha(q)) * aux.
# Per-query adaptive regularization that never weakens the main InfoNCE signal.
run_candidate "E_alpha_gate_augment" \
  --training.freeze_encoder false \
  --training.gamma_joint true \
  --training.include_semantic_gamma true \
  --training.include_keywords_gamma true \
  --training.alpha_train_only true \
  --training.alpha_mix_weight "$ALPHA_MIX_WEIGHT" \
  --training.contrastive_simcse_dropout_weight "$SIMCSE_WEIGHT" \
  --training.contrastive_soft_fn_attract_weight 0.0 \
  --training.contrastive_loss_alpha_gate true \
  --training.contrastive_loss_alpha_gate_mode augment

echo
column -s, -t "$SUMMARY_CSV" || cat "$SUMMARY_CSV"
echo
printf 'Summary saved to %s\n' "$SUMMARY_CSV"
