#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/3eedqa
EED=$ROOT/3EED
PY=/root/miniconda3/envs/agiclass/bin/python
DATA=/root/autodl-tmp/3eed_data/multi_grounding
SOURCE_QA=$DATA/scene_reasoning_qa_v2_noleak
TOKENS=$DATA/scene_multi_balanced_tokens
QA_OUT=$DATA/scene_qa_v2_balanced_tokens
RUNS=$DATA/runs
PROJECTOR_OUT=$RUNS/scene_qa_v2_pairwise_projector
LORA_OUT=$RUNS/scene_qa_v2_pairwise_lora
SHUFFLED_OUT=$RUNS/scene_qa_v2_pairwise_shuffled
STAGE=$DATA/scene_qa_v2_pairwise_status

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EED:$ROOT/qa_pipeline"

mkdir -p "$STAGE" "$TOKENS" "$QA_OUT" "$RUNS"
STATUS=$STAGE/status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [BALANCED_CHECKPOINT]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  CKPT=$1
else
  CKPT=$(find "$DATA/scene_multi_balanced_run" -type f \
    -name ckpt_epoch_last.pth | sort | tail -1)
fi
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "balanced Grounding checkpoint not found" >&2
  exit 1
fi
mark "USE balanced Grounding checkpoint $CKPT"

if [[ ! -f "$STAGE/export_train.done" ]]; then
  mark "EXPORT balanced train scene tokens"
  (cd "$EED" && bash scripts_multi/eval_scene_multi.sh "$CKPT" train "$TOKENS") \
    > "$STAGE/export_train.log" 2>&1
  touch "$STAGE/export_train.done"
fi

if [[ ! -f "$STAGE/export_val.done" ]]; then
  mark "EXPORT balanced val scene tokens"
  (cd "$EED" && bash scripts_multi/eval_scene_multi.sh "$CKPT" val "$TOKENS") \
    > "$STAGE/export_val.log" 2>&1
  touch "$STAGE/export_val.done"
fi

if [[ ! -f "$STAGE/build.done" ]]; then
  mark "VERIFY balanced token exports"
  "$PY" "$ROOT/qa_pipeline/verify_multi_export.py" "$TOKENS/train" \
    > "$STAGE/verify_train.json"
  "$PY" "$ROOT/qa_pipeline/verify_multi_export.py" "$TOKENS/val" \
    > "$STAGE/verify_val.json"

  mark "BUILD no-leak QA using balanced Grounding tokens"
  "$PY" "$ROOT/qa_pipeline/build_scene_token_qa.py" \
    --exports "$TOKENS" \
    --qa "$SOURCE_QA/qa.jsonl" \
    --private-gt "$SOURCE_QA/private_gt.jsonl" \
    --output "$QA_OUT" --iou 0.25 --token-variant contrastive \
    --eligibility all --splits train,val --make-shuffled \
    > "$STAGE/build.log" 2>&1
  touch "$STAGE/build.done"
fi

COMMON=(--relation-arch pairwise --relation-layers 2 \
  --aux-relation-weight 0.2 --eval-splits val)

if [[ ! -f "$STAGE/projector.done" ]]; then
  mark "TRAIN pairwise relational projector"
  "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode projector \
    --data "$QA_OUT/joint/qa.jsonl" \
    --private-gt "$QA_OUT/joint/private_gt.jsonl" \
    --tokens-root "$QA_OUT/joint" \
    --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --out-dir "$PROJECTOR_OUT" --epochs 3 "${COMMON[@]}" \
    > "$STAGE/projector.log" 2>&1
  touch "$STAGE/projector.done"
fi

if [[ ! -f "$STAGE/lora.done" ]]; then
  mark "TRAIN LoRA initialized from pairwise projector"
  "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode lora \
    --init-projector "$PROJECTOR_OUT/best.pt" \
    --data "$QA_OUT/joint/qa.jsonl" \
    --private-gt "$QA_OUT/joint/private_gt.jsonl" \
    --tokens-root "$QA_OUT/joint" \
    --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --out-dir "$LORA_OUT" --epochs 3 "${COMMON[@]}" \
    > "$STAGE/lora.log" 2>&1
  touch "$STAGE/lora.done"
fi

if [[ ! -f "$STAGE/shuffled.done" ]]; then
  mark "EVALUATE pairwise model with same-class shuffled tokens"
  "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval --with-lora \
    --init-projector "$LORA_OUT/best.pt" \
    --data "$QA_OUT/shuffled/qa.jsonl" \
    --private-gt "$QA_OUT/shuffled/private_gt.jsonl" \
    --tokens-root "$QA_OUT/shuffled" \
    --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --out-dir "$SHUFFLED_OUT" "${COMMON[@]}" \
    > "$STAGE/shuffled.log" 2>&1
  touch "$STAGE/shuffled.done"
fi

mark "ALL_DONE"
