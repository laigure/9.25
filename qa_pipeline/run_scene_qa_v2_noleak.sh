#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/3eedqa
EED=$ROOT/3EED
PY=/root/miniconda3/envs/agiclass/bin/python
DATA=/root/autodl-tmp/3eed_data/multi_grounding
CKPT=$DATA/scene_multi_run/Train_waymo-scene-multi_Val_waymo-scene-multi/0922_1716/ckpt_epoch_100.pth
SOURCE_QA=$DATA/scene_reasoning_qa_v2_noleak
TOKENS=$DATA/scene_multi_tokens
QA_OUT=$DATA/scene_qa_v2_tokens
RUNS=$DATA/runs
PROJECTOR_OUT=$RUNS/scene_qa_v2_projector_swapfix
LORA_OUT=$RUNS/scene_qa_v2_lora_swapfix
SHUFFLED_OUT=$RUNS/scene_qa_v2_shuffled_swapfix
STAGE=$DATA/scene_qa_v2_status

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EED:$ROOT/qa_pipeline"

mkdir -p "$STAGE" "$TOKENS" "$QA_OUT" "$RUNS"
STATUS=$STAGE/status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

if [[ ! -f "$STAGE/export_train.done" ]]; then
  mark "EXPORT train scene tokens"
  (cd "$EED" && bash scripts_multi/eval_scene_multi.sh "$CKPT" train) \
    > "$STAGE/export_train.log" 2>&1
  touch "$STAGE/export_train.done"
fi

if [[ ! -f "$STAGE/export_val.done" ]]; then
  mark "EXPORT val scene tokens"
  (cd "$EED" && bash scripts_multi/eval_scene_multi.sh "$CKPT" val) \
    > "$STAGE/export_val.log" 2>&1
  touch "$STAGE/export_val.done"
fi

if [[ ! -f "$STAGE/build.done" ]]; then
  mark "VERIFY token exports"
  "$PY" "$ROOT/qa_pipeline/verify_multi_export.py" "$TOKENS/train" \
    > "$STAGE/verify_train.json"
  "$PY" "$ROOT/qa_pipeline/verify_multi_export.py" "$TOKENS/val" \
    > "$STAGE/verify_val.json"

  mark "BUILD no-leak token QA and same-class shuffled control"
  "$PY" "$ROOT/qa_pipeline/build_scene_token_qa.py" \
    --exports "$TOKENS" \
    --qa "$SOURCE_QA/qa.jsonl" \
    --private-gt "$SOURCE_QA/private_gt.jsonl" \
    --output "$QA_OUT" --iou 0.25 --token-variant contrastive \
    --eligibility all --splits train,val --make-shuffled \
    > "$STAGE/build.log" 2>&1
  touch "$STAGE/build.done"
fi

if [[ ! -f "$STAGE/projector.done" ]]; then
  mark "TRAIN relational projector"
  "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode projector \
    --data "$QA_OUT/joint/qa.jsonl" \
    --private-gt "$QA_OUT/joint/private_gt.jsonl" \
    --tokens-root "$QA_OUT/joint" \
    --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --out-dir "$PROJECTOR_OUT" --epochs 3 --eval-splits val \
    > "$STAGE/projector.log" 2>&1
  touch "$STAGE/projector.done"
fi

if [[ ! -f "$STAGE/lora.done" ]]; then
  mark "TRAIN LoRA initialized from best projector"
  "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode lora \
    --init-projector "$PROJECTOR_OUT/best.pt" \
    --data "$QA_OUT/joint/qa.jsonl" \
    --private-gt "$QA_OUT/joint/private_gt.jsonl" \
    --tokens-root "$QA_OUT/joint" \
    --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --out-dir "$LORA_OUT" --epochs 3 --eval-splits val \
    > "$STAGE/lora.log" 2>&1
  touch "$STAGE/lora.done"
fi

if [[ ! -f "$STAGE/shuffled.done" ]]; then
  mark "EVALUATE same-class shuffled tokens"
  "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval --with-lora \
    --init-projector "$LORA_OUT/best.pt" \
    --data "$QA_OUT/shuffled/qa.jsonl" \
    --private-gt "$QA_OUT/shuffled/private_gt.jsonl" \
    --tokens-root "$QA_OUT/shuffled" \
    --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --out-dir "$SHUFFLED_OUT" --eval-splits val \
    > "$STAGE/shuffled.log" 2>&1
  touch "$STAGE/shuffled.done"
fi

mark "ALL_DONE"
