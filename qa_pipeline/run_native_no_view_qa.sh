#!/usr/bin/env bash
set -euo pipefail

# QA-only continuation.  This script deliberately does not build annotations,
# train Grounding, select a Grounding checkpoint or export new Grounding tokens.
ROOT=/root/3eedqa
EED=$ROOT/3EED
PY=/root/miniconda3/envs/agiclass/bin/python
DATA=/root/autodl-tmp/3eed_data/multi_grounding
SOURCE_V1=$DATA/native_natural_qa_v1
SOURCE_QA=$DATA/native_natural_qa_v2_no_view
TOKENS=$DATA/native_multi_tokens
QA_OUT=$DATA/native_natural_qa_tokens_v2_no_view
RUNS=$DATA/runs
STAGE=$DATA/native_no_view_qa_status

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EED:$ROOT/qa_pipeline"

mkdir -p "$STAGE" "$QA_OUT" "$RUNS"
STATUS=$STAGE/status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

[[ -f "$SOURCE_V1/qa.jsonl" && -f "$SOURCE_V1/private_gt.jsonl" ]] || {
  echo "source native QA v1 is missing: $SOURCE_V1" >&2
  exit 1
}
[[ -d "$TOKENS/train" && -d "$TOKENS/val" ]] || {
  echo "existing Grounding token exports are missing: $TOKENS" >&2
  exit 1
}

if [[ ! -f "$STAGE/data.done" ]]; then
  mark "BUILD no-view QA v2 from existing native QA; Grounding stays unchanged"
  "$PY" "$ROOT/qa_pipeline/build_native_no_view_qa.py" \
    --source "$SOURCE_V1" --output "$SOURCE_QA" \
    > "$STAGE/build_no_view_qa.log" 2>&1
  touch "$STAGE/data.done"
fi

if [[ ! -f "$STAGE/attach.done" ]]; then
  mark "ATTACH existing native Grounding tokens to no-view QA"
  "$PY" "$ROOT/qa_pipeline/build_scene_token_qa.py" \
    --exports "$TOKENS" --qa "$SOURCE_QA/qa.jsonl" \
    --private-gt "$SOURCE_QA/private_gt.jsonl" --output "$QA_OUT" \
    --iou 0.25 --token-variant contrastive --eligibility all \
    --splits train,val --make-shuffled > "$STAGE/attach.log" 2>&1
  touch "$STAGE/attach.done"
fi

GPU_LIST=$(nvidia-smi -L 2>/dev/null || true)
if [[ -z "$GPU_LIST" ]]; then
  mark "GPU_REQUIRED: no-view data and existing-token alignment are ready"
  exit 3
fi

run_arm() {
  local name=$1
  local weight=$2
  local projector=$RUNS/native_no_view_${name}_projector
  local lora=$RUNS/native_no_view_${name}_lora
  local shuffled=$RUNS/native_no_view_${name}_shuffled
  local common=(--relation-arch pairwise --relation-layers 2
    --aux-relation-weight "$weight" --eval-splits val --max-new-tokens 96)

  if [[ ! -f "$STAGE/${name}_projector.done" ]]; then
    if [[ -f "$projector/eval_val.json" ]]; then
      mark "RECOVER no-view $name Projector: evaluation exists"
    elif [[ -f "$projector/best.pt" && -f "$projector/history.json" ]]; then
      mark "RESUME no-view $name Projector validation generation"
      "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval \
        --init-projector "$projector/best.pt" \
        --data "$QA_OUT/joint/qa.jsonl" \
        --private-gt "$QA_OUT/joint/private_gt.jsonl" \
        --tokens-root "$QA_OUT/joint" \
        --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --out-dir "$projector" "${common[@]}" \
        >> "$STAGE/${name}_projector.log" 2>&1
    else
      mark "TRAIN no-view $name Pairwise Projector, auxiliary weight $weight"
      "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode projector \
        --data "$QA_OUT/joint/qa.jsonl" \
        --private-gt "$QA_OUT/joint/private_gt.jsonl" \
        --tokens-root "$QA_OUT/joint" \
        --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --out-dir "$projector" --epochs 3 "${common[@]}" \
        > "$STAGE/${name}_projector.log" 2>&1
    fi
    touch "$STAGE/${name}_projector.done"
  fi

  if [[ ! -f "$STAGE/${name}_lora.done" ]]; then
    if [[ -f "$lora/eval_val.json" ]]; then
      mark "RECOVER no-view $name LoRA: evaluation exists"
    elif [[ -f "$lora/best.pt" && -f "$lora/history.json" ]]; then
      mark "RESUME no-view $name LoRA validation generation"
      "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval --with-lora \
        --init-projector "$lora/best.pt" \
        --data "$QA_OUT/joint/qa.jsonl" \
        --private-gt "$QA_OUT/joint/private_gt.jsonl" \
        --tokens-root "$QA_OUT/joint" \
        --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --out-dir "$lora" "${common[@]}" \
        >> "$STAGE/${name}_lora.log" 2>&1
    else
      mark "TRAIN no-view $name Qwen LoRA"
      "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode lora \
        --init-projector "$projector/best.pt" \
        --data "$QA_OUT/joint/qa.jsonl" \
        --private-gt "$QA_OUT/joint/private_gt.jsonl" \
        --tokens-root "$QA_OUT/joint" \
        --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --out-dir "$lora" --epochs 3 "${common[@]}" \
        > "$STAGE/${name}_lora.log" 2>&1
    fi
    touch "$STAGE/${name}_lora.done"
  fi

  if [[ ! -f "$STAGE/${name}_shuffled.done" ]]; then
    mark "EVALUATE no-view $name LoRA with same-class shuffled tokens"
    "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval --with-lora \
      --init-projector "$lora/best.pt" \
      --data "$QA_OUT/shuffled/qa.jsonl" \
      --private-gt "$QA_OUT/shuffled/private_gt.jsonl" \
      --tokens-root "$QA_OUT/shuffled" \
      --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --out-dir "$shuffled" "${common[@]}" \
      > "$STAGE/${name}_shuffled.log" 2>&1
    touch "$STAGE/${name}_shuffled.done"
  fi
}

run_arm pairwise_noaux 0.0
run_arm pairwise_aux 0.2
mark "ALL_DONE"
touch "$STAGE/ALL_DONE"
