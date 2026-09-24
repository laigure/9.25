#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/3eedqa
EED=$ROOT/3EED
PY=/root/miniconda3/envs/agiclass/bin/python
DATA=/root/autodl-tmp/3eed_data/multi_grounding
SOURCE_QA=$DATA/native_natural_qa_v1
TOKENS=$DATA/native_multi_tokens
QA_OUT=$DATA/native_natural_qa_tokens
RUNS=$DATA/runs
STAGE=$DATA/native_multi_natural_status
GROUND_RUN=$DATA/native_multi_run

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EED:$ROOT/qa_pipeline"

mkdir -p "$STAGE" "$TOKENS" "$QA_OUT" "$RUNS"
STATUS=$STAGE/status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

if [[ ! -f "$STAGE/data.done" ]]; then
  mark "BUILD native single-caption N=2/3 Grounding annotations"
  "$PY" "$ROOT/qa_pipeline/build_native_others_grounding.py" \
    --data-root "$EED/data/3eed/waymo" --split-dir "$EED/data/splits" \
    --out-dir "$EED/data" > "$STAGE/build_grounding_data.log" 2>&1
  mark "BUILD category-balanced natural QA without coordinates"
  "$PY" "$ROOT/qa_pipeline/build_native_natural_qa.py" \
    --annotations "$EED/data" --output "$SOURCE_QA" \
    > "$STAGE/build_qa.log" 2>&1
  (cd "$EED" && "$PY" scripts_multi/smoke_multi_dataset.py \
    --dataset waymo-native-multi) > "$STAGE/smoke_dataset.log" 2>&1
  touch "$STAGE/data.done"
fi

if [[ ! -e /dev/nvidia0 ]]; then
  mark "GPU_REQUIRED: /dev/nvidia0 is absent; data is ready, training not started"
  exit 3
fi

if [[ ! -f "$STAGE/grounding.done" ]]; then
  mark "TRAIN native single-caption multi-target Grounding from public checkpoint"
  (cd "$EED" && bash scripts_multi/train_native_multi.sh) \
    > "$STAGE/grounding.log" 2>&1
  touch "$STAGE/grounding.done"
fi

if [[ ! -f "$STAGE/select.done" ]]; then
  run_dir=$(find "$GROUND_RUN" -type f -name ckpt_epoch_last.pth \
    -printf '%h\n' | sort | tail -1)
  [[ -n "$run_dir" ]] || { echo "native Grounding run directory missing" >&2; exit 1; }
  "$PY" "$ROOT/qa_pipeline/select_native_checkpoint.py" \
    --run-dir "$run_dir" --log "$STAGE/grounding.log" \
    --selection-out "$STAGE/selected_checkpoint.json" \
    > "$STAGE/selected_checkpoint.txt"
  touch "$STAGE/select.done"
fi
CKPT=$($PY -c "import json; print(json.load(open('$STAGE/selected_checkpoint.json'))['selected_checkpoint'])")
mark "USE native Grounding checkpoint $CKPT"

for split in train val; do
  if [[ ! -f "$STAGE/export_${split}.done" ]]; then
    mark "EXPORT $split native Grounding tokens"
    (cd "$EED" && bash scripts_multi/export_native_multi_tokens.sh \
      "$CKPT" "$split" "$TOKENS") > "$STAGE/export_${split}.log" 2>&1
    "$PY" "$ROOT/qa_pipeline/verify_multi_export.py" "$TOKENS/$split" \
      > "$STAGE/verify_${split}.json"
    touch "$STAGE/export_${split}.done"
  fi
done

if [[ ! -f "$STAGE/attach.done" ]]; then
  mark "ATTACH correct and same-class shuffled implicit tokens to natural QA"
  "$PY" "$ROOT/qa_pipeline/build_scene_token_qa.py" \
    --exports "$TOKENS" --qa "$SOURCE_QA/qa.jsonl" \
    --private-gt "$SOURCE_QA/private_gt.jsonl" --output "$QA_OUT" \
    --iou 0.25 --token-variant contrastive --eligibility all \
    --splits train,val --make-shuffled > "$STAGE/attach.log" 2>&1
  touch "$STAGE/attach.done"
fi

run_arm() {
  local name=$1
  local weight=$2
  local projector=$RUNS/native_${name}_projector
  local lora=$RUNS/native_${name}_lora
  local shuffled=$RUNS/native_${name}_shuffled
  local common=(--relation-arch pairwise --relation-layers 2
    --aux-relation-weight "$weight" --eval-splits val --max-new-tokens 96)

  if [[ ! -f "$STAGE/${name}_projector.done" ]]; then
    mark "TRAIN $name Pairwise Projector, auxiliary weight $weight"
    "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode projector \
      --data "$QA_OUT/joint/qa.jsonl" \
      --private-gt "$QA_OUT/joint/private_gt.jsonl" \
      --tokens-root "$QA_OUT/joint" \
      --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --out-dir "$projector" --epochs 3 "${common[@]}" \
      > "$STAGE/${name}_projector.log" 2>&1
    touch "$STAGE/${name}_projector.done"
  fi
  if [[ ! -f "$STAGE/${name}_lora.done" ]]; then
    mark "TRAIN $name Qwen LoRA from its best Pairwise Projector"
    "$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode lora \
      --init-projector "$projector/best.pt" \
      --data "$QA_OUT/joint/qa.jsonl" \
      --private-gt "$QA_OUT/joint/private_gt.jsonl" \
      --tokens-root "$QA_OUT/joint" \
      --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
      --out-dir "$lora" --epochs 3 "${common[@]}" \
      > "$STAGE/${name}_lora.log" 2>&1
    touch "$STAGE/${name}_lora.done"
  fi
  if [[ ! -f "$STAGE/${name}_shuffled.done" ]]; then
    mark "EVALUATE $name Qwen LoRA with same-class shuffled tokens"
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
