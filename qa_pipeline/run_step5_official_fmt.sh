#!/usr/bin/env bash
# Step 5: grounding-correct multi-object tokens -> projector + Qwen LoRA -> QA eval.
# Runs after scripts_multi/train_multi_official_fmt.sh finishes. One GPU, sequential.
set -euo pipefail

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

E3=/root/3eedqa/3EED
QA=/root/3eedqa/qa_pipeline
export PYTHONPATH="$E3:$QA"
DATA=/root/autodl-tmp/3eed_data/multi_grounding
CKPT="$DATA/instance_fix_run/Train_waymo-multi_Val_waymo-multi"/*/ckpt_epoch_last.pth

cd "$E3"
ckpt=$(ls -t $CKPT | head -1)
echo "=== USING CHECKPOINT $ckpt $(date -u +%FT%TZ) ==="

for split in train val test; do
  echo "=== EXPORT $split $(date -u +%FT%TZ) ==="
  bash scripts_multi/eval_multi_official_fmt.sh "$ckpt" "$split" \
    >> "$DATA/instance_fix_export_$split.log" 2>&1
done
echo "=== EXPORT DONE $(date -u +%FT%TZ) ==="

mkdir -p "$DATA/final_tokens"
for split in train val test; do
  ln -sfn "$DATA/instance_fix_tokens_$split" "$DATA/final_tokens/$split"
done

cd "$QA"
echo "=== BUILD QA $(date -u +%FT%TZ) ==="
python build_correct_multi_qa.py \
  --exports "$DATA/final_tokens" \
  --qa artifacts/qa_v3/scenario_qa/qa.jsonl \
  --private-gt artifacts/qa_v3/scenario_qa/private_gt.jsonl \
  --neutral-bank artifacts/qa_v2/neutral_bank.jsonl \
  --output "$DATA/qa_correct_instance_fix" \
  --iou 0.25 --splits train,val,test
echo "=== BUILD DONE $(date -u +%FT%TZ) ==="

echo "=== TRAIN LORA $(date -u +%FT%TZ) ==="
python train_scenario_qa.py --mode lora \
  --data "$DATA/qa_correct_instance_fix/joint/qa.jsonl" \
  --private-gt "$DATA/qa_correct_instance_fix/joint/private_gt.jsonl" \
  --tokens-root "$DATA/qa_correct_instance_fix/joint" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$DATA/runs/instance_fix_lora" \
  --epochs 3 --eval-splits val,test
echo "=== STEP5 DONE $(date -u +%FT%TZ) ==="
