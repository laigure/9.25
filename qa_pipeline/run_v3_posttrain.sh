#!/usr/bin/env bash
set -euo pipefail

base=/root/autodl-tmp/3eed_data/qa_v3
features=/root/autodl-tmp/3eed_data/qa_v2
trainer=/root/3eedqa/qa_pipeline/train_scenario_qa.py
checkpoint="$base/runs/full_lora/best.pt"

while kill -0 27868 2>/dev/null; do
  sleep 20
done

test -s "$checkpoint"
mkdir -p "$base/runs/full_eval" "$base/runs/shuffle_probe"

echo "BEGIN_FULL_EVAL $(date -u +%FT%TZ)"
python "$trainer" --mode eval --with-lora \
  --init-projector "$checkpoint" \
  --data "$base/scenario_qa/qa.jsonl" \
  --private-gt "$base/scenario_qa/private_gt.jsonl" \
  --tokens-root "$features" \
  --out-dir "$base/runs/full_eval" \
  --eval-splits val,test
echo "END_FULL_EVAL $(date -u +%FT%TZ)"

echo "BEGIN_SHUFFLE_EVAL $(date -u +%FT%TZ)"
python "$trainer" --mode eval --with-lora \
  --init-projector "$checkpoint" \
  --data "$base/scenario_qa/qa.jsonl" \
  --private-gt "$base/scenario_qa/private_gt.jsonl" \
  --tokens-root "$features/shuffled" \
  --out-dir "$base/runs/shuffle_probe" \
  --eval-per-scenario 50 --eval-splits val
echo "END_SHUFFLE_EVAL $(date -u +%FT%TZ)"
