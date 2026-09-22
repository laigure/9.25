#!/bin/bash
# Sequential Qwen LoRA runs for the selected token groups (T1 baseline, T6,
# T9). One group at a time, independent run dirs, identical questions and
# hyper-parameters (v3 settings). Per group:
#   1) train (in-training val monitor = v3's 50/type cap, for best-epoch pick)
#   2) full val+test eval, normal                -> eval_{split}.json   (main scores)
#   3) full val+test eval, A/B swap on multi-ref rows -> eval_{split}_swap.json
#   4) full val+test eval, shuffled-donor features, multi-ref rows
#                                                  -> eval_{split}_shuffle42.json
# Logs to /root/autodl-tmp/3eed_data/qa_v3/group_runs.log with stage markers.
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export TOKENIZERS_PARALLELISM=false
cd /root/3eedqa/qa_pipeline || exit 1
RUNS=/root/autodl-tmp/3eed_data/qa_v3/runs
for GROUP in T1 T6 T9; do
  OUT=$RUNS/group_$GROUP
  mkdir -p "$OUT"
  echo "=== GROUP $GROUP TRAIN $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python train_group_qa.py --group "$GROUP" --mode lora --out-dir "$OUT" \
    || { echo "GROUP $GROUP TRAIN FAILED"; exit 1; }
  echo "=== GROUP $GROUP EVAL_FULL $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python train_group_qa.py --group "$GROUP" --mode eval --init-projector "$OUT/best.pt" \
    --eval-per-scenario 0 --eval-splits val,test --out-dir "$OUT" \
    || { echo "GROUP $GROUP EVAL_FULL FAILED"; exit 1; }
  echo "=== GROUP $GROUP EVAL_SWAP $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python train_group_qa.py --group "$GROUP" --mode eval --init-projector "$OUT/best.pt" \
    --eval-per-scenario 0 --only-multi-ref --swap --eval-splits val,test --out-dir "$OUT" \
    || { echo "GROUP $GROUP EVAL_SWAP FAILED"; exit 1; }
  echo "=== GROUP $GROUP EVAL_SHUFFLE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python train_group_qa.py --group "$GROUP" --mode eval --init-projector "$OUT/best.pt" \
    --eval-per-scenario 0 --only-multi-ref --shuffle-seed 42 --eval-splits val,test \
    --out-dir "$OUT" \
    || { echo "GROUP $GROUP EVAL_SHUFFLE FAILED"; exit 1; }
  echo "=== GROUP $GROUP DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done
echo GROUP_RUNS_ALL_DONE
