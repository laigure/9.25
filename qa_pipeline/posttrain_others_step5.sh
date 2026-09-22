#!/usr/bin/env bash
# Overnight chain after the others-multi grounding run finishes:
#   wait ckpt -> export train/val tokens -> build correct-multi QA banks
#   -> projector + Qwen LoRA (joint) -> val eval joint -> val eval shuffled -> done.
# One GPU, sequential. Writes $STAGE/state and $STAGE/status.log.
set -uo pipefail

ROOT=/root/3eedqa
DATA=/root/autodl-tmp/3eed_data/multi_grounding
RUN_GLOB="$DATA/others_linked_run/Train_waymo-others-multi_Val_waymo-others-multi/*"
TOKENS=$DATA/others_linked_tokens
QA_OUT=$DATA/qa_correct_others
LORA_OUT=$DATA/runs/others_lora
SHUF_OUT=$DATA/runs/others_lora_shuffled
STAGE=$DATA/others_linked_posttrain
STATE=$STAGE/state
STATUS=$STAGE/status.log

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/3EED:$ROOT/qa_pipeline"

mkdir -p "$STAGE" "$LORA_OUT" "$SHUF_OUT"
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
stage() { echo "$1" > "$STATE"; mark "stage=$1"; }
abort() { stage ABORT; mark "ABORT: $*"; exit 1; }
free_gb() { df -Pk "$DATA" | awk 'NR==2{printf "%d", $4/1048576}'; }

stage WAITING_CKPT
deadline=$(( $(date +%s) + 8*3600 ))
while true; do
  ckpt=$(ls -t $RUN_GLOB/ckpt_epoch_200.pth $RUN_GLOB/ckpt_epoch_last.pth 2>/dev/null | head -1)
  [[ -n $ckpt ]] && break
  [[ $(date +%s) -gt $deadline ]] && abort "no epoch-200/last checkpoint within 8h under $RUN_GLOB"
  sleep 60
done
mark "checkpoint detected: $ckpt"

stage WAITING_EXIT
deadline=$(( $(date +%s) + 5400 ))
while pgrep -f "train_dist_mod.py.*others_linked_run" >/dev/null; do
  [[ $(date +%s) -gt $deadline ]] && break
  sleep 30
done
# The trainer's own final validation also occupies the GPU; wait for real
# memory release instead of trusting the process list alone.
deadline=$(( $(date +%s) + 5400 ))
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -gt 10000 ]]; do
  [[ $(date +%s) -gt $deadline ]] && abort "GPU still occupied (>10G) 90 min after final checkpoint"
  sleep 30
done
mark "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB used)"
sleep 30

/root/miniconda3/envs/agiclass/bin/python - "$ckpt" <<'PY' >> "$STATUS" 2>&1 || abort "checkpoint unreadable"
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu")
assert "model" in checkpoint and "epoch" in checkpoint
print("checkpoint_ok epoch={} tensors={}".format(checkpoint["epoch"], len(checkpoint["model"])), flush=True)
PY

stage EXPORT_TRAIN
[[ $(free_gb) -lt 2 ]] && abort "disk below 2G before export"
timeout 7200 bash "$ROOT/3EED/scripts_multi/export_others_multi_tokens.sh" "$ckpt" train \
  >> "$STAGE/export_train.log" 2>&1 || abort "train export failed or timed out"

stage EXPORT_VAL
timeout 7200 bash "$ROOT/3EED/scripts_multi/export_others_multi_tokens.sh" "$ckpt" val \
  >> "$STAGE/export_val.log" 2>&1 || abort "val export failed or timed out"

/root/miniconda3/envs/agiclass/bin/python - "$TOKENS" <<'PY' >> "$STATUS" 2>&1 || abort "export sample count mismatch"
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
for split, expected in (("train", 1137), ("val", 1300)):
    files = sorted((root / split).rglob("*.npz"))
    total = 0
    for path in files:
        with np.load(path) as data:
            total += len(data["target_count"])
    print(f"{split}: files={len(files)} samples={total} expected={expected}", flush=True)
    assert files and total == expected, (split, total)
PY

stage BUILD_QA
timeout 3600 /root/miniconda3/envs/agiclass/bin/python "$ROOT/qa_pipeline/build_correct_multi_qa.py" \
  --exports "$TOKENS" \
  --qa "$ROOT/qa_pipeline/artifacts/qa_v3/scenario_qa/qa.jsonl" \
  --private-gt "$ROOT/qa_pipeline/artifacts/qa_v3/scenario_qa/private_gt.jsonl" \
  --neutral-bank "$ROOT/qa_pipeline/artifacts/qa_v2/neutral_bank.jsonl" \
  --output "$QA_OUT" --iou 0.25 --splits train,val \
  >> "$STAGE/build_qa.log" 2>&1 || abort "QA build failed or timed out"

/root/miniconda3/envs/agiclass/bin/python - "$QA_OUT" <<'PY' >> "$STATUS" 2>&1 || abort "QA yield below guard"
import json
import sys
summary = json.load(open(sys.argv[1] + "/summary.json", encoding="utf-8"))
matched = summary["qa_matched_after_donor"]
print("qa_matched_after_donor", matched, flush=True)
assert matched.get("val", 0) >= 150, matched
assert matched.get("train", 0) >= 300, matched
PY

stage LORA
[[ $(free_gb) -lt 1 ]] && abort "disk below 1G before LoRA"
timeout 36000 /root/miniconda3/envs/agiclass/bin/python "$ROOT/qa_pipeline/train_scenario_qa.py" --mode lora \
  --data "$QA_OUT/joint/qa.jsonl" \
  --private-gt "$QA_OUT/joint/private_gt.jsonl" \
  --tokens-root "$QA_OUT/joint" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$LORA_OUT" --epochs 3 --eval-splits val \
  >> "$STAGE/lora.log" 2>&1 || abort "LoRA training failed or timed out"

stage SHUFFLED_EVAL
timeout 10800 /root/miniconda3/envs/agiclass/bin/python "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval \
  --init-projector "$LORA_OUT/best.pt" --with-lora \
  --data "$QA_OUT/shuffled/qa.jsonl" \
  --private-gt "$QA_OUT/shuffled/private_gt.jsonl" \
  --tokens-root "$QA_OUT/shuffled" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$SHUF_OUT" --eval-splits val \
  >> "$STAGE/shuffled_eval.log" 2>&1 || mark "WARN: shuffled eval failed (non-fatal, chain continues)"

/root/miniconda3/envs/agiclass/bin/python - "$LORA_OUT" "$SHUF_OUT" "$STAGE/final_summary.txt" <<'PY' >> "$STATUS" 2>&1
import json
import sys
lines = []
for tag, path in (("joint", sys.argv[1]), ("shuffled", sys.argv[2])):
    try:
        result = json.load(open(path + "/eval_val.json", encoding="utf-8"))
        lines.append("{} n={} accuracy={:.4f} by_scenario={}".format(
            tag, result["n"], result["accuracy"],
            {k: round(v["accuracy"], 3) for k, v in result["by_scenario"].items()}))
    except Exception as exc:
        lines.append("{} unavailable: {}".format(tag, exc))
open(sys.argv[3], "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
PY

stage ALL_DONE
mark "chain complete; see $STAGE/final_summary.txt"
