#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/3eedqa
RUN_DIR=/root/autodl-tmp/3eed_data/multi_grounding/instance_fix_run/Train_waymo-multi_Val_waymo-multi/0921_2041
CKPT="$RUN_DIR/ckpt_epoch_last.pth"
LOG_ROOT=/root/autodl-tmp/3eed_data/multi_grounding/instance_fix_posttrain
STATUS="$LOG_ROOT/status.log"

mkdir -p "$LOG_ROOT"
printf '%s waiting for %s\n' "$(date '+%F %T')" "$CKPT" >> "$STATUS"
while [[ ! -s "$CKPT" ]]; do
  sleep 60
done

# The trainer saves ckpt_epoch_last before its own final validation. Wait for
# every worker from this run to exit so standalone evaluation cannot contend
# for GPU memory with that final validation.
printf '%s final checkpoint detected; waiting for trainer exit\n' "$(date '+%F %T')" >> "$STATUS"
while pgrep -f 'train_dist_mod.py.*instance_fix_run' >/dev/null; do
  sleep 30
done

# Give the filesystem a short settling interval, then verify that torch can
# read the checkpoint before evaluation.
sleep 30
/root/miniconda3/envs/agiclass/bin/python - "$CKPT" <<'PY' >> "$STATUS" 2>&1
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
assert "model" in checkpoint and "epoch" in checkpoint
print(f"checkpoint_ok path={path} epoch={checkpoint['epoch']}", flush=True)
PY

cd "$ROOT/3EED"
for split in val test; do
  printf '%s starting %s evaluation\n' "$(date '+%F %T')" "$split" >> "$STATUS"
  bash scripts_multi/eval_multi_official_fmt.sh "$CKPT" "$split" \
    > "$LOG_ROOT/eval_${split}.log" 2>&1
  printf '%s completed %s evaluation\n' "$(date '+%F %T')" "$split" >> "$STATUS"
done

printf '%s all grounding evaluations complete\n' "$(date '+%F %T')" >> "$STATUS"
