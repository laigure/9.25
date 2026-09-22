#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
cd /root/3eedqa/3EED

train_pid=38466
run_root=/root/autodl-tmp/3eed_data/grounding_repro
ckpt="$run_root/Train_waymo_Val_waymo/full_b4_seed0/ckpt_epoch_last.pth"
export_dir="$run_root/final_waymo_tokens"

echo "Waiting for 3EED training PID $train_pid"
while kill -0 "$train_pid" 2>/dev/null; do
    sleep 60
done

if [ ! -s "$ckpt" ]; then
    echo "Training exited without a nonempty final checkpoint: $ckpt" >&2
    exit 1
fi

echo "Final checkpoint found: $ckpt"
torchrun --standalone --nproc_per_node=1 train_dist_mod.py \
    --num_decoder_layers 6 \
    --use_color \
    --data_root data/3eed \
    --split_dir data/3eed/splits \
    --batch_size 8 \
    --dataset waymo --test_dataset waymo \
    --detect_intermediate --joint_det \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend --eval \
    --checkpoint_path "$ckpt" \
    --export_qa_tokens "$export_dir"

export EXPORT_DIR="$export_dir"
python - <<'PY'
import glob
import json
import os
import numpy as np

files = sorted(glob.glob(os.path.join(os.environ['EXPORT_DIR'], 'rank*', '*.npz')))
if not files:
    raise RuntimeError('No final token files exported')
z = np.load(files[0])
report = {
    'first_file': files[0],
    'num_batch_files': len(files),
    'feature_name': str(z['feature_name']),
    'selection_policy': str(z['selection_policy']),
    'feature_shape': list(z['object_token'].shape),
    'pred_center_shape': list(z['pred_center'].shape),
    'pred_size_shape': list(z['pred_size'].shape),
    'query_index_shape': list(z['query_index'].shape),
}
with open(os.path.join(os.environ['EXPORT_DIR'], 'shape_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
PY
