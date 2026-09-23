#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

BASE=/root/autodl-tmp/3eed_data/multi_grounding
INIT=$BASE/scene_multi_run/Train_waymo-scene-multi_Val_waymo-scene-multi/0922_1716/ckpt_epoch_100.pth
OUT=$BASE/scene_multi_balanced_run

python -m torch.distributed.launch --nproc_per_node 1 --master_port 23767 \
  train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
  --text_token_budget 512 --use_color --weight_decay 0.0005 \
  --data_root data/ --batch_size 8 --num_workers 4 --max_epoch 30 \
  --val_freq 5 --save_freq 15 --print_freq 25 \
  --lr_backbone 2e-5 --lr 5e-5 --text_encoder_lr 5e-6 \
  --lr_decay_epochs 20 27 \
  --init_checkpoint_path "$INIT" \
  --balance_target_counts --target_count_samples 900 800 600 300 0 \
  --dataset waymo-scene-multi --test_dataset waymo-scene-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend --augment_det \
  --log_dir "$OUT"

# Epoch 30 and last contain the same final model state. Keep the canonical
# last checkpoint plus epoch 15 to stay within the current data-disk budget.
run_dir=$(find "$OUT" -mindepth 2 -maxdepth 2 -type d -name '09*' | sort | tail -1)
if [[ -n "$run_dir" && -f "$run_dir/ckpt_epoch_30.pth" && -f "$run_dir/ckpt_epoch_last.pth" ]]; then
  rm -f "$run_dir/ckpt_epoch_30.pth"
fi
