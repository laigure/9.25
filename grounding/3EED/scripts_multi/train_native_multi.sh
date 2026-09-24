#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO

BASE=/root/autodl-tmp/3eed_data/multi_grounding
INIT=/root/autodl-tmp/3eed_data/checkpoints/ckpt_6384.pth
OUT=$BASE/native_multi_run

# Every row is one released caption and its unambiguously mentioned native
# targets. N=1 is absent. N=3 is cycled to counter its smaller source count.
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23771 \
  train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
  --text_token_budget 256 --use_color --weight_decay 0.0005 \
  --data_root data/ --batch_size 8 --num_workers 4 --max_epoch 30 \
  --val_freq 5 --save_freq 15 --print_freq 25 \
  --lr_backbone 2e-5 --lr 5e-5 --text_encoder_lr 5e-6 \
  --lr_decay_epochs 20 27 \
  --init_checkpoint_path "$INIT" \
  --balance_target_counts --target_count_samples 0 600 300 0 0 \
  --dataset waymo-native-multi --test_dataset waymo-native-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend --augment_det \
  --log_dir "$OUT"

# The epoch-30 and last files contain the same final state. Keep the canonical
# last file and epoch 15 after both have been safely written.
run_dir=$(find "$OUT" -type f -name ckpt_epoch_last.pth -printf '%h\n' | sort | tail -1)
if [[ -n "$run_dir" && -f "$run_dir/ckpt_epoch_30.pth" ]]; then
  rm -f "$run_dir/ckpt_epoch_30.pth"
fi
