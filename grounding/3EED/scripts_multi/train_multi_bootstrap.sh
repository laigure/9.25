#!/usr/bin/env bash
set -euo pipefail

# Custom two/three-target neutral-reference bootstrap, not official 3EED multi labels.
# Run from the 3EED repository root after copying waymo_multi_*_info.pkl into data/.
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23741 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --weight_decay 0.0005 --data_root data/ \
  --batch_size 4 --num_workers 4 --max_epoch 10 \
  --val_freq 2 --save_freq 2 --print_freq 50 \
  --lr_backbone 2e-5 --lr 2e-5 --lr_decay_epochs 8 9 \
  --dataset waymo-multi --test_dataset waymo-multi \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --self_attend --augment_det \
  --init_checkpoint_path /root/autodl-tmp/3eed_data/checkpoints/ckpt_6384.pth \
  --log_dir /root/autodl-tmp/3eed_data/multi_grounding/bootstrap_run
