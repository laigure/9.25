#!/usr/bin/env bash
set -euo pipefail

# Step 3: official multi-object recipe from 3EED scripts_multi/train_multi_3eed.sh
# (from scratch -- no init checkpoint, batch 12, lr 1e-4, 200 epochs, lr decay at
# 75/80) on the reconstructed official-format annotations. val_freq is 5 instead
# of the official 1 to keep the run within one night; report Acc@0.25 / Acc@0.5 /
# mIoU per class with scripts_multi/eval_multi_official_fmt.sh afterwards.
# Checkpoints are saved every 25 epochs to fit the 50 GB data disk.
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23744 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --weight_decay 0.0005 --data_root data/ \
  --batch_size 12 --num_workers 4 --max_epoch 200 \
  --val_freq 5 --save_freq 25 --print_freq 25 \
  --lr_backbone 1e-4 --lr 1e-4 --lr_decay_epochs 75 80 \
  --dataset waymo-multi --test_dataset waymo-multi \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --self_attend --augment_det \
  --log_dir /root/autodl-tmp/3eed_data/multi_grounding/instance_fix_run
