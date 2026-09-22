#!/usr/bin/env bash
set -euo pipefail

# Run from the remote checkout. The official Waymo script hard-codes GPU 4
# and batch 24; this 24 GB single-GPU host uses GPU 0 and batch 4.
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
cd /root/3eedqa/3EED

torchrun --standalone --nproc_per_node=1 train_dist_mod.py \
    --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root data/3eed \
    --split_dir data/3eed/splits \
    --val_freq 5 --batch_size 4 --save_freq 10 --print_freq 50 \
    --max_epoch 100 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset waymo --test_dataset waymo \
    --detect_intermediate --joint_det \
    --lr_decay_epochs 25 26 \
    --use_soft_token_loss --use_contrastive_align \
    --self_attend \
    --log_dir /root/autodl-tmp/3eed_data/grounding_repro \
    --flag full_b4_seed0
