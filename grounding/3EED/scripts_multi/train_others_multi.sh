#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23755 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --weight_decay 0.0005 --data_root data/ \
  --batch_size 12 --num_workers 4 --max_epoch 200 \
  --val_freq 5 --save_freq 25 --print_freq 25 \
  --lr_backbone 1e-4 --lr 1e-4 --lr_decay_epochs 75 80 \
  --dataset waymo-others-multi --test_dataset waymo-others-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend --augment_det \
  --log_dir /root/autodl-tmp/3eed_data/multi_grounding/others_linked_run
