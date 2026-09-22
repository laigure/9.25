#!/usr/bin/env bash
set -euo pipefail

# Step 2 smoke on the reconstructed official-format annotations
# (qa_pipeline/artifacts/multi_grounding_official_fmt/waymo_multi_{train,val}_info.pkl).
# Same recipe as the official scripts_multi/train_multi_3eed.sh -- from scratch,
# all modules lr 1e-4 -- but under --debug: the dataset overfits annos[:128] and
# each epoch stops after 11 batches, i.e. batch 4 x 11 = 44 samples per epoch.
# Purpose: watch the training loss go down and later export per-target query
# indices to confirm different targets land on different queries.
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23743 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --weight_decay 0.0005 --data_root data/ \
  --batch_size 4 --num_workers 0 --max_epoch 20 \
  --val_freq 5 --save_freq 20 --print_freq 5 \
  --lr_backbone 1e-4 --lr 1e-4 \
  --dataset waymo-multi --test_dataset waymo-multi \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --self_attend --augment_det \
  --debug \
  --log_dir /root/autodl-tmp/3eed_data/multi_grounding/coordinate_fix_smoke
