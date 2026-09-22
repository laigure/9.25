#!/usr/bin/env bash
set -euo pipefail

# Resume of the official multi-object recipe. The first attempt stopped at epoch
# 130 because the data disk filled up while saving the checkpoint (Qwen download
# ran concurrently); no code or config changed. --checkpoint_path restores model,
# optimizer and scheduler state, so start_epoch continues at 126 with the same
# lr schedule (1e-6 after the 75/80 decay) and the run finishes at epoch 200.
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
CKPT_125=/root/autodl-tmp/3eed_data/multi_grounding/official_fmt_run/Train_waymo-multi_Val_waymo-multi/0921_1624/ckpt_epoch_125.pth
test -s "$CKPT_125"
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23744 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --weight_decay 0.0005 --data_root data/ \
  --batch_size 12 --num_workers 4 --max_epoch 200 \
  --val_freq 5 --save_freq 5 --print_freq 25 \
  --lr_backbone 1e-4 --lr 1e-4 --lr_decay_epochs 75 80 \
  --dataset waymo-multi --test_dataset waymo-multi \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align \
  --self_attend --augment_det \
  --checkpoint_path "$CKPT_125" \
  --log_dir /root/autodl-tmp/3eed_data/multi_grounding/official_fmt_resume
