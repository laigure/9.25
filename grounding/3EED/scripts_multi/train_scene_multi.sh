#!/usr/bin/env bash
set -euo pipefail
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23765 \
  train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
  --text_token_budget 512 --use_color --weight_decay 0.0005 \
  --data_root data/ --batch_size 8 --num_workers 4 --max_epoch 100 \
  --val_freq 10 --save_freq 100 --print_freq 25 \
  --lr_backbone 5e-5 --lr 1e-4 --lr_decay_epochs 75 90 \
  --init_checkpoint_path /root/autodl-tmp/3eed_data/multi_grounding/others_linked_run/Train_waymo-others-multi_Val_waymo-others-multi/0921_2210/ckpt_epoch_200.pth \
  --allow_partial_init \
  --dataset waymo-scene-multi --test_dataset waymo-scene-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend --augment_det \
  --log_dir /root/autodl-tmp/3eed_data/multi_grounding/scene_multi_run
