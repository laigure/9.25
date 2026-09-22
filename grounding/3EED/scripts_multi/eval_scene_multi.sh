#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT SPLIT" >&2
  exit 2
fi
checkpoint=$1
split=$2
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23766 \
  train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
  --text_token_budget 512 --use_color --data_root data/ \
  --batch_size 3 --num_workers 4 \
  --dataset waymo-scene-multi --test_dataset waymo-scene-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend --eval --eval_split "$split" \
  --checkpoint_path "$checkpoint" \
  --export_qa_tokens "/root/autodl-tmp/3eed_data/multi_grounding/scene_multi_tokens/$split"
