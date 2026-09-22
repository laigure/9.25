#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi
checkpoint=$1
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23756 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --data_root data/ --batch_size 4 --num_workers 4 \
  --dataset waymo-others-multi --test_dataset waymo-others-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend \
  --eval --eval_split val --checkpoint_path "$checkpoint" \
  --export_qa_tokens /root/autodl-tmp/3eed_data/multi_grounding/others_linked_tokens_val
