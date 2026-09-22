#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT {train|val|test}" >&2
  exit 2
fi
checkpoint=$1
split=$2
if [[ $split != train && $split != val && $split != test ]]; then
  echo "split must be train, val or test" >&2
  exit 2
fi
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
# Reports Acc@0.25 / Acc@0.5 (per target and joint) plus mIoU overall and per
# class, and exports the selected 288-D tokens with their query indices.
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23745 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --data_root data/ --batch_size 4 --num_workers 4 \
  --dataset waymo-multi --test_dataset waymo-multi \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align --self_attend \
  --eval --eval_split "$split" --checkpoint_path "$checkpoint" \
  --export_qa_tokens "/root/autodl-tmp/3eed_data/multi_grounding/instance_fix_tokens_$split"
