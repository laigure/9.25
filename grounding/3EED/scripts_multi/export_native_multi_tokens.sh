#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 CHECKPOINT {train|val} [EXPORT_ROOT]" >&2
  exit 2
fi
checkpoint=$1
split=$2
export_root=${3:-/root/autodl-tmp/3eed_data/multi_grounding/native_multi_tokens}
if [[ $split != train && $split != val ]]; then
  echo "split must be train or val" >&2
  exit 2
fi

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23772 \
  train_dist_mod.py --num_decoder_layers 6 --num_target 256 \
  --text_token_budget 256 --use_color --data_root data/ \
  --batch_size 4 --num_workers 4 \
  --dataset waymo-native-multi --test_dataset waymo-native-multi \
  --detect_intermediate --joint_det --use_soft_token_loss \
  --use_contrastive_align --self_attend --eval --eval_split "$split" \
  --checkpoint_path "$checkpoint" \
  --export_qa_tokens "$export_root/$split"
