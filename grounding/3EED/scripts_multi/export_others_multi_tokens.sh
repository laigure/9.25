#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 CHECKPOINT {train|val}" >&2
  exit 2
fi
checkpoint=$1
split=$2
if [[ $split != train && $split != val ]]; then
  echo "split must be train or val" >&2
  exit 2
fi
export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
# Exports one 288-D final-decoder query feature per referred object. Selection
# uses soft-token text-span scores with a unique assignment per target, never
# GT boxes. GT boxes in the npz are for correctness filtering only.
cd /root/3eedqa/3EED
python -m torch.distributed.launch --nproc_per_node 1 --master_port 23758 \
  train_dist_mod.py --num_decoder_layers 6 --use_color \
  --data_root data/ --batch_size 4 --num_workers 4 \
  --dataset waymo-others-multi --test_dataset waymo-others-multi \
  --detect_intermediate --joint_det \
  --use_soft_token_loss --use_contrastive_align --self_attend \
  --eval --eval_split "$split" --checkpoint_path "$checkpoint" \
  --export_qa_tokens "/root/autodl-tmp/3eed_data/multi_grounding/others_linked_tokens/$split"
