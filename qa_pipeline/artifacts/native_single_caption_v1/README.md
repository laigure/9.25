# Native single-caption multi-target dataset v1

This artifact is generated only from one released 3EED `ground_info` caption at
a time. It does not concatenate captions and contains no single-target
Grounding rows.

## Grounding annotations

- `waymo_native_multi_train_info.pkl`: 644 records, including 622 N=2 and 22 N=3.
- `waymo_native_multi_val_info.pkl`: 714 records, including 683 N=2 and 31 N=3.
- `waymo_native_multi_summary.json`: generation counts and policy.
- `waymo_native_multi_examples.json`: 20 readable full records.
- `audit.json`: raw released-data audit before non-spatial description filtering.

An `others` box is retained only if its class has one box and its noun is
introduced as another entity in the same caption. Repeated references to the
main target are excluded. Records whose retained targets cannot be assigned
distinct non-spatial descriptions are also excluded.

## Natural QA

- `qa.jsonl`: 22,680 public training/evaluation rows.
- `private_gt.jsonl`: geometry, answer keys and Grounding metadata used only to
  construct labels, auxiliary losses and evaluation metrics.
- `summary.json`: category counts and leakage audit.
- `examples.json`: one public example for every scenario.

The public rows are exactly balanced across four categories: spatial relation,
distance, ego motion and object motion, with 5,670 rows each. Questions and
answers contain natural descriptions and view names. They contain no box,
center, coordinate, `Object A/B` alias or multiple-choice answer letter.

`private_gt.jsonl` intentionally contains coordinates. Training code may read
them only in the optional auxiliary relation-loss path. They are never appended
to a Grounding token, prompt or Qwen embedding and are unnecessary at inference.

## Reproduction

Generate the files with:

```bash
python qa_pipeline/build_native_others_grounding.py \
  --data-root 3EED/data/3eed/waymo \
  --split-dir 3EED/data/splits \
  --out-dir 3EED/data

python qa_pipeline/build_native_natural_qa.py \
  --annotations 3EED/data \
  --output /root/autodl-tmp/3eed_data/multi_grounding/native_natural_qa_v1
```

The full recoverable training entry is `qa_pipeline/run_native_multi_natural_qa.sh`.
