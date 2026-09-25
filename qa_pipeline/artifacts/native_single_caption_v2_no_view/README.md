# Native single-caption QA v2 (no public view names)

This artifact changes only the downstream QA text and prompt binding.  It keeps
the native multi-target Grounding annotations, epoch-30 Grounding checkpoint,
exported 288-dimensional tokens, object IDs and token indices unchanged.

## Public model input

- natural non-spatial object descriptions;
- the question;
- one ordered implicit Grounding token for each described object.

Public rows contain no view names, coordinates, boxes, centers, categories or
`Object A/Object B` aliases.  Source view metadata remains only in
`private_gt.jsonl` for auditing and is not read by the model prompt.

Example:

```text
Q: Use the ego vehicle's current heading as north. From the perspective of the
   pedestrian while facing north, describe where the white postal vehicle is
   located in one complete sentence.
A: The white postal vehicle is in front of the pedestrian.
```

## Files and counts

- `qa.jsonl`: 22,680 public QA rows (10,584 train; 12,096 val).
- `private_gt.jsonl`: private geometry and strict answer keys.
- `examples.json`: one example for every scenario type.
- `summary.json`: leakage audit and oracle evaluator results.
- `SHA256SUMS`: artifact checksums.

All four categories contain 5,670 rows.  Full-answer oracle evaluation is 100%,
and the audit reports zero public view text/fields, role aliases, coordinates,
and duplicate model inputs.

Generation command:

```bash
python qa_pipeline/build_native_no_view_qa.py \
  --source /root/autodl-tmp/3eed_data/multi_grounding/native_natural_qa_v1 \
  --output /root/autodl-tmp/3eed_data/multi_grounding/native_natural_qa_v2_no_view
```

The QA-only training entry point is `qa_pipeline/run_native_no_view_qa.sh`.
