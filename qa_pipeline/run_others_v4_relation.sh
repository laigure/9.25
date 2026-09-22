#!/usr/bin/env bash
# Rebuild the correct-grounding QA diagnostic with contrastive tokens and
# paired A/B training rows, then train relation-projector -> LoRA in stages.
set -euo pipefail

ROOT=/root/3eedqa
DATA=/root/autodl-tmp/3eed_data/multi_grounding
EXPORTS=$DATA/others_linked_tokens
QA_OUT=$DATA/qa_correct_others_v4_contrastive
PROJECTOR_OUT=$DATA/runs/others_v4_relation_projector
LORA_OUT=$DATA/runs/others_v4_relation_lora
SHUFFLED_OUT=$DATA/runs/others_v4_relation_shuffled
SWAP_OUT=$DATA/runs/others_v4_relation_swap
STAGE=$DATA/others_v4_relation_status
PY=/root/miniconda3/envs/agiclass/bin/python

export PATH=/root/miniconda3/envs/agiclass/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/3EED:$ROOT/qa_pipeline"

mkdir -p "$STAGE"
STATUS=$STAGE/status.log
mark() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

mark "BUILD contrastive QA with train A/B completion"
rm -rf "$QA_OUT"
"$PY" "$ROOT/qa_pipeline/build_correct_multi_qa.py" \
  --exports "$EXPORTS" \
  --qa "$ROOT/qa_pipeline/artifacts/qa_v3/scenario_qa/qa.jsonl" \
  --private-gt "$ROOT/qa_pipeline/artifacts/qa_v3/scenario_qa/private_gt.jsonl" \
  --neutral-bank "$ROOT/qa_pipeline/artifacts/qa_v2/neutral_bank.jsonl" \
  --output "$QA_OUT" --iou 0.25 --splits train,val \
  --token-variant contrastive --augment-train-swaps \
  > "$STAGE/build.log" 2>&1

"$PY" - "$QA_OUT" <<'PY' | tee -a "$STATUS"
import collections, json, sys
root = sys.argv[1]
summary = json.load(open(root + "/summary.json", encoding="utf-8"))
rows = [json.loads(x) for x in open(root + "/joint/qa.jsonl", encoding="utf-8")]
private = {x["qa_id"]: x for x in map(json.loads, open(root + "/joint/private_gt.jsonl", encoding="utf-8"))}
print(json.dumps(summary, ensure_ascii=False, indent=2))
for scenario in ("which_is_left", "closer_to_ego"):
    keys = collections.Counter(private[r["qa_id"]]["answer_key"] for r in rows
                               if r["qa_split"] == "train" and r["scenario"] == scenario)
    assert keys["A"] == keys["B"] and keys["A"] > 0, (scenario, keys)
    print("balanced", scenario, dict(keys))
assert summary["token_variant"] == "contrastive"
assert summary["swap_eval_rows"] > 0
PY

mark "PROJECTOR stage"
rm -rf "$PROJECTOR_OUT"
"$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode projector \
  --data "$QA_OUT/joint/qa.jsonl" \
  --private-gt "$QA_OUT/joint/private_gt.jsonl" \
  --tokens-root "$QA_OUT/joint" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$PROJECTOR_OUT" --epochs 3 --eval-splits val \
  > "$STAGE/projector.log" 2>&1

mark "LORA stage initialized from relational projector"
rm -rf "$LORA_OUT"
"$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode lora \
  --init-projector "$PROJECTOR_OUT/best.pt" \
  --data "$QA_OUT/joint/qa.jsonl" \
  --private-gt "$QA_OUT/joint/private_gt.jsonl" \
  --tokens-root "$QA_OUT/joint" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$LORA_OUT" --epochs 3 --eval-splits val \
  > "$STAGE/lora.log" 2>&1

mark "SHUFFLED donor-token evaluation"
rm -rf "$SHUFFLED_OUT"
"$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval --with-lora \
  --init-projector "$LORA_OUT/best.pt" \
  --data "$QA_OUT/shuffled/qa.jsonl" \
  --private-gt "$QA_OUT/shuffled/private_gt.jsonl" \
  --tokens-root "$QA_OUT/shuffled" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$SHUFFLED_OUT" --eval-splits val \
  > "$STAGE/shuffled.log" 2>&1

mark "A/B swapped validation consistency evaluation"
rm -rf "$SWAP_OUT"
"$PY" "$ROOT/qa_pipeline/train_scenario_qa.py" --mode eval --with-lora \
  --init-projector "$LORA_OUT/best.pt" \
  --data "$QA_OUT/swap_eval/qa.jsonl" \
  --private-gt "$QA_OUT/swap_eval/private_gt.jsonl" \
  --tokens-root "$QA_OUT/swap_eval" \
  --model-path /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --out-dir "$SWAP_OUT" --eval-splits val \
  > "$STAGE/swap.log" 2>&1

"$PY" - "$LORA_OUT" "$SHUFFLED_OUT" "$SWAP_OUT" <<'PY' | tee "$STAGE/final_summary.txt" | tee -a "$STATUS"
import json, sys
for tag, root in zip(("joint", "shuffled", "ab_swap"), sys.argv[1:]):
    result = json.load(open(root + "/eval_val.json", encoding="utf-8"))
    by = {k: round(v["accuracy"], 4) for k, v in result["by_scenario"].items()}
    print(tag, "n", result["n"], "accuracy", round(result["accuracy"], 4), "by", by)
PY

mark "ALL_DONE"
