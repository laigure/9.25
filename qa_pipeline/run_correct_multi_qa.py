"""Evaluate the same correctly grounded QA questions with three token banks."""

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from train_scenario_qa import ScenarioTrainer, evaluate, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--original-tokens-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gen-batch", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        model_path=args.model_path, mode="eval", with_lora=True,
        init_projector=args.checkpoint, geom="none", no_tokens=False,
        pred_geometry=None, oracle_center=False, tokens_root=args.original_tokens_root,
        data=str(args.dataset / "original/qa.jsonl"), gen_batch=args.gen_batch,
        batch_size=2,
    )
    trainer = ScenarioTrainer(config)
    outputs = {}
    expected_ids = None
    for variant in ("original", "joint", "shuffled"):
        folder = args.dataset / variant
        records = read_jsonl(folder / "qa.jsonl")
        ids = [row["qa_id"] for row in records]
        if expected_ids is None:
            expected_ids = ids
        assert ids == expected_ids, "Variants must use exactly the same questions in the same order"
        private = {row["qa_id"]: row for row in read_jsonl(folder / "private_gt.jsonl")}
        trainer.tokens_root = str(args.original_tokens_root if variant == "original" else folder)
        trainer._tokens_cache.clear()
        trainer.check_token_identity(records)
        predictions = trainer.generate(records)
        result = evaluate(records, predictions, private)
        outputs[variant] = result
        (args.out / f"eval_{variant}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(variant, "n", result["n"], "accuracy", result["accuracy"],
              "by_scenario", result["by_scenario"], flush=True)
    original = {row["qa_id"]: row for row in outputs["original"]["rows"]}
    joint = {row["qa_id"]: row for row in outputs["joint"]["rows"]}
    shuffled = {row["qa_id"]: row for row in outputs["shuffled"]["rows"]}
    changed = Counter()
    for qa_id in expected_ids:
        changed["joint_vs_original_prediction_changed"] += (
            joint[qa_id]["prediction"] != original[qa_id]["prediction"])
        changed["shuffle_vs_joint_prediction_changed"] += (
            shuffled[qa_id]["prediction"] != joint[qa_id]["prediction"])
        changed["joint_correct_shuffle_wrong"] += (
            joint[qa_id]["correct"] and not shuffled[qa_id]["correct"])
        changed["joint_wrong_shuffle_correct"] += (
            not joint[qa_id]["correct"] and shuffled[qa_id]["correct"])
    summary = {"n": len(expected_ids),
               "accuracy": {name: result["accuracy"] for name, result in outputs.items()},
               "paired": dict(changed)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
