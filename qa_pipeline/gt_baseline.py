"""Experiment A: GT-geometry baseline over the clean QA set.

Purpose: validate the QA labels and the evaluation code before any model
training, and fix the reference number the trained models must beat.

Steps
  1. rebuild every answer from the GT relative geometry with the exact
     template build_qa.py used to write the labels;
  2. read those answers back through qa_eval.parse_answer / evaluate ->
     must be 100% exact with distance error <= the 0.05 m rounding;
  3. run a random baseline (uniform of the 4 relation combos, median train
     distance) through the same evaluator.

No model and no GPU are involved. Exits non-zero if step 2 fails, printing
the offending rows.

Run: python gt_baseline.py
"""

import json
import os
import random
import statistics

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_qa import direction_phrase  # noqa: E402
from qa_eval import evaluate  # noqa: E402

DATA = os.path.join(HERE, "data")
SPLITS = ("train", "val", "test")


def gt_answer(record):
    return ("The target object is located {}, approximately {:.1f} meters away."
            .format(direction_phrase(record["relations"]),
                    record["relative_geometry"]["distance_3d"]))


def main():
    with open(os.path.join(DATA, "qa_waymo_clean.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]

    per_split = {}
    for split in SPLITS:
        subset = [record for record in records if record["qa_split"] == split]
        if not subset:
            continue
        result = evaluate(subset, [gt_answer(record) for record in subset], tag=split)
        per_split[split] = result["metrics"]

    overall = evaluate(records, [gt_answer(record) for record in records], tag="all")
    metrics = overall["metrics"]

    # Step 2 check: labels and evaluator must agree perfectly.
    failures = [row for row in overall["rows"] if not row["exact_ok"]]
    checks = {
        "exact_accuracy_is_1": metrics["exact_accuracy"] == 1.0,
        "format_validity_is_1": metrics["format_validity"] == 1.0,
        "distance_mae_within_rounding": metrics["distance"]["mae"] <= 0.051,
        "n_failures": len(failures),
    }
    if failures:
        print("FAILED rows (first 10):")
        for row in failures[:10]:
            print(json.dumps(row, ensure_ascii=False, indent=2))

    # Step 3: random baseline over the same records.
    rng = random.Random(0)
    train_distances = [record["relative_geometry"]["distance_3d"]
                       for record in records if record["qa_split"] == "train"]
    median_distance = statistics.median(train_distances)
    random_predictions = []
    for _ in records:
        lateral = rng.choice(["left", "right"])
        longitudinal = rng.choice(["front", "behind"])
        random_predictions.append(
            "The target object is located to the {} and {} the reference object, "
            "approximately {:.1f} meters away.".format(
                lateral, "in front of" if longitudinal == "front" else "behind",
                median_distance))
    random_result = evaluate(records, random_predictions, tag="random_baseline")

    samples = []
    rng_sample = random.Random(7)
    for index in rng_sample.sample(range(len(records)), 20):
        record = records[index]
        row = overall["rows"][index]
        samples.append({
            "qa_id": record["qa_id"],
            "question": record["question"],
            "gt_answer": gt_answer(record),
            "parsed": {"lateral": row["pred_lateral"],
                       "longitudinal": row["pred_longitudinal"],
                       "distance": row["pred_distance"]},
            "exact_ok": row["exact_ok"],
        })

    report = {
        "experiment": "A_gt_baseline",
        "data": "qa_waymo_clean.jsonl",
        "n_records": len(records),
        "metrics_all": metrics,
        "per_split": per_split,
        "random_baseline": random_result["metrics"],
        "checks": checks,
        "random_baseline_definition": (
            "uniform of the 4 relation combos + median train distance, "
            "same evaluator as the models"),
    }
    with open(os.path.join(DATA, "experiment_a_gt_baseline.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA, "experiment_a_samples.txt"), "w",
              encoding="utf-8") as handle:
        for sample in samples:
            handle.write("Q: {}\n".format(sample["question"]))
            handle.write("GT answer: {}\n".format(sample["gt_answer"]))
            handle.write("parsed: {} exact_ok={}\n\n".format(
                sample["parsed"], sample["exact_ok"]))

    print(json.dumps({"metrics_all": metrics,
                      "random_baseline": random_result["metrics"],
                      "checks": checks}, indent=2, ensure_ascii=False))
    for sample in samples[:5]:
        print("Q:", sample["question"])
        print("A:", sample["gt_answer"])
        print()

    if not all((checks["exact_accuracy_is_1"], checks["format_validity_is_1"],
                checks["distance_mae_within_rounding"])):
        raise SystemExit("Experiment A check failed: labels and evaluator disagree")
    print("Experiment A passed: labels and evaluator agree (exact=1.0, mae<=0.05)")


if __name__ == "__main__":
    main()
