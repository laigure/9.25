"""Break down full predicted-token QA by Grounding success and scenario."""

import argparse
import collections
import json


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def metric(counter):
    correct, total = counter
    return {"correct": correct, "n": total,
            "accuracy": correct / total if total else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    qa = read_jsonl(args.qa_root + "/qa.jsonl")
    private = {row["qa_id"]: row for row in
               read_jsonl(args.qa_root + "/private_gt.jsonl")}
    evaluated = json.load(open(args.eval, encoding="utf-8"))
    by_grounding = collections.defaultdict(lambda: [0, 0])
    by_iou = collections.defaultdict(lambda: [0, 0])
    pairs = {}

    for row in evaluated["rows"]:
        gt = private[row["qa_id"]]
        passed = bool(gt["diagnostic_grounding_pass"])
        correct = int(row["correct"])
        for key in (("all", passed), (row["scenario"], passed)):
            by_grounding[key][0] += correct
            by_grounding[key][1] += 1
        min_iou = min(gt["diagnostic_grounding_ious"])
        bucket = ("lt_0.10" if min_iou < 0.10 else
                  "0.10_to_0.25" if min_iou < 0.25 else
                  "0.25_to_0.50" if min_iou < 0.50 else "ge_0.50")
        by_iou[bucket][0] += correct
        by_iou[bucket][1] += 1

    for row in qa:
        key = frozenset(ref["object_id"] for ref in row["object_refs"])
        pairs.setdefault(key, bool(private[row["qa_id"]]["diagnostic_grounding_pass"]))

    majority_correct = 0
    label_distribution = {}
    for scenario in sorted({row["scenario"] for row in qa}):
        labels = collections.Counter(private[row["qa_id"]]["answer_key"]
                                     for row in qa if row["scenario"] == scenario)
        label_distribution[scenario] = dict(labels)
        majority_correct += max(labels.values())

    result = {
        "end_to_end": {key: value for key, value in evaluated.items() if key != "rows"},
        "majority_baseline": majority_correct / len(qa),
        "qa_rows": len(qa),
        "unique_grounding_pairs": len(pairs),
        "pairs_grounding_correct": sum(pairs.values()),
        "pairs_grounding_incorrect": len(pairs) - sum(pairs.values()),
        "by_grounding": {
            "{}:{}".format(scenario, "correct" if passed else "incorrect"): metric(value)
            for (scenario, passed), value in sorted(by_grounding.items(), key=str)
        },
        "by_min_target_iou": {key: metric(value) for key, value in by_iou.items()},
        "label_distribution": label_distribution,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
