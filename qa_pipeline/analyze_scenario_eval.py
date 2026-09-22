"""Report row and scene-level QA metrics with train-prior baselines."""

import argparse
import collections
import json


def load_jsonl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--private-gt", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    qa = load_jsonl(args.qa)
    by_id = {x["qa_id"]: x for x in qa}
    gt = {x["qa_id"]: x["answer_key"] for x in load_jsonl(args.private_gt)}
    evaluation = json.load(open(args.eval, encoding="utf-8"))
    rows = evaluation["rows"]
    prior = collections.defaultdict(collections.Counter)
    for x in qa:
        if x["qa_split"] == "train":
            prior[x["scenario"]][gt[x["qa_id"]]] += 1
    majority = {scenario: counts.most_common(1)[0][0]
                for scenario, counts in prior.items()}
    by_scenario = collections.defaultdict(list)
    by_scene = collections.defaultdict(list)
    predictions = collections.defaultdict(collections.Counter)
    for row in rows:
        source = by_id[row["qa_id"]]
        by_scenario[row["scenario"]].append(row)
        by_scene[source["scene_id"]].append(row)
        predictions[row["scenario"]][row["pred_key"]] += 1
    n = len(rows)
    result = {
        "n_rows": n,
        "n_scenes": len(by_scene),
        "accuracy": sum(x["correct"] for x in rows) / max(n, 1),
        "majority_baseline": sum(gt[x["qa_id"]] == majority[x["scenario"]]
                                 for x in rows) / max(n, 1),
        "scene_macro_accuracy": sum(sum(x["correct"] for x in values) / len(values)
                                    for values in by_scene.values()) / max(len(by_scene), 1),
        "by_scenario": {
            scenario: {"n": len(values),
                       "accuracy": sum(x["correct"] for x in values) / len(values),
                       "majority_baseline": sum(gt[x["qa_id"]] == majority[scenario]
                                                for x in values) / len(values),
                       "prediction_distribution": dict(predictions[scenario])}
            for scenario, values in sorted(by_scenario.items())}}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
