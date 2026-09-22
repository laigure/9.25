"""Candidate grounding recall of the text ranking, per QA split and layer.

Data property of the rich export, independent of any screening group: for
each object referenced by the QA question set, take the text-ranking top-k
candidates of a decoder layer and check whether any of them has IoU>=0.25
with the object's GT box (GT used offline only). Reports recall@1/3/5 at the
reference level for both ranking policies.

Output: screening/candidate_recall.json
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", default="/root/autodl-tmp/3eed_data/qa_v3/scenario_qa")
    parser.add_argument("--features", default="/root/autodl-tmp/3eed_data/qa_features_v1")
    parser.add_argument("--out", default="/root/autodl-tmp/3eed_data/screening/candidate_recall.json")
    args = parser.parse_args()

    index = {}
    for record in (json.loads(line) for line in
                   open(os.path.join(args.features, "records.jsonl"), encoding="utf-8")):
        index[record["object_id"]] = (record["feature_file"], record["feature_row"])

    refs_by_split = {}
    for line in open(os.path.join(args.qa_dir, "qa.jsonl"), encoding="utf-8"):
        row = json.loads(line)
        for ref in row["object_refs"]:
            refs_by_split.setdefault(row["qa_split"], set()).add(ref["object_id"])

    cache = {}
    results = {}
    for split, object_ids in refs_by_split.items():
        results[split] = {}
        object_ids = sorted(object_ids)
        for layer in range(6):
            hits_full = [0, 0, 0]
            hits_noun1 = 0
            for object_id in object_ids:
                rel, row = index[object_id]
                if rel not in cache:
                    cache[rel] = np.load(os.path.join(args.features, rel))
                data = cache[rel]
                iou = data["layer_iou"][row, layer]
                idx = data["layer_top5_idx"][row, layer]
                for k, slot in ((1, 0), (3, 1), (5, 2)):
                    hits_full[slot] += any(float(iou[i]) >= 0.25 for i in idx[:k])
                noun_idx = int(np.argmax(data["layer_scores_noun"][row, layer]))
                hits_noun1 += float(iou[noun_idx]) >= 0.25
            n = len(object_ids)
            results[split][f"layer_{layer}"] = {
                "n_objects": n,
                "recall@1_full": round(hits_full[0] / n, 4),
                "recall@3_full": round(hits_full[1] / n, 4),
                "recall@5_full": round(hits_full[2] / n, 4),
                "recall@1_noun": round(hits_noun1 / n, 4),
            }
        print(split, json.dumps(results[split]["layer_5"], ensure_ascii=False), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(json.dumps(results, indent=2))
    print("written", args.out)


if __name__ == "__main__":
    main()
