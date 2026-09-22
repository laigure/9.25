"""Diagnostic linear heads on v3 implicit tokens, trained on train sequences."""

import argparse
import collections
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(row, token):
    vectors = [token[obj["token_index"]] for obj in row["object_refs"]]
    if len(vectors) == 1:
        return vectors[0]
    if len(vectors) == 2:
        a, b = vectors
        return np.concatenate((a, b, a - b))
    a, b, c = vectors
    return np.concatenate((a, b, c, a - b, a - c, b - c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="/root/autodl-tmp/3eed_data/qa_v3/scenario_qa/qa.jsonl")
    ap.add_argument("--private-gt", default="/root/autodl-tmp/3eed_data/qa_v3/scenario_qa/private_gt.jsonl")
    ap.add_argument("--tokens", default="/root/autodl-tmp/3eed_data/qa_v2/tokens/waymo_tokens.npz")
    args = ap.parse_args()
    qa = [json.loads(x) for x in open(args.qa, encoding="utf-8")]
    gt = {x["qa_id"]: x["answer_key"] for x in map(json.loads,
          open(args.private_gt, encoding="utf-8"))}
    with np.load(args.tokens) as data:
        token = data["object_token"]
    scenarios = sorted({x["scenario"] for x in qa})
    result = {}
    for scenario in scenarios:
        subset = [x for x in qa if x["scenario"] == scenario]
        arrays = {}
        for split in ("train", "val", "test"):
            rows = [x for x in subset if x["qa_split"] == split]
            arrays[split] = (np.asarray([features(x, token) for x in rows]),
                             np.asarray([gt[x["qa_id"]] for x in rows]))
        if len(set(arrays["train"][1])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1, max_iter=500, class_weight="balanced"))
        model.fit(*arrays["train"])
        result[scenario] = {split: {"n": len(arrays[split][1]),
                                    "accuracy": round(float(model.score(*arrays[split])), 4)}
                            for split in arrays}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
