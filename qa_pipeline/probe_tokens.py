"""Diagnostic linear probes for whether neutral Grounding tokens encode geometry."""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/3eed_data/qa_v2")
    args = ap.parse_args()
    qa = [json.loads(x) for x in open(os.path.join(args.base, "scenario_qa", "qa.jsonl"))]
    gt = {x["qa_id"]: x["answer_key"] for x in map(
        json.loads, open(os.path.join(args.base, "scenario_qa", "private_gt.jsonl")))}
    with np.load(os.path.join(args.base, "tokens", "waymo_tokens.npz")) as data:
        token = data["object_token"]
    output = {}
    for scenario in ("ego_location", "relative_location", "closer_of_two"):
        arrays = {}
        for split in ("train", "val", "test"):
            rows = [r for r in qa if r["qa_split"] == split and r["scenario"] == scenario]
            x, y = [], []
            for row in rows:
                vec = [token[obj["token_index"]] for obj in row["object_refs"]]
                if scenario == "ego_location":
                    feature = vec[0]
                elif scenario == "relative_location":
                    feature = np.concatenate((vec[0], vec[1], vec[0] - vec[1]))
                else:
                    feature = np.concatenate((vec[0] - vec[2], vec[1] - vec[2],
                                              vec[0] - vec[1]))
                x.append(feature)
                y.append(gt[row["qa_id"]])
            arrays[split] = (np.asarray(x), np.asarray(y))
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1, max_iter=500, class_weight="balanced"))
        model.fit(*arrays["train"])
        output[scenario] = {split: {"n": len(arrays[split][1]),
                                    "accuracy": float(model.score(*arrays[split]))}
                            for split in arrays}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
