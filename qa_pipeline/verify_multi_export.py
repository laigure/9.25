"""Verify every exported multi-object token keeps its role and query mapping."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files = list(args.directory.rglob("*.npz"))
    if not files:
        raise SystemExit("No NPZ files")
    samples = targets = 0
    count_distribution = {}
    policies = set()
    for path in files:
        with np.load(path) as data:
            policy = data["selection_policy"].item()
            policies.add(policy)
            assert policy in {
                "soft_token_unique_assignment_text_spans",
                "soft_token_unique_assignment_normalized_text_spans",
                "soft_token_top1_text_span",
            }
            for row, count in enumerate(data["target_count"].tolist()):
                ids = json.loads(data["object_ids_json"][row])
                count = int(count)
                assert count == len(ids) and 1 <= count <= 5, (path, row)
                token = data["object_token"][row]
                center = data["pred_center"][row]
                indices = data["query_index"][row]
                contrastive_token = data["contrastive_object_token"][row]
                contrastive_indices = data["contrastive_query_index"][row]
                if count == 1 and token.ndim == 1:
                    token, center = token[None, :], center[None, :]
                    indices = np.asarray([indices])
                    contrastive_token = contrastive_token[None, :]
                    contrastive_indices = np.asarray([contrastive_indices])
                else:
                    token, center = token[:count], center[:count]
                    indices = indices[:count]
                    contrastive_token = contrastive_token[:count]
                    contrastive_indices = contrastive_indices[:count]
                indices = indices.tolist()
                contrastive_indices = contrastive_indices.tolist()
                assert len(indices) == len(set(indices))
                assert len(contrastive_indices) == len(set(contrastive_indices))
                assert token.shape == contrastive_token.shape == (count, 288)
                assert np.isfinite(token).all() and np.isfinite(contrastive_token).all()
                assert np.isfinite(center).all()
                samples += 1
                targets += count
                count_distribution[count] = count_distribution.get(count, 0) + 1
    print(json.dumps({"files": len(files), "samples": samples,
                      "targets": targets, "count_distribution": count_distribution,
                      "selection_policies": sorted(policies)}, indent=2))


if __name__ == "__main__":
    main()
