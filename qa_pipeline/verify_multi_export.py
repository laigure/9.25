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
    for path in files:
        with np.load(path) as data:
            assert data["selection_policy"].item() == "soft_token_unique_assignment_text_spans"
            for row, count in enumerate(data["target_count"].tolist()):
                ids = json.loads(data["object_ids_json"][row])
                assert count == len(ids) and count in (2, 3), (path, row)
                indices = data["query_index"][row, :count].tolist()
                assert len(indices) == len(set(indices))
                assert data["object_token"][row, :count].shape == (count, 288)
                assert np.isfinite(data["object_token"][row, :count]).all()
                assert np.isfinite(data["pred_center"][row, :count]).all()
                samples += 1
                targets += count
    print(json.dumps({"files": len(files), "samples": samples, "targets": targets}, indent=2))


if __name__ == "__main__":
    main()
