"""Build the per-object geometry array used by Experiments D/E/oracle.

Runs on the dataset host (the 473 export npz files live there). Rebuilds the
grounding-predicted geometry in exactly the row order of the stacked token
npz (waymo_tokens.npz), which is the order build_object_bank.py used:

  * iterate eval_waymo_tokens/**/*.npz in sorted glob order, rows in order
  * take pred_center / pred_size from the export, gt center/size from gt_box
  * HARD CHECK: the concatenated object_token rows must equal the stacked
    token npz exactly, and the rebuilt values must match object_bank.jsonl
  * object_id comes from the stacked npz (the export npz has no id)

Output: <tokens_root>/waymo_geometry.npz with pred_center, pred_size,
gt_center, gt_size, object_id.

Usage (on the host):
    python build_geom_npz.py --tokens-root /root/autodl-tmp/3eed_data/qa_v1 \
        --export-root /root/autodl-tmp/3eed_data \
        --bank /root/autodl-tmp/3eed_data/qa_v1/object_bank.jsonl
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-root", default="/root/autodl-tmp/3eed_data/qa_v1")
    parser.add_argument("--export-root", default="/root/autodl-tmp/3eed_data")
    parser.add_argument("--platform", default="waymo")
    parser.add_argument("--bank", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pattern = os.path.join(args.export_root, "eval_{}_tokens".format(args.platform),
                           "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))
    assert files, "no export npz under " + pattern
    print("export files:", len(files))

    centers, sizes, gcenters, gsizes, tokens = [], [], [], [], []
    for path in files:
        data = np.load(path)
        centers.append(data["pred_center"].astype(np.float32))
        sizes.append(data["pred_size"].astype(np.float32))
        gt = np.asarray(data["gt_box"], dtype=np.float32)
        gcenters.append(gt[:, :3])
        gsizes.append(gt[:, 3:6])
        tokens.append(np.asarray(data["object_token"], dtype=np.float32))
    pred_center = np.concatenate(centers)
    pred_size = np.concatenate(sizes)
    gt_center = np.concatenate(gcenters)
    gt_size = np.concatenate(gsizes)
    token_rows = np.concatenate(tokens)

    stacked_path = os.path.join(args.tokens_root, "tokens",
                                "{}_tokens.npz".format(args.platform))
    stacked = np.load(stacked_path)
    object_id = stacked["object_id"]
    assert pred_center.shape[0] == len(object_id), \
        "row count {} != stacked {}".format(pred_center.shape[0], len(object_id))
    assert np.array_equal(token_rows, np.asarray(stacked["object_token"])), \
        "rebuilt rows do not match the stacked token npz row for row"
    print("row order verified against", stacked_path, "rows", len(object_id))

    if args.bank and os.path.exists(args.bank):
        bank = [json.loads(line) for line in open(args.bank, encoding="utf-8")]
        bank = [r for r in bank if r["platform"] == args.platform]
        assert len(bank) == len(object_id)
        bank_center = np.array([r["pred_center"] for r in bank], dtype=np.float32)
        bank_size = np.array([r["pred_size"] for r in bank], dtype=np.float32)
        bank_gt = np.array([r["gt_box"][:3] for r in bank], dtype=np.float32)
        for name, rebuilt, ref in (("pred_center", pred_center, bank_center),
                                   ("pred_size", pred_size, bank_size),
                                   ("gt_center", gt_center, bank_gt)):
            gap = float(np.abs(rebuilt - ref).max())
            print("{} vs object_bank max abs diff {:.6f}".format(name, gap))
            assert gap < 1e-4, name + " disagrees with object_bank"
        assert all(str(r["object_id"]) == str(object_id[i])
                   for i, r in enumerate(bank)), "object_id order mismatch"

    error = np.linalg.norm(pred_center - gt_center, axis=1)
    print("center error: median {:.3f} m | p95 {:.3f} m | max {:.3f} m".format(
        float(np.median(error)), float(np.percentile(error, 95)), float(error.max())))
    print("pred_center mean", pred_center.mean(axis=0).round(3),
          "std", pred_center.std(axis=0).round(3))

    out = args.out or os.path.join(args.tokens_root, "tokens",
                                   "{}_geometry.npz".format(args.platform))
    np.savez_compressed(out, pred_center=pred_center, pred_size=pred_size,
                        gt_center=gt_center, gt_size=gt_size, object_id=object_id)
    print("wrote", out)


if __name__ == "__main__":
    main()
