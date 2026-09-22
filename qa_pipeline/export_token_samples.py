"""Pull a few readable sample records out of the exported grounding token NPZ files.

Each NPZ batch written by `--export_qa_tokens` holds the selected object token
plus its metadata for every caption in the batch. This script picks, per
platform, three rows spread across the box-error range (best / median / worst
distance between predicted center and GT center), writes them as one readable
JSON, and copies the source NPZ batches next to it. It is meant for handing
samples back to the local workstation, not for training.

    python export_token_samples.py --mode waymo drone quad \
        --out-dir /root/autodl-tmp/3eed_data/token_samples
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np

TOKEN_DECIMALS = 5
MAX_PROBE_FILES = 60


def candidates(files):
    rows = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        for index in range(len(data["query_index"])):
            predicted = data["pred_center"][index]
            ground_truth = data["gt_box"][index][:3]
            rows.append({
                "file": path,
                "row": index,
                "score": float(data["match_score"][index]),
                "center_error_m": float(np.linalg.norm(predicted - ground_truth)),
            })
    return rows


def to_record(platform, entry, npz_name):
    data = np.load(entry["file"], allow_pickle=True)
    index = entry["row"]
    token = data["object_token"][index]
    return {
        "platform": platform,
        "source_npz": npz_name,
        "row_in_batch": index,
        "utterance": str(data["utterance"][index]),
        "query_index": int(data["query_index"][index]),
        "match_score": round(float(data["match_score"][index]), 4),
        "pred_center": [round(float(v), 3) for v in data["pred_center"][index]],
        "pred_size": [round(float(v), 3) for v in data["pred_size"][index]],
        "gt_box_7": [round(float(v), 3) for v in data["gt_box"][index]],
        "center_error_m": round(entry["center_error_m"], 3),
        "token_dim": int(token.shape[0]),
        "token_l2_norm": round(float(np.linalg.norm(token)), 3),
        "object_token": [round(float(v), TOKEN_DECIMALS) for v in token],
        "meta_path": str(data["meta_path"][index]),
        "feature_name": str(data["feature_name"]),
        "selection_policy": str(data["selection_policy"]),
        "checkpoint_path": str(data["checkpoint_path"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", nargs="+", default=["waymo", "drone", "quad"],
                        help="platforms to sample (must match the token export dirs)")
    parser.add_argument("--tokens-root", default="/root/autodl-tmp/3eed_data")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/3eed_data/token_samples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for platform in args.mode:
        pattern = os.path.join(args.tokens_root, "eval_{}_tokens".format(platform),
                               "**", "*.npz")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            print("no token files for", platform)
            continue
        step = max(1, len(files) // MAX_PROBE_FILES)
        rows = candidates(files[::step][:MAX_PROBE_FILES])
        rows.sort(key=lambda row: row["center_error_m"])
        picks, used_files = [], set()
        for target in (0, len(rows) // 2, len(rows) - 1):
            # nearest position to the best / median / worst error, from an
            # unused batch so the three samples stay independent
            for position in sorted(range(len(rows)), key=lambda p: (abs(p - target), p)):
                if rows[position]["file"] not in used_files:
                    used_files.add(rows[position]["file"])
                    picks.append(rows[position])
                    break

        records = []
        for number, entry in enumerate(picks, 1):
            npz_name = "{}_sample_{:02d}_batch.npz".format(platform, number)
            shutil.copyfile(entry["file"], os.path.join(args.out_dir, npz_name))
            record = to_record(platform, entry, npz_name)
            records.append(record)
            print("{} sample {}: score {} err {}m :: {}".format(
                platform, number, record["match_score"], record["center_error_m"],
                record["utterance"][:70]), flush=True)

        out_path = os.path.join(args.out_dir, "token_samples_{}.json".format(platform))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)
        print("wrote", out_path, "({} records)".format(len(records)))


if __name__ == "__main__":
    main()
