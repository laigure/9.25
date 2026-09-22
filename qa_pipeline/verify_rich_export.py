"""Cross-check a rich-feature export against its records.jsonl.

For every record, load the npz shard it points to and verify the stored
row content matches (noun top1 candidate index and its IoU at the last
layer, which the export writes from the same forward pass). Also checks
per-split shard counts against ceil(eligible / batch_size).

Exit code 0 only if everything matches.

Usage: python verify_rich_export.py [--features DIR] [--batch-size 8]
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="/root/autodl-tmp/3eed_data/qa_features_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               open(os.path.join(args.features, "records.jsonl"), encoding="utf-8")]
    print("records:", len(records))

    by_split = {}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)

    ok = True
    for split, rows in sorted(by_split.items()):
        by_file = {}
        for record in rows:
            by_file.setdefault(record["feature_file"], []).append(record)
        found = [name for name in os.listdir(os.path.join(args.features, "rank00"))
                 if name.startswith(split + "_batch") and name.endswith(".npz")]
        expected = -(-len(rows) // args.batch_size)
        shard_ok = len(found) == expected
        ok = ok and shard_ok
        print(f"{split}: records {len(rows)} | shards {len(found)} (expected {expected})"
              f" {'ok' if shard_ok else 'MISMATCH'}")

        bad = 0
        checked = 0
        for rel, file_rows in sorted(by_file.items()):
            with np.load(os.path.join(args.features, rel)) as data:
                if data["layer_features_top5"].shape[1:] != (6, 5, 288):
                    print("  shape mismatch:", rel, data["layer_features_top5"].shape)
                    ok = False
                    continue
                for record in file_rows:
                    row = record["feature_row"]
                    idx = int(data["noun_top1_idx_last"][row])
                    iou = float(data["layer_iou"][row, 5, idx])
                    checked += 1
                    if idx != int(record["noun_top1_idx_last"]):
                        bad += 1
                    elif abs(iou - float(record["noun_top1_iou_last"])) > 1e-4:
                        bad += 1
        print(f"  rows checked {checked} | content mismatches {bad}")
        ok = ok and bad == 0

    stats_path = os.path.join(args.features, "export_stats.json")
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path, encoding="utf-8"))
        for split, entry in sorted(stats.get("splits", {}).items()):
            keys = [k for k in ("exported", "eligible", "shards",
                                "noun_policy_query_index_match_vs_bank",
                                "noun_policy_last_layer") if k in entry]
            print(split, "stats:", json.dumps({k: entry[k] for k in keys}, ensure_ascii=False))

    print("VERIFY", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
