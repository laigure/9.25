"""Summarize selected versus GT-nearest-16 box hits from multi-target eval."""

import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.predictions.rglob("prediction.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("prefix") == "last_" and "nearest16_ious" in row:
            records.append(row)
    if not records:
        raise SystemExit("No final-layer multi-target records found")
    summary = {"samples": len(records), "target_counts": dict(Counter(len(r["ious"]) for r in records))}
    for threshold in (0.25, 0.5):
        for name, field in (("selected", "ious"), ("nearest16_gt_center", "nearest16_ious")):
            per_target = [v >= threshold for r in records for v in r[field]]
            all_targets = [all(v >= threshold for v in r[field]) for r in records]
            summary[f"{name}_target_at_{threshold}"] = sum(per_target) / len(per_target)
            summary[f"{name}_all_at_{threshold}"] = sum(all_targets) / len(all_targets)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
