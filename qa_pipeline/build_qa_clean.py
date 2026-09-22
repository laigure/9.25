"""Build the strict v1 Waymo QA set (qa_waymo_clean.jsonl) + split/report.

Input : qa_pipeline/data/qa_waymo.jsonl        (never modified; keeps flags)
Output: qa_pipeline/data/qa_waymo_clean.jsonl
        qa_pipeline/data/qa_waymo_clean_stats.json

Filtering rules (directive 2026-09-20):
  1. drop rows with leak_audit.answer_direction_leaked == True (label leakage)
  2. drop rows with expression_collision == True (the 12 ambiguous pairs)
  3. near/far: v1 has no such labels, nothing to remove
Splitting: random by SCENE (scene_id), seed 42, 70/15/15 train/val/test.
  A scene lies entirely in one split (asserted below), so no pair from the
  same scene can land in two splits.

Binning done here (not in build_pairs.py, which already stores continuous
geometry), so thresholds can change without reprocessing raw data:
  - grounding_quality.bucket from target_iou / reference_iou against 0.5
    (0.5 = the official Acc@0.5 strict threshold): high_high / high_low /
    low_low. A 0.25 variant is counted in the stats only, to keep the row
    schema stable.

Provenance note: all 2,600 v1 pairs come from 3EED *val* scenes, because
tokens were exported during validation only. This train/val/test is a fresh
leakage-free random scene-level split inside those 903 scenes; it is NOT the
3EED train split.

Run:
    python build_qa_clean.py
"""

import argparse
import json
import os
import random
from collections import Counter

THRESHOLD = 0.5
THRESHOLD_ALT = 0.25
SEED = 42
RATIOS = (0.70, 0.15, 0.15)

HERE = os.path.dirname(os.path.abspath(__file__))


def quantiles(values, points=(("p05", 0.05), ("median", 0.5), ("p95", 0.95))):
    if not values:
        return {name: None for name, _ in points}
    xs = sorted(values)
    out = {}
    for name, p in points:
        idx = p * (len(xs) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(xs) - 1)
        frac = idx - lo
        out[name] = round(xs[lo] * (1 - frac) + xs[hi] * frac, 4)
    return out


def iou_bucket(value):
    if value < 0.1:
        return "lt_0.1"
    if value < 0.25:
        return "0.1_to_0.25"
    if value < 0.5:
        return "0.25_to_0.5"
    if value < 0.75:
        return "0.5_to_0.75"
    return "ge_0.75"


def quality_bucket(target_iou, reference_iou, threshold=THRESHOLD):
    high_target = target_iou >= threshold
    high_reference = reference_iou >= threshold
    if high_target and high_reference:
        return "high_high"
    if not high_target and not high_reference:
        return "low_low"
    return "high_low"


def split_stats(rows):
    distance = [r["relative_geometry"]["distance_3d"] for r in rows]
    min_iou = [r["meta"]["min_iou"] for r in rows]
    return {
        "scenes": len({r["scene_id"] for r in rows}),
        "pairs": len(rows),
        "relation_counts": dict(sorted(Counter(
            "{}_and_{}".format(*r["relations"]) for r in rows).items())),
        "bearing_sector_counts": dict(sorted(Counter(
            r["bearing_sector"] for r in rows).items())),
        "distance_3d_m": quantiles(distance),
        "min_iou": quantiles(min_iou),
        "min_iou_buckets": dict(sorted(Counter(
            iou_bucket(v) for v in min_iou).items())),
        "quality_bucket_counts": dict(sorted(Counter(
            r["grounding_quality"]["bucket"] for r in rows).items())),
        "quality_bucket_counts_alt_0.25": dict(sorted(Counter(
            quality_bucket(r["meta"]["target_iou"], r["meta"]["reference_iou"],
                           THRESHOLD_ALT) for r in rows).items())),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join(HERE, "data", "qa_waymo.jsonl"))
    parser.add_argument("--out", default=os.path.join(HERE, "data", "qa_waymo_clean.jsonl"))
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    dropped = {"answer_direction_leaked": [], "expression_collision": []}
    kept = []
    for row in rows:
        leaked = row["leak_audit"]["answer_direction_leaked"]
        collision = row["expression_collision"]
        if leaked:
            dropped["answer_direction_leaked"].append(row["qa_id"])
        if collision:
            dropped["expression_collision"].append(row["qa_id"])
        if leaked or collision:
            continue
        kept.append(row)

    # Scene-level split: shuffle the sorted scene list with a fixed seed.
    scenes = sorted({row["scene_id"] for row in kept})
    rng = random.Random(SEED)
    rng.shuffle(scenes)
    n_total = len(scenes)
    n_train = int(round(RATIOS[0] * n_total))
    n_val = int(round(RATIOS[1] * n_total))
    scene_split = {}
    for scene in scenes[:n_train]:
        scene_split[scene] = "train"
    for scene in scenes[n_train:n_train + n_val]:
        scene_split[scene] = "val"
    for scene in scenes[n_train + n_val:]:
        scene_split[scene] = "test"

    for row in kept:
        row["qa_split"] = scene_split[row["scene_id"]]
        target_iou = row["meta"]["target_iou"]
        reference_iou = row["meta"]["reference_iou"]
        row["grounding_quality"] = {
            "target_iou": target_iou,
            "reference_iou": reference_iou,
            "threshold": THRESHOLD,
            "bucket": quality_bucket(target_iou, reference_iou),
        }

    per_split = {name: [] for name in ("train", "val", "test")}
    for row in kept:
        per_split[row["qa_split"]].append(row)

    # Leakage confirmation: nothing flagged may survive; scenes never cross.
    overlaps = []
    names = ["train", "val", "test"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = ({r["scene_id"] for r in per_split[names[i]]} &
                      {r["scene_id"] for r in per_split[names[j]]})
            overlaps.extend(sorted(shared))
    clean_leaked = sum(r["leak_audit"]["answer_direction_leaked"] for r in kept)
    clean_collision = sum(r["expression_collision"] for r in kept)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": args.out,
        "original_total": len(rows),
        "dropped": {
            "answer_direction_leaked": len(dropped["answer_direction_leaked"]),
            "expression_collision": len(dropped["expression_collision"]),
            "union": len(rows) - len(kept),
            "note": "the 12 collision rows are a subset of the 75 leaked rows",
        },
        "scenes_total": len({row["scene_id"] for row in rows}),
        "scenes_in_clean": n_total,
        "scenes_dropped_entirely": len({row["scene_id"] for row in rows}) - n_total,
        "clean_total": len(kept),
        "dropped_qa_ids": dropped,
        "split": {
            "unit": "scene",
            "seed": SEED,
            "ratios": {"train": RATIOS[0], "val": RATIOS[1], "test": RATIOS[2]},
            "scenes_assigned": {
                "train": n_train, "val": n_val, "test": n_total - n_train - n_val},
            "scene_ids": {name: sorted({r["scene_id"] for r in per_split[name]})
                          for name in names},
        },
        "per_split": {name: split_stats(per_split[name]) for name in names},
        "leakage_check": {
            "answer_direction_leaked_in_clean": clean_leaked,
            "expression_collision_in_clean": clean_collision,
            "scene_overlap_between_splits": overlaps if overlaps else "none",
            "verdict": "pass" if (clean_leaked == 0 and clean_collision == 0
                                  and not overlaps) else "FAIL",
        },
        "near_far_labels": "absent_by_design",
        "provenance_note": (
            "all pairs are 3EED val scenes; this split is a random scene-level "
            "split within those 903 scenes, not the 3EED train split"),
    }
    stats_path = args.out.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    assert len(kept) == sum(len(v) for v in per_split.values())
    assert summary["leakage_check"]["verdict"] == "pass"
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("dropped_qa_ids", "split")},
                     indent=2, ensure_ascii=False))
    print("wrote", args.out, "and", stats_path)


if __name__ == "__main__":
    main()
