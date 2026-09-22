"""Human-checkable sanity report for the v1 Waymo QA set, plus aggregate stats.

Prints a random sample of pairs (expressions, GT centers, recomputed geometry,
question/answer, token pointers) and verifies with independent recomputation:
same frame, distinct expressions, 288-d tokens load from the npz, GT boxes
match the bank, dx/dy signs agree with the relation labels. Aggregate stats
cover scene/object/pair counts, relation and distance distributions, and IoU
spread including pairs per grounding-quality bucket.

    python sanity_check.py --qa .../qa_v1/qa_waymo.jsonl --bank .../qa_v1/object_bank.jsonl \
        --tokens-root .../qa_v1 --out .../qa_v1/sanity_check_waymo.txt \
        --report .../qa_v1/sanity_check_waymo.json
"""

import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(array.min()), 4),
        "p05": round(float(np.percentile(array, 5)), 4),
        "median": round(float(np.median(array)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "max": round(float(array.max()), 4),
    }


def check_record(qa, bank, tokens, token_ids):
    """Return (checks dict, lines list) for one QA record."""
    target = bank[qa["target_object_id"]]
    reference = bank[qa["reference_object_id"]]
    geometry = qa["relative_geometry"]
    # Independent recomputation from GT centers, in the verified frame.
    dx = round(target["gt_box"][0] - reference["gt_box"][0], 3)
    dy = round(target["gt_box"][1] - reference["gt_box"][1], 3)
    dz = round(target["gt_box"][2] - reference["gt_box"][2], 3)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    expected = [("left" if dy > 0 else "right"), ("front" if dx > 0 else "behind")]
    target_token = tokens[target["token_index"]] if tokens is not None else None
    reference_token = tokens[reference["token_index"]] if tokens is not None else None

    def token_ok(vector):
        return vector is None or (vector.shape == (288,) and not np.isnan(vector).any())

    checks = {
        "same_frame": qa["scene_id"] == target["scene_id"] == reference["scene_id"],
        "expressions_distinct": qa["target_expression"] != qa["reference_expression"],
        "gt_boxes_match_bank": (qa["target_gt_box"] == target["gt_box"]
                                and qa["reference_gt_box"] == reference["gt_box"]),
        "geometry_recomputed": (abs(dx - geometry["dx"]) < 1e-3
                                and abs(dy - geometry["dy"]) < 1e-3
                                and abs(dz - geometry["dz"]) < 1e-3
                                and abs(distance - geometry["distance_3d"]) < 1e-3),
        "direction_consistent": expected == qa["relations"],
        "token_dim_288": token_ok(target_token) and token_ok(reference_token),
        "token_id_matches": tokens is None or (
            token_ids[target["token_index"]] == target["object_id"]
            and token_ids[reference["token_index"]] == reference["object_id"]),
    }
    lines = [
        "target   [{} | {}] {}".format(target["object_id"].split("#")[-1],
                                       target["category"], qa["target_expression"]),
        "  gt_center {} | token_index {} (dim {})".format(
            [round(v, 2) for v in target["gt_box"][:3]], target["token_index"],
            target["token_dim"]),
        "reference[{} | {}] {}".format(reference["object_id"].split("#")[-1],
                                       reference["category"], qa["reference_expression"]),
        "  gt_center {} | token_index {} (dim {})".format(
            [round(v, 2) for v in reference["gt_box"][:3]], reference["token_index"],
            reference["token_dim"]),
        "geometry dx={} dy={} dz={} | dist2d={} dist3d={} | bearing={} sector={}".format(
            geometry["dx"], geometry["dy"], geometry["dz"], geometry["distance_2d"],
            geometry["distance_3d"], geometry["bearing_deg"], qa["bearing_sector"]),
        "relations {}".format(qa["relations"]),
        "Q: {}".format(qa["question"]),
        "A: {}".format(qa["answer"]),
        "checks: {} | grounding min_iou {} ({}) | leak {}".format(
            " ".join("{}={}".format(k, "OK" if v else "FAIL") for k, v in checks.items()),
            qa["meta"]["min_iou"], qa["meta"]["quality_bucket"],
            qa["leak_audit"]["answer_direction_leaked"]),
    ]
    return checks, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa", default="/root/autodl-tmp/3eed_data/qa_v1/qa_waymo.jsonl")
    parser.add_argument("--bank", default="/root/autodl-tmp/3eed_data/qa_v1/object_bank.jsonl")
    parser.add_argument("--tokens-root", default="/root/autodl-tmp/3eed_data/qa_v1")
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(args.tokens_root, "sanity_check.txt")
    report_path = args.report or os.path.join(args.tokens_root, "sanity_check.json")

    bank, platform = {}, None
    with open(args.bank, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            bank[record["object_id"]] = record
    records = []
    with open(args.qa, encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    platform = records[0]["platform"]

    tokens, token_ids = None, None
    token_file = os.path.join(args.tokens_root, records[0]["target_token_path"])
    if os.path.exists(token_file):
        data = np.load(token_file, allow_pickle=True)
        tokens = data["object_token"]
        token_ids = [str(v) for v in data["object_id"]]

    sample = random.Random(args.seed).sample(records, min(args.sample, len(records)))
    lines = ["random sample of {} pairs (seed {})".format(len(sample), args.seed),
             "=" * 90]
    failures = Counter()
    for number, qa in enumerate(sample, 1):
        checks, block = check_record(qa, bank, tokens, token_ids)
        for key, ok in checks.items():
            if not ok:
                failures[key] += 1
        lines.append("[{:02d}] {}".format(number, qa["qa_id"]))
        lines.extend(block)
        lines.append("-" * 90)

    scenes = {r["scene_id"] for r in records}
    objects = set()
    for r in records:
        objects.add(r["target_object_id"])
        objects.add(r["reference_object_id"])
    distances = np.asarray([r["relative_geometry"]["distance_3d"] for r in records])
    min_ious = np.asarray([r["meta"]["min_iou"] for r in records], dtype=np.float64)
    bins = [0, 5, 10, 20, 40, 1e9]
    histogram = {"{:.0f}-{:.0f}m".format(bins[i], bins[i + 1]): int(count)
                 for i, count in enumerate(np.histogram(distances, bins=bins)[0])}

    summary = {
        "qa_records": len(records),
        "scenes": len(scenes),
        "objects": len(objects),
        "lateral_counts": dict(Counter(r["relations"][0] for r in records)),
        "longitudinal_counts": dict(Counter(r["relations"][1] for r in records)),
        "sector_counts": dict(Counter(r["bearing_sector"] for r in records)),
        "relation_pair_counts": dict(Counter(
            "{}_and_{}".format(*r["relations"]) for r in records)),
        "distance_3d_m": quantiles(distances),
        "distance_bins": histogram,
        "min_iou": quantiles(min_ious),
        "target_iou": quantiles([r["meta"]["target_iou"] for r in records]),
        "reference_iou": quantiles([r["meta"]["reference_iou"] for r in records]),
        "quality_bucket_counts": dict(Counter(r["meta"]["quality_bucket"] for r in records)),
        "answer_direction_leaked_rate": round(float(np.mean(
            [r["leak_audit"]["answer_direction_leaked"] for r in records])), 4),
        "sample_check_failures": dict(failures),
        "token_file": os.path.relpath(token_file, args.tokens_root),
        "token_rows": None if tokens is None else int(tokens.shape[0]),
    }
    lines.append("aggregate summary")
    lines.append(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print("wrote", out, "and", report_path)


if __name__ == "__main__":
    main()
