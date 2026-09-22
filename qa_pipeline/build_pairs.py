"""Pair objects that share a frame and compute relative geometry from GT boxes.

Only same-frame ordered pairs (A, B) = "A relative to B" are produced, and all
geometry comes from GT box centers. Continuous quantities (dx/dy/dz, planar and
3D distance, bearing) are stored raw; the `relations` list and `bearing_sector`
are derived convenience labels that can be recomputed later without touching
the source data.

Frame convention (verified in coordinate_convention.md): front = +x, left = +y,
up = +z. Bearing = atan2(dy, dx) degrees; sectors: front |b|<=45,
left 45<b<=135, behind |b|>135, right -135<=b<-45. The `relations` list uses the
independent-axis form (front/behind from dx sign, left/right from dy sign),
which is what the answer templates speak; both forms are kept for comparison.

    python build_pairs.py --bank /root/autodl-tmp/3eed_data/qa_v1/object_bank.jsonl \
        --platform waymo --out /root/autodl-tmp/3eed_data/qa_v1/pairs_waymo.jsonl
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict

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


def bearing_sector(degrees):
    if -45 <= degrees <= 45:
        return "front"
    if 45 < degrees <= 135:
        return "left"
    if -135 <= degrees < -45:
        return "right"
    return "behind"


def pair_geometry(target, reference):
    """All quantities from GT box centers; +x front, +y left (see module doc)."""
    dx = target[0] - reference[0]
    dy = target[1] - reference[1]
    dz = target[2] - reference[2]
    distance_2d = math.hypot(dx, dy)
    distance_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
    bearing_deg = math.degrees(math.atan2(dy, dx))
    relations = [
        "left" if dy > 0 else "right",
        "front" if dx > 0 else "behind",
    ]
    return {
        "dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4),
        "distance_2d": round(distance_2d, 4),
        "distance_3d": round(distance_3d, 4),
        "bearing_deg": round(bearing_deg, 3),
        "bearing_sector": bearing_sector(bearing_deg),
        "relations": relations,
    }


def quality_bucket(iou):
    if iou is None:
        return "no_iou"
    for edge in (0.1, 0.25, 0.5, 0.75):
        if iou < edge:
            return "lt_{}".format(edge)
    return "ge_0.75"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", default="/root/autodl-tmp/3eed_data/qa_v1/object_bank.jsonl")
    parser.add_argument("--platform", default="waymo")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-pairs-per-frame", type=int, default=30,
                        help="deterministic cap on ordered pairs per frame")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.bank.rsplit("/", 1)[0] + "/pairs_{}.jsonl".format(args.platform)

    frames = defaultdict(list)
    with open(args.bank, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["platform"] == args.platform:
                frames[record["scene_id"]].append(record)

    multi_frames = {k: v for k, v in frames.items() if len(v) >= 2}
    pairs, pairs_per_frame = [], Counter()
    capped_frames = 0
    for scene_id in sorted(multi_frames):
        records = multi_frames[scene_id]
        ordered = [(a, b) for a in records for b in records if a["object_id"] != b["object_id"]]
        ordered.sort(key=lambda ab: (ab[0]["object_id"], ab[1]["object_id"]))
        if len(ordered) > args.max_pairs_per_frame:
            capped_frames += 1
            ordered = random.Random(scene_id).sample(ordered, args.max_pairs_per_frame)
            ordered.sort(key=lambda ab: (ab[0]["object_id"], ab[1]["object_id"]))
        pairs_per_frame[len(ordered)] += 1
        for target, reference in ordered:
            geometry = pair_geometry(target["gt_box"][:3], reference["gt_box"][:3])
            ious = [v for v in (target["iou"], reference["iou"]) if v is not None]
            pair = {
                "pair_id": "{}::{}::{}".format(scene_id, target["object_id"],
                                               reference["object_id"]),
                "platform": args.platform,
                "scene_id": scene_id,
                "frame_id": target["frame_id"],
                "target_object_id": target["object_id"],
                "reference_object_id": reference["object_id"],
                "target_category": target["category"],
                "reference_category": reference["category"],
                "target_gt_box": target["gt_box"],
                "reference_gt_box": reference["gt_box"],
                "min_iou": round(min(ious), 5) if ious else None,
                "quality_bucket": quality_bucket(min(ious) if ious else None),
            }
            pair.update(geometry)
            pairs.append(pair)

    with open(args.out, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    sector_counts = Counter(p["bearing_sector"] for p in pairs)
    lateral = Counter(p["relations"][0] for p in pairs)
    longitudinal = Counter(p["relations"][1] for p in pairs)
    lateral_agreement = [
        (p["relations"][0] == p["bearing_sector"]) for p in pairs
        if p["bearing_sector"] in ("left", "right")
    ]
    stats = {
        "platform": args.platform,
        "frames_with_tokens": len(frames),
        "frames_with_two_or_more": len(multi_frames),
        "objects_in_multi_frames": sum(len(v) for v in multi_frames.values()),
        "pairs": len(pairs),
        "capped_frames": capped_frames,
        "max_pairs_per_frame": args.max_pairs_per_frame,
        "pairs_per_frame": {str(k): v for k, v in sorted(pairs_per_frame.items())},
        "bearing_sector_counts": dict(sector_counts),
        "lateral_counts": dict(lateral),
        "longitudinal_counts": dict(longitudinal),
        "sector_vs_axis_lateral_agreement": round(
            float(np.mean(lateral_agreement)), 4) if lateral_agreement else None,
        "boundary_zone_rate_40_50deg": round(float(np.mean(
            [40 <= abs(p["bearing_deg"]) <= 50 for p in pairs])), 4),
        "distance_2d_m": quantiles([p["distance_2d"] for p in pairs]),
        "distance_3d_m": quantiles([p["distance_3d"] for p in pairs]),
        "abs_dy_m": quantiles([abs(p["dy"]) for p in pairs]),
        "min_iou": quantiles([p["min_iou"] for p in pairs if p["min_iou"] is not None]),
        "target_iou_ge_0.25_rate": round(float(np.mean(
            [p["min_iou"] >= 0.25 for p in pairs if p["min_iou"] is not None])), 4),
        "quality_bucket_counts": dict(Counter(p["quality_bucket"] for p in pairs)),
        "distance_note": "near/far thresholds intentionally not applied in v1",
        "label_source": "ground_truth_geometry",
    }
    stats_path = args.out.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("wrote", args.out, "and", stats_path)


if __name__ == "__main__":
    main()
