"""Audit 3EED metadata before writing the scene converter and QA generator.

Runs on the host that holds the dataset (CPU only, no model, no GPU). Reads
`meta_info.json` for sampled frames of one or more platforms, plus a few lidar
files, and reports:

  * top-level metadata keys and the `ground_info` entry schema. Each entry
    carries one referred object plus a nested `others` list of further boxes
    (`bbox_3d_other` / `class_other`), which is where reference objects for
    two-object QA come from.
  * `bbox_3d` length and per-index value ranges, to pin the parameter order
  * caption coverage, class-word matches, spatial-word leakage rate
  * which axis convention reproduces the spatial words found in captions,
    scored on every candidate frame interpretation: raw metadata, the model's
    front-view rotation (waymo), and pose applied / inverted (drone, quad)
  * whether `pose` moves a box toward or away from the point cloud

The spatial-word check is the point of this script. `left`/`right`/`front`/
`behind` in a caption are the only ground truth available for the axis
convention, and the QA generator must reproduce that convention exactly.

Nothing is written except the JSON report.

    python audit_3eed.py --data-root data/3eed --split-dir data/3eed/splits \
        --out audit_report.json
"""

import argparse
import ast
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:  # metadata-only audit still works without numpy
    np = None


WAYMO_VIEWS = ["F", "FL", "FR", "SL", "SR"]
WAYMO_VIEW_ANGLES = {"F": 0.0, "FL": -45.0, "FR": 45.0, "SL": -90.0, "SR": 90.0}

RELATION_WORDS = re.compile(r"\b(left|right|front|behind|back)\b", re.IGNORECASE)
ANY_SPATIAL = re.compile(
    r"\b(left|right|front|behind|back|near|far|closest|farthest|between|"
    r"above|below|next to|beside|adjacent|opposite)\b",
    re.IGNORECASE,
)

# Words that constrain the lateral (side) axis of the ego frame.
LATERAL_WORDS = {"left", "right"}

# Each convention fixes the meaning of the two horizontal axes. x is always
# the longitudinal axis; hypotheses differ in its sign and in which direction
# of y counts as `left`.
CONVENTIONS = {
    "x_fwd_y_left": "front = +x, left = +y (ROS style, right-handed, z up)",
    "x_fwd_y_right": "front = +x, left = -y (image style)",
    "x_back_y_left": "front = -x, left = +y",
    "x_back_y_right": "front = -x, left = -y (180 degree flipped)",
}

MIN_AXIS_M = 1.0
AXIS_MARGIN_M = 0.5


def load_synonyms(path):
    """Read WAYMO_SYNONYMS / M3ED_SYNONYMS out of the training repo.

    The lists live in `src/joint_det_dataset.py`, which imports torch and the
    compiled ops, so parse the file instead of importing it.
    """
    synonyms = {}
    if not path or not os.path.exists(path):
        return synonyms
    try:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
    except (OSError, SyntaxError):
        return synonyms
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = getattr(target, "id", None)
            if name in ("WAYMO_SYNONYMS", "M3ED_SYNONYMS"):
                try:
                    synonyms[name] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return synonyms


def class_words(synonyms, dataset, klass):
    if dataset == "waymo":
        table = synonyms.get("WAYMO_SYNONYMS", {})
    else:
        table = synonyms.get("M3ED_SYNONYMS", {})
    return table.get(klass, [klass])


def rotate_xy(vector, angle_deg):
    """Rotate a 3D vector about z, matching utils/transform_waymo.py."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return [cos_a * x - sin_a * y, sin_a * x + cos_a * y, z]


def unit_relationship(box):
    """Normalize one bbox_3d entry into a dict, keeping unknown tail values."""
    values = [float(v) for v in box]
    return {
        "values": values,
        "center": values[:3],
        "dims": values[3:6] if len(values) >= 6 else [],
        "yaw_rad": values[6] if len(values) >= 7 else None,
    }


def find_relation_word(caption):
    """Return the single relation word in the caption, else None."""
    words = {match.group(1).lower() for match in RELATION_WORDS.finditer(caption)}
    if len(words) != 1:
        return None
    word = words.pop()
    return "behind" if word == "back" else word


def convention_matches(convention, center, word):
    """Check one hypothesis against one caption word.

    The lidar frame origin is the sensor, so the box center offset is
    ego-relative. Only the sign of the relevant coordinate is tested, so a
    diagonal object still votes on the axis it is being asked about. Returns
    None when the offset is too small to be trusted.
    """
    x, y = center[0], center[1]
    if word in LATERAL_WORDS:
        if abs(y) < MIN_AXIS_M:
            return None
        left_is_positive = convention.endswith("y_left")
        predicted_left = y > 0 if left_is_positive else y < 0
        return predicted_left == (word == "left")
    if abs(x) < MIN_AXIS_M:
        return None
    front_is_positive = convention.startswith("x_fwd")
    predicted_front = x > 0 if front_is_positive else x < 0
    return predicted_front == (word == "front")


def dominant_axis(center, margin=AXIS_MARGIN_M):
    """Name the geometric relation a QA generator would emit for this center."""
    lateral, longitudinal = center[1], center[0]
    if abs(lateral) > abs(longitudinal) + margin:
        return "left" if lateral > 0 else "right"
    if abs(longitudinal) > abs(lateral) + margin:
        return "front" if longitudinal > 0 else "behind"
    return None


def relation_test(entries, variant):
    """Tally hypotheses against the captions of one frame variant.

    `entries` is a list of (caption, center) pairs for a single frame.
    """
    result = {
        "variant": variant,
        "entries": len(entries),
        "lateral_samples": 0,
        "longitudinal_samples": 0,
        "per_word": {word: {"match": 0, "total": 0} for word in
                     ("left", "right", "front", "behind")},
        "dominant_axis_agreement": {"match": 0, "total": 0},
        "convention_score": {name: {"match": 0, "total": 0} for name in CONVENTIONS},
    }
    for caption, center in entries:
        word = find_relation_word(caption)
        if word is None:
            continue
        if word in LATERAL_WORDS:
            result["lateral_samples"] += 1
        else:
            result["longitudinal_samples"] += 1
        for name in CONVENTIONS:
            verdict = convention_matches(name, center, word)
            if verdict is None:
                continue
            result["convention_score"][name]["total"] += 1
            if verdict:
                result["convention_score"][name]["match"] += 1
        if dominant_axis(center) is not None:
            result["dominant_axis_agreement"]["total"] += 1
            if dominant_axis(center) == word:
                result["dominant_axis_agreement"]["match"] += 1
        result["per_word"][word]["total"] += 1
        if convention_matches("x_fwd_y_left", center, word):
            result["per_word"][word]["match"] += 1
    for name, score in result["convention_score"].items():
        score["rate"] = round(score["match"] / score["total"], 4) if score["total"] else None
        score["meaning"] = CONVENTIONS[name]
    for word, score in result["per_word"].items():
        score["rate"] = round(score["match"] / score["total"], 4) if score["total"] else None
    result["dominant_axis_agreement"]["rate"] = (
        round(result["dominant_axis_agreement"]["match"] /
              result["dominant_axis_agreement"]["total"], 4)
        if result["dominant_axis_agreement"]["total"] else None
    )
    return result


def index_stats(rows, limit=9):
    """Per-index min/max/mean over a list of numeric vectors."""
    if not rows:
        return []
    width = min(max(len(row) for row in rows), limit)
    stats = []
    for index in range(width):
        column = [row[index] for row in rows if len(row) > index and math.isfinite(row[index])]
        if not column:
            stats.append(None)
            continue
        stats.append({
            "min": round(min(column), 4),
            "max": round(max(column), 4),
            "mean": round(sum(column) / len(column), 4),
        })
    return stats


def describe_counter(counter, top=20):
    return dict(counter.most_common(top))


def load_lidar(path, max_points=200000):
    if np is None:
        return None
    if path.endswith(".npy"):
        points = np.load(path)
    else:
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 4 == 0:
            points = points.reshape(-1, 4)
        else:
            points = points.reshape(-1, 3)
    points = points[:, :3]
    if points.shape[0] > max_points:
        step = points.shape[0] // max_points
        points = points[::step]
    return points


def summarize_points(points):
    if points is None or points.shape[0] == 0:
        return {"available": False}
    radius = np.linalg.norm(points, axis=1)
    return {
        "available": True,
        "points_sampled": int(points.shape[0]),
        "axis_min": [round(float(v), 3) for v in points.min(axis=0)],
        "axis_max": [round(float(v), 3) for v in points.max(axis=0)],
        "axis_mean": [round(float(v), 3) for v in points.mean(axis=0)],
        "axis_std": [round(float(v), 3) for v in points.std(axis=0)],
        "radius_p05_p50_p95": [
            round(float(np.percentile(radius, q)), 3) for q in (5, 50, 95)
        ],
        "inside_100m_rate": round(float((radius < 100).mean()), 4),
        "inside_200m_rate": round(float((radius < 200).mean()), 4),
    }


def pose_direction_test(poses, centers, points):
    """Decide whether `pose` maps ego -> world or world -> ego.

    A box center in the ego frame should land inside the point cloud when
    mapped the correct way round, so count how often each direction does.
    """
    if np is None or not poses or points is None or points.shape[0] == 0:
        return {"available": False}
    limits = points.min(axis=0), points.max(axis=0)
    span = limits[1] - limits[0]
    low, high = limits[0] - 0.1 * span, limits[1] + 0.1 * span

    def inside(center):
        return bool(np.all(center >= low) and np.all(center <= high))

    direct_hits = inverse_hits = 0
    tested = 0
    for pose, center in zip(poses, centers):
        matrix = np.array(pose, dtype=np.float64)
        if matrix.shape != (4, 4):
            continue
        homogeneous = np.array([center[0], center[1], center[2], 1.0])
        direct = (matrix @ homogeneous)[:3]
        inverse = (np.linalg.inv(matrix) @ homogeneous)[:3]
        direct_hits += inside(direct)
        inverse_hits += inside(inverse)
        tested += 1
    if not tested:
        return {"available": False}
    return {
        "available": True,
        "boxes_tested": tested,
        "pose_maps_to_cloud_rate": round(direct_hits / tested, 4),
        "inverse_maps_to_cloud_rate": round(inverse_hits / tested, 4),
        "conclusion": (
            "pose maps ego -> world"
            if direct_hits > inverse_hits
            else "pose maps world -> ego"
            if inverse_hits > direct_hits
            else "ambiguous on this sample"
        ),
    }


def audit_platform(data_root, split_dir, dataset, splits, max_frames, random_seed,
                   lidar_samples, object_sample_cap, synonyms):
    platform_dir = os.path.join(data_root, dataset)
    sequences = []
    missing_splits = []
    for split in splits:
        split_file = os.path.join(split_dir, f"{dataset}_{split}.txt")
        if not os.path.exists(split_file):
            missing_splits.append(split_file)
            continue
        with open(split_file) as handle:
            sequences += [line.strip() for line in handle if line.strip()]
    if missing_splits:
        return {"error": "split file not found: " + ", ".join(missing_splits)}

    frames = sorted({
        (sequence, frame)
        for sequence in sequences
        if os.path.isdir(os.path.join(platform_dir, sequence))
        for frame in os.listdir(os.path.join(platform_dir, sequence))
        if os.path.isdir(os.path.join(platform_dir, sequence, frame))
    })
    total_frames = len(frames)
    if max_frames and total_frames > max_frames:
        frames = random.Random(random_seed).sample(frames, max_frames)
        frames.sort()

    report = {
        "splits": splits,
        "sequences_listed": len(set(sequences)),
        "frames_available": total_frames,
        "frames_scanned": len(frames),
        "missing_meta": 0,
        "unreadable_meta": 0,
        "meta_top_level_keys": Counter(),
        "ground_info_entry_keys": Counter(),
        "others_entry_keys": Counter(),
        "ground_info_per_frame": Counter(),
        "others_per_entry": Counter(),
        "others_num_mismatch": 0,
        "classes": Counter(),
        "others_classes": Counter(),
        "class_string_absent_in_caption": 0,
        "loader_class_matched": 0,
        "loader_class_unmatched_examples": [],
        "caption_with_any_spatial_word": 0,
        "caption_with_relation_word": 0,
        "caption_multi_relation_word": 0,
        "caption_samples": [],
        "ground_info_total": 0,
        "others_total": 0,
        "ground_info_with_id_field": 0,
        "frames_with_two_captions": 0,
        "duplicate_caption_frames": 0,
        "bbox_rows": defaultdict(list),
        "others_bbox_rows": defaultdict(list),
        "relation_tests": {},
        "poses": [],
        "pose_centers": [],
        "pose_shape": Counter(),
        "samples": {},
    }
    # variant name -> list of (caption, center) under that interpretation of
    # the metadata coordinates. The variant with the highest caption match rate
    # is the frame the QA labels must be generated in.
    relation_entries = defaultdict(list)

    for sequence, frame in frames:
        frame_path = os.path.join(platform_dir, sequence, frame)
        meta_path = os.path.join(frame_path, "meta_info.json")
        if not os.path.exists(meta_path):
            report["missing_meta"] += 1
            continue
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            report["unreadable_meta"] += 1
            continue

        for key in meta:
            report["meta_top_level_keys"][key] += 1

        ground_info = meta.get("ground_info") or []
        report["ground_info_per_frame"][len(ground_info)] += 1
        report["ground_info_total"] += len(ground_info)

        if not report["samples"]:
            report["samples"] = {
                "frame": os.path.join(sequence, frame),
                "meta_top_level_keys": sorted(meta.keys()),
                "ground_info_0": ground_info[:1],
            }
        elif ground_info and len(report["caption_samples"]) < 5:
            report["caption_samples"].append({
                "frame": os.path.join(sequence, frame),
                "class": ground_info[0].get("class"),
                "caption": str(ground_info[0].get("caption"))[:400],
            })

        pose = meta.get("pose")
        pose_matrix = pose_inverse = None
        if isinstance(pose, dict):
            for key in pose:
                report["pose_shape"][f"dict:{key}"] += 1
        elif isinstance(pose, list):
            report["pose_shape"]["{}x{}".format(
                len(pose), len(pose[0]) if pose and isinstance(pose[0], list) else 0)] += 1
            report["poses"].append(pose)
            if ground_info and ground_info[0].get("bbox_3d"):
                report["pose_centers"].append(
                    unit_relationship(ground_info[0]["bbox_3d"])["center"]
                )
            if np is not None:
                candidate = np.array(pose, dtype=np.float64)
                if candidate.shape == (4, 4):
                    try:
                        pose_matrix = candidate
                        pose_inverse = np.linalg.inv(candidate)
                    except np.linalg.LinAlgError:
                        pose_matrix = pose_inverse = None

        captions = []
        for entry in ground_info:
            for key in entry:
                report["ground_info_entry_keys"][key] += 1
            if "id" in entry or "object_id" in entry or "track_id" in entry:
                report["ground_info_with_id_field"] += 1
            klass = str(entry.get("class", "")).lower()
            report["classes"][klass] += 1
            caption = str(entry.get("caption", "")).strip()
            captions.append(caption)
            if klass and klass not in caption.lower():
                report["class_string_absent_in_caption"] += 1
            words = class_words(synonyms, dataset, klass)
            found = any(
                re.search(r"\b" + re.escape(word) + r"\b", caption, re.IGNORECASE)
                for word in words
            )
            if found:
                report["loader_class_matched"] += 1
            elif len(report["loader_class_unmatched_examples"]) < 10:
                report["loader_class_unmatched_examples"].append(
                    {"class": klass, "caption": caption}
                )
            if ANY_SPATIAL.search(caption):
                report["caption_with_any_spatial_word"] += 1
            relation_hits = RELATION_WORDS.findall(caption)
            if relation_hits:
                report["caption_with_relation_word"] += 1
            if len({word.lower() for word in relation_hits}) > 1:
                report["caption_multi_relation_word"] += 1

            box = entry.get("bbox_3d")
            if not box:
                continue
            relationship = unit_relationship(box)
            row = relationship["values"]
            if len(report["bbox_rows"][len(row)]) < object_sample_cap:
                report["bbox_rows"][len(row)].append(row)
            center = relationship["center"]
            relation_entries["raw_metadata_frame"].append((caption, center))
            if dataset == "waymo":
                lidar_id = int(frame.split("_")[-1]) if "_" in frame else 0
                angle = WAYMO_VIEW_ANGLES[WAYMO_VIEWS[lidar_id]]
                relation_entries["rotated_by_sensor_view"].append(
                    (caption, rotate_xy(center, angle))
                )
            if pose_matrix is not None:
                homogeneous = np.array([center[0], center[1], center[2], 1.0])
                relation_entries["pose_applied"].append(
                    (caption, [round(float(v), 6) for v in (pose_matrix @ homogeneous)[:3]])
                )
                relation_entries["pose_inverse_applied"].append(
                    (caption, [round(float(v), 6) for v in (pose_inverse @ homogeneous)[:3]])
                )

            nested = entry.get("others") or []
            report["others_per_entry"][len(nested)] += 1
            report["others_total"] += len(nested)
            if entry.get("others_num") not in (None, len(nested)):
                report["others_num_mismatch"] += 1
            for other in nested:
                if not isinstance(other, dict):
                    continue
                for key in other:
                    report["others_entry_keys"][key] += 1
                other_class = str(other.get("class_other", "")).lower()
                if other_class:
                    report["others_classes"][other_class] += 1
                other_box = other.get("bbox_3d_other")
                if other_box and len(report["others_bbox_rows"][len(other_box)]) < object_sample_cap:
                    report["others_bbox_rows"][len(other_box)].append(
                        [float(value) for value in other_box]
                    )

        distinct_captions = {caption for caption in captions if caption}
        if len(distinct_captions) >= 2:
            report["frames_with_two_captions"] += 1
            if len(captions) > len(distinct_captions):
                report["duplicate_caption_frames"] += 1

    for variant, entries in relation_entries.items():
        report["relation_tests"][variant] = relation_test(entries, variant)

    report["bbox_lengths"] = {str(length): len(rows) for length, rows in report["bbox_rows"].items()}
    report["bbox_index_stats"] = {
        str(length): index_stats(rows) for length, rows in report["bbox_rows"].items()
    }
    report["others_bbox_lengths"] = {
        str(length): len(rows) for length, rows in report["others_bbox_rows"].items()
    }
    report["others_bbox_index_stats"] = {
        str(length): index_stats(rows) for length, rows in report["others_bbox_rows"].items()
    }
    report["bbox_rows"] = {str(length): len(rows) for length, rows in report["bbox_rows"].items()}
    report["others_bbox_rows"] = {
        str(length): len(rows) for length, rows in report["others_bbox_rows"].items()
    }
    report["classes"] = describe_counter(report["classes"])
    report["others_classes"] = describe_counter(report["others_classes"])
    report["meta_top_level_keys"] = dict(report["meta_top_level_keys"])
    report["ground_info_entry_keys"] = dict(report["ground_info_entry_keys"])
    report["others_entry_keys"] = dict(report["others_entry_keys"])
    report["others_per_entry"] = dict(sorted(report["others_per_entry"].items()))
    report["pose_shape"] = dict(report["pose_shape"])
    report["caption_samples"] = report["caption_samples"][:5]
    report["ground_info_per_frame"] = dict(sorted(report["ground_info_per_frame"].items()))

    total = report["ground_info_total"] or 1
    report["caption_any_spatial_rate"] = round(
        report["caption_with_any_spatial_word"] / total, 4
    )
    report["caption_relation_word_rate"] = round(
        report["caption_with_relation_word"] / total, 4
    )
    report["class_string_absent_rate"] = round(
        report["class_string_absent_in_caption"] / total, 4
    )
    report["loader_class_match_rate"] = round(report["loader_class_matched"] / total, 4)
    report["frames_with_two_caption_rate"] = round(
        report["frames_with_two_captions"] / max(len(frames), 1), 4
    )

    if lidar_samples:
        points = None
        for sequence, frame in frames[:lidar_samples]:
            extension = ".npy" if dataset == "waymo" else ".bin"
            lidar_path = os.path.join(platform_dir, sequence, frame, f"lidar{extension}")
            if os.path.exists(lidar_path):
                try:
                    points = load_lidar(lidar_path)
                except (OSError, ValueError):
                    points = None
                if points is not None:
                    report["lidar_path_sample"] = os.path.join(sequence, frame, f"lidar{extension}")
                    break
        report["lidar"] = summarize_points(points)
        report["pose_direction_test"] = pose_direction_test(
            report["poses"][:200], report["pose_centers"][:200], points
        )
    report["poses"] = {"count": len(report["poses"])} if report["poses"] else {"count": 0}
    report["pose_centers"] = {"count": len(report["pose_centers"])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/3eed",
                        help="directory holding waymo/, drone/, quad/")
    parser.add_argument("--split-dir", default="data/3eed/splits")
    parser.add_argument("--datasets", nargs="+", default=["waymo", "drone", "quad"])
    parser.add_argument("--splits", nargs="+", default=["val"],
                        choices=["train", "val"],
                        help="split files to merge, e.g. --splits val train")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="frames sampled per platform, 0 means all")
    parser.add_argument("--lidar-samples", type=int, default=1,
                        help="lidar files to read per platform, 0 disables")
    parser.add_argument("--object-sample-cap", type=int, default=5000)
    parser.add_argument("--synonym-file", default="src/joint_det_dataset.py",
                        help="training repo source holding WAYMO_SYNONYMS / "
                             "M3ED_SYNONYMS; used to reproduce the loader's "
                             "class-word match")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="audit_report.json")
    args = parser.parse_args()

    report = {
        "config": {
            "data_root": args.data_root,
            "split_dir": args.split_dir,
            "splits": args.splits,
            "max_frames": args.max_frames,
            "lidar_samples": args.lidar_samples,
            "synonym_file": args.synonym_file,
            "seed": args.seed,
        },
        "platforms": {},
    }
    synonyms = load_synonyms(args.synonym_file)
    report["synonyms_loaded"] = {name: len(table) for name, table in synonyms.items()}
    if not synonyms:
        print(f"note: no synonyms loaded from {args.synonym_file}, "
              f"class matching falls back to the exact class string")
    for dataset in args.datasets:
        print(f"auditing {dataset} ...", flush=True)
        report["platforms"][dataset] = audit_platform(
            args.data_root, args.split_dir, dataset, args.splits, args.max_frames,
            args.seed, args.lidar_samples, args.object_sample_cap, synonyms,
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    for dataset, info in report["platforms"].items():
        print(f"\n=== {dataset} ===")
        if "error" in info:
            print(" ", info["error"])
            continue
        print(f"  frames: {info['frames_scanned']}/{info['frames_available']} "
              f"(missing meta {info['missing_meta']})")
        print(f"  meta keys: {list(info['meta_top_level_keys'])}")
        print(f"  ground_info/frame: {info['ground_info_per_frame']}")
        print(f"  others per entry: {info['others_per_entry']} "
              f"(total {info['others_total']}, num mismatch {info['others_num_mismatch']})")
        print(f"  ground_info entry keys: {list(info['ground_info_entry_keys'])}")
        print(f"  others entry keys: {list(info['others_entry_keys'])}")
        print(f"  bbox lengths: {info['bbox_lengths']} | others: {info['others_bbox_lengths']}")
        print(f"  classes: {list(info['classes'])[:12]}")
        print(f"  others classes: {list(info['others_classes'])[:12]}")
        print(f"  caption spatial-word rate: {info['caption_any_spatial_rate']}, "
              f"relation-word rate: {info['caption_relation_word_rate']}")
        print(f"  class string absent rate: {info['class_string_absent_rate']}, "
              f"loader class-word match rate: {info['loader_class_match_rate']}")
        if info["loader_class_unmatched_examples"]:
            print(f"    unmatched example: {info['loader_class_unmatched_examples'][0]}")
        print(f"  frames with >=2 distinct captions: "
              f"{info['frames_with_two_captions']} ({info['frames_with_two_caption_rate']})")
        print(f"  pose shape: {info['pose_shape']}")
        print(f"  lidar: {info.get('lidar')}")
        print(f"  pose direction: {info.get('pose_direction_test')}")
        for name, test in info.get("relation_tests", {}).items():
            print(f"  {name}: lateral {test['lateral_samples']}, "
                  f"longitudinal {test['longitudinal_samples']}")
            for convention, score in test["convention_score"].items():
                print(f"    {convention}: {score['rate']} on {score['total']} samples")
            word_rates = ", ".join(
                "{}: {}".format(word, score["rate"])
                for word, score in test["per_word"].items()
            )
            agreement = test["dominant_axis_agreement"]
            print("    per word, x_fwd_y_left convention: {}".format(word_rates))
            print("    dominant-axis agreement: {} on {}".format(
                agreement["rate"], agreement["total"]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
