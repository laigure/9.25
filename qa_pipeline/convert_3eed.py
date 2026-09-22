"""Convert raw 3EED metadata into the unified scene JSONL used by generate_qa.py.

The QA frame was fixed by measurement, not by copying the loader (see
RESEARCH_PROGRESS.md and reports/audit_report.json, reports/projection_report.json):

    front = +x, left = +y, up = +z

in the frame the grounding model sees. The converter therefore mirrors the
training loader exactly, because QA geometry must live in the same frame as the
model's input:

  * waymo: rotate boxes (and, for a consumer, the raw points) about z by the
    sensor-view angle {F: 0, FL: -45, FR: 45, SL: -90, SR: 90}, as
    utils/transform_waymo.py does. The caption relation vote drops from 0.9488
    on this frame back to a coin flip on the raw frame, so the rotation is
    mandatory.
  * drone: shift z by +1.8 m on points and boxes, as joint_det_dataset.py does.
  * quad: no transform; the pose conversion stays unused, exactly as in the
    loader. Caption votes degrade when pose is applied, and pose maps ego to
    world, which would destroy the ego-centric relation frame.
  * pose is never applied on any platform.

Boxes written here are already transformed. `frame.points_transform` tells a
consumer what to apply to the raw cloud at `point_cloud_path` so points and
boxes stay in one frame.

Raw schema notes (measured on all 20,367 frames):
  * `ground_info` entries: class, caption, bbox_3d, bbox_2d_proj, others,
    others_num. `others` nests inside each entry; entries are 7-dim on waymo,
    9-dim on drone/quad where only the first 7 are used. Layout is
    [x, y, z, length, width, height, yaw, ...]; yaw is radians, already wrapped
    on waymo but unwrapped on drone/quad, so it is wrapped here.
  * There is no object id anywhere: every id below is generated and marked.
  * `bbox_2d_proj` is xyxy in image pixels.
  * `others` boxes carry no caption and no stable identity; the same physical
    object repeats across a frame's entries. They are deduplicated by
    (class, center to 1 cm) into role "context" records, and a context record
    that coincides with a referred object is linked rather than dropped.

    python convert_3eed.py --data-root data/3eed --split-dir data/3eed/splits \
        --splits train val --out /root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl
"""

import argparse
import ast
import json
import math
import os
import re
from collections import Counter

WAYMO_VIEWS = ["F", "FL", "FR", "SL", "SR"]
WAYMO_VIEW_ANGLES = {"F": 0.0, "FL": -45.0, "FR": 45.0, "SL": -90.0, "SR": 90.0}

DRONE_HEIGHT_OFFSET_M = 1.8
DUPLICATE_CENTER_M = 2.0

RELATION_WORDS = ("left", "right", "front", "behind")

FRAME_NAMES = {
    "waymo": "waymo_rotated_sensor_view",
    "drone": "drone_raw_sensor_frame",
    "quad": "quad_raw_sensor_frame",
}

# front = +x, left = +y, up = +z, therefore right = -y.
FRAME_AXES = {
    "right_axis": [0.0, -1.0, 0.0],
    "forward_axis": [1.0, 0.0, 0.0],
    "up_axis": [0.0, 0.0, 1.0],
}


def view_angle_deg(frame_dir):
    """The loader maps int(scan_id.split('_')[-1]) through WAYMO_VIEWS."""
    try:
        index = int(frame_dir.split("_")[-1])
    except ValueError:
        return None
    if not 0 <= index < len(WAYMO_VIEWS):
        return None
    return WAYMO_VIEW_ANGLES[WAYMO_VIEWS[index]]


def rotate_xy(x, y, angle_deg):
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return cos_a * x - sin_a * y, sin_a * x + cos_a * y


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def load_synonyms(path):
    """Read WAYMO_SYNONYMS / M3ED_SYNONYMS with ast (no torch import)."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("SYNONYMS"):
                    try:
                        tables[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return tables or None


def synonyms_for(tables, platform):
    """The loader uses WAYMO_SYNONYMS for waymo and M3ED_SYNONYMS otherwise."""
    if not tables:
        return None
    name = "WAYMO_SYNONYMS" if platform == "waymo" else "M3ED_SYNONYMS"
    return tables.get(name)


def loader_class_matched(caption, category, synonyms):
    """Reproduce the loader's synonym match on a formatted caption."""
    if synonyms is None:
        return None
    words = synonyms.get(category, [category]) if isinstance(synonyms, dict) else [category]
    text = " " + " ".join(caption.replace(",", " ,").split()) + " "
    for word in words:
        if re.search(r"\b{}\b".format(re.escape(word)), text, flags=re.IGNORECASE):
            return True
    return False


def spatial_word_counts(caption):
    counts = Counter()
    for word in RELATION_WORDS:
        counts[word] = len(re.findall(r"\b{}\b".format(word), caption, flags=re.IGNORECASE))
    return {word: counts[word] for word in RELATION_WORDS if counts[word]}


def parse_box(raw, platform, frame_dir):
    """Return a gt_box dict in the QA frame, or raise ValueError."""
    if not isinstance(raw, list) or len(raw) < 7:
        raise ValueError("bbox_3d must have at least 7 values")
    values = [float(v) for v in raw]
    x, y, z = values[0], values[1], values[2]
    length, width, height, yaw = values[3], values[4], values[5], values[6]
    angle = view_angle_deg(frame_dir)
    if platform == "waymo":
        if angle is None:
            raise ValueError("unparsable waymo view id: {}".format(frame_dir))
        x, y = rotate_xy(x, y, angle)
        yaw += math.radians(angle)
    elif platform == "drone":
        z += DRONE_HEIGHT_OFFSET_M
    if not all(math.isfinite(v) for v in (x, y, z, length, width, height, yaw)):
        raise ValueError("non-finite box value")
    if min(length, width, height) <= 0:
        raise ValueError("nonpositive box size")
    return {
        "center": [round(x, 4), round(y, 4), round(z, 4)],
        "size": [round(length, 4), round(width, 4), round(height, 4)],
        "yaw_rad": round(wrap_angle(yaw), 4),
        "yaw_raw_rad": round(float(values[6]), 4),
    }


def parse_split_map(split_dir, dataset, splits, conflicts):
    mapping = {}
    for split in splits:
        path = os.path.join(split_dir, "{}_{}.txt".format(dataset, split))
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                sequence = line.strip()
                if not sequence:
                    continue
                if sequence in mapping and mapping[sequence] != split:
                    conflicts.append([dataset, sequence, mapping[sequence], split])
                    mapping[sequence] = "val"  # never let a frame be both
                else:
                    mapping[sequence] = split
    return mapping


def frame_paths(data_root, platform, sequence, frame_dir):
    base = os.path.join(platform, sequence, frame_dir)
    return {
        "point_cloud_path": os.path.join(base, "lidar.npy" if platform == "waymo" else "lidar.bin"),
        "image_path": os.path.join(base, "image.jpg"),
        "metadata_path": os.path.join(base, "meta_info.json"),
    }


def convert_frame(data_root, platform, sequence, frame_dir, split, synonyms, stats):
    paths = frame_paths(data_root, platform, sequence, frame_dir)
    meta_path = os.path.join(data_root, paths["metadata_path"])
    try:
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        stats["skipped"]["unreadable_meta"] += 1
        return None
    entries = meta.get("ground_info") or []
    if not entries:
        stats["skipped"]["no_ground_info"] += 1
        return None

    scene_id = "{}/{}/{}".format(platform, sequence, frame_dir)
    frame_name = FRAME_NAMES[platform]
    angle = view_angle_deg(frame_dir) if platform == "waymo" else None
    if platform == "waymo" and angle is None:
        stats["skipped"]["unparsable_waymo_view"] += 1
        return None
    points_transform = (
        {"rot_z_deg": angle} if platform == "waymo"
        else {"translate_z_m": DRONE_HEIGHT_OFFSET_M} if platform == "drone"
        else None
    )
    scene = {
        "scene_id": scene_id,
        "split": split,
        "platform": platform,
        "sequence": sequence,
        "point_cloud_path": paths["point_cloud_path"],
        "image_path": paths["image_path"],
        "metadata_path": paths["metadata_path"],
        "frame": {
            "name": frame_name,
            "units": "m",
            "origin": "sensor",
            "convention": {"front": "+x", "left": "+y", "up": "+z"},
            "points_transform": points_transform,
            "boxes_already_transformed": True,
        },
        "objects": [],
    }
    scene["frame"].update(FRAME_AXES)

    if platform == "waymo":
        scene["frame"]["view_angle_deg"] = angle

    platform_synonyms = synonyms_for(synonyms, platform)
    objects = scene["objects"]
    for index, entry in enumerate(entries):
        category = str(entry.get("class", "")).lower()
        caption = str(entry.get("caption") or "").strip()
        try:
            box = parse_box(entry.get("bbox_3d"), platform, frame_dir)
        except (TypeError, ValueError):
            stats["skipped"]["bad_box"] += 1
            continue
        box["frame"] = frame_name
        box["bbox_3d_raw"] = [round(float(v), 4) for v in entry["bbox_3d"]]
        box["bbox_2d_proj"] = (
            [round(float(v), 2) for v in entry["bbox_2d_proj"]]
            if entry.get("bbox_2d_proj") else None
        )
        matched = loader_class_matched(caption, category, platform_synonyms) if caption else None
        objects.append({
            "object_id": "{}#g{}".format(scene_id, index),
            "id_source": "generated",
            "role": "referred",
            "source": "ground_info",
            "annotation_index": index,
            "category": category,
            "loader_visible": matched,
            "referring_expressions": (
                [{"text": caption, "source": "ground_info.caption", "language": "en"}]
                if caption else []
            ),
            "spatial_word_counts": spatial_word_counts(caption) if caption else {},
            "gt_box": box,
        })
        stats["class_counts"][category] += 1
        if matched:
            stats["loader_visible"] += 1
        if caption:
            stats["captions"] += 1
            if spatial_word_counts(caption):
                stats["captions_with_relation_word"] += 1

    referred_count = len(objects)
    if referred_count >= 2:
        stats["frames_with_two_captions"] += 1

    # Context objects: deduplicate the per-entry `others` lists into records.
    context = {}
    for index, entry in enumerate(entries):
        for other_index, other in enumerate(entry.get("others") or []):
            category = str(other.get("class_other", "")).lower()
            try:
                box = parse_box(other.get("bbox_3d_other"), platform, frame_dir)
            except (TypeError, ValueError):
                stats["skipped"]["bad_other_box"] += 1
                continue
            key = (category, round(box["center"][0], 2), round(box["center"][1], 2),
                   round(box["center"][2], 2))
            record = context.get(key)
            if record is None:
                box["frame"] = frame_name
                box["bbox_3d_raw"] = [round(float(v), 4) for v in other["bbox_3d_other"]]
                box["bbox_2d_proj"] = None
                record = {
                    "object_id": "{}#c{}".format(scene_id, len(context)),
                    "id_source": "generated",
                    "role": "context",
                    "source": "others",
                    "annotation_index": None,
                    "category": category,
                    "loader_visible": None,
                    "referring_expressions": [],
                    "spatial_word_counts": {},
                    "source_refs": [],
                    "gt_box": box,
                }
                context[key] = record
                objects.append(record)
            record["source_refs"].append({"entry_index": index, "other_index": other_index})

    for record in list(context.values()):
        center = record["gt_box"]["center"]
        nearest = None
        for referred in objects[:referred_count]:
            if referred["category"] != record["category"]:
                continue
            other_center = referred["gt_box"]["center"]
            distance = math.dist(center, other_center)
            if distance <= DUPLICATE_CENTER_M and (nearest is None or distance < nearest[1]):
                nearest = (referred["object_id"], distance)
        if nearest is not None:
            record["probable_duplicate_of_referred"] = {
                "object_id": nearest[0], "center_distance_m": round(nearest[1], 3)}
            stats["context_duplicates_linked"] += 1

    stats["scenes"] += 1
    stats["referred_objects"] += referred_count
    stats["context_objects"] += len(context)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/3eed")
    parser.add_argument("--split-dir", default="data/3eed/splits")
    parser.add_argument("--datasets", nargs="+", default=["waymo", "drone", "quad"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--synonym-file", default="src/joint_det_dataset.py",
                        help="source file with WAYMO_SYNONYMS/M3ED_SYNONYMS")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="limit frames per platform (0 = all)")
    parser.add_argument("--out", default="3eed_scenes.jsonl")
    parser.add_argument("--stats-out", default="3eed_scenes_stats.json")
    args = parser.parse_args()

    synonyms = load_synonyms(args.synonym_file)
    if synonyms is None:
        print("warning: synonyms not loaded from", args.synonym_file,
              "- loader_visible will be null")

    per_platform = {}
    conflicts = []
    total = 0
    with open(args.out, "w", encoding="utf-8") as sink:
        for platform in args.datasets:
            platform_dir = os.path.join(args.data_root, platform)
            if not os.path.isdir(platform_dir):
                print("skipping missing platform dir:", platform_dir)
                continue
            stats = {
                "scenes": 0, "referred_objects": 0, "context_objects": 0,
                "loader_visible": 0, "captions": 0, "captions_with_relation_word": 0,
                "frames_with_two_captions": 0, "context_duplicates_linked": 0,
                "class_counts": Counter(), "skipped": Counter(),
            }
            split_map = parse_split_map(args.split_dir, platform, args.splits, conflicts)
            frames = []
            for sequence, split in sorted(split_map.items()):
                sequence_dir = os.path.join(platform_dir, sequence)
                if not os.path.isdir(sequence_dir):
                    stats["skipped"]["sequence_dir_missing"] += 1
                    continue
                for frame_dir in sorted(os.listdir(sequence_dir)):
                    if os.path.isdir(os.path.join(sequence_dir, frame_dir)):
                        frames.append((sequence, frame_dir, split))
            stats["frames_available"] = len(frames)
            stats["frames_outside_split"] = 0
            if args.max_frames:
                frames = frames[:args.max_frames]
            for sequence, frame_dir, split in frames:
                scene = convert_frame(args.data_root, platform, sequence, frame_dir,
                                      split, synonyms, stats)
                if scene is None:
                    continue
                sink.write(json.dumps(scene, ensure_ascii=False) + "\n")
                total += 1
                if total % 2000 == 0:
                    print("wrote", total, "scenes", flush=True)
            stats["class_counts"] = dict(stats["class_counts"].most_common())
            stats["skipped"] = dict(stats["skipped"])
            per_platform[platform] = stats
            print("{}: {} scenes, {} referred, {} context".format(
                platform, stats["scenes"], stats["referred_objects"],
                stats["context_objects"]), flush=True)

    report = {"config": vars(args), "split_conflicts": conflicts,
              "platforms": per_platform,
              "totals": {
                  "scenes": sum(s["scenes"] for s in per_platform.values()),
                  "referred_objects": sum(s["referred_objects"] for s in per_platform.values()),
                  "context_objects": sum(s["context_objects"] for s in per_platform.values()),
              }}
    with open(args.stats_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("wrote {} scenes to {} and stats to {}".format(total, args.out, args.stats_out))


if __name__ == "__main__":
    main()
