"""Generate geometry-grounded spatial QA from normalized scene JSONL.

Only GT boxes are read. Predicted boxes and model features never enter labels.
"""

import argparse
import json
import math
import re
from pathlib import Path


SPATIAL_WORDS = re.compile(
    r"\b(left|right|front|behind|back|near|far|closest|farthest|between|"
    r"above|below|next to|beside|adjacent|opposite|first|last)\b",
    re.IGNORECASE,
)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(vector, name):
    if len(vector) != 3 or not all(isinstance(x, (int, float)) for x in vector):
        raise ValueError(f"{name} must be a 3D numeric vector")
    norm = math.sqrt(dot(vector, vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"{name} must be a unit vector")
    return vector


def validate_scene(scene):
    for field in ("scene_id", "split", "platform", "point_cloud_path", "frame", "objects"):
        if field not in scene:
            raise ValueError(f"missing scene field: {field}")
    if scene["split"] not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    frame = scene["frame"]
    if frame.get("units") != "m":
        raise ValueError("frame.units must be m")
    right = unit(frame["right_axis"], "right_axis")
    forward = unit(frame["forward_axis"], "forward_axis")
    up = unit(frame["up_axis"], "up_axis")
    if any(abs(dot(a, b)) > 0.01 for a, b in
           ((right, forward), (right, up), (forward, up))):
        raise ValueError("frame axes must be orthogonal")
    if not frame.get("name"):
        raise ValueError("frame.name is required")
    ids = set()
    for obj in scene["objects"]:
        for field in ("object_id", "category", "gt_box", "referring_expressions"):
            if field not in obj:
                raise ValueError(f"missing object field: {field}")
        if obj["object_id"] in ids:
            raise ValueError(f"duplicate object_id: {obj['object_id']}")
        ids.add(obj["object_id"])
        box = obj["gt_box"]
        for key in ("center", "size"):
            if len(box[key]) != 3 or not all(math.isfinite(float(x)) for x in box[key]):
                raise ValueError(f"invalid {key} for {obj['object_id']}")
        if any(float(x) <= 0 for x in box["size"]):
            raise ValueError(f"nonpositive size for {obj['object_id']}")
        if not math.isfinite(float(box["yaw_rad"])):
            raise ValueError(f"invalid yaw_rad for {obj['object_id']}")
        if box.get("frame") != frame["name"]:
            raise ValueError(f"box frame mismatch for {obj['object_id']}")


def safe_expression(obj):
    for entry in obj["referring_expressions"]:
        text = entry["text"] if isinstance(entry, dict) else entry
        text = text.strip().rstrip(".")
        if text and not SPATIAL_WORDS.search(text):
            return text
    return None


def generate_for_scene(scene, min_separation=0.5, axis_margin=0.25):
    validate_scene(scene)
    right = scene["frame"]["right_axis"]
    forward = scene["frame"]["forward_axis"]
    objects = scene["objects"]
    for target in objects:
        target_name = safe_expression(target)
        if target_name is None:
            continue
        for reference in objects:
            if target["object_id"] == reference["object_id"]:
                continue
            reference_name = safe_expression(reference)
            if reference_name is None:
                continue
            delta = [float(a) - float(b) for a, b in zip(
                target["gt_box"]["center"], reference["gt_box"]["center"]
            )]
            planar_distance = math.hypot(dot(delta, right), dot(delta, forward))
            if planar_distance < min_separation:
                continue
            lateral = dot(delta, right)
            longitudinal = dot(delta, forward)
            # Only claim a dominant direction. Diagonal and nearly tied pairs
            # need a more expressive relation vocabulary and are skipped here.
            if abs(lateral) > abs(longitudinal) + axis_margin:
                relation = "right" if lateral > 0 else "left"
            elif abs(longitudinal) > abs(lateral) + axis_margin:
                relation = "front" if longitudinal > 0 else "behind"
            else:
                continue
            phrase = {
                "left": "to the left of",
                "right": "to the right of",
                "front": "in front of",
                "behind": "behind",
            }[relation]
            yield {
                "qa_id": f"{scene['scene_id']}:{target['object_id']}:{reference['object_id']}",
                "scene_id": scene["scene_id"],
                "split": scene["split"],
                "platform": scene["platform"],
                "target_object_id": target["object_id"],
                "reference_object_id": reference["object_id"],
                "question": f"Where is {target_name} relative to {reference_name}?",
                "answer": f"{target_name.capitalize()} is {phrase} {reference_name}, approximately {planar_distance:.1f} meters away.",
                "relation": relation,
                "distance_m": round(planar_distance, 3),
                "distance_kind": "center_to_center_bev",
                "label_source": "ground_truth_geometry",
                "frame": scene["frame"]["name"],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="normalized scene JSONL")
    parser.add_argument("output", type=Path, help="QA JSONL")
    args = parser.parse_args()
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                for qa in generate_for_scene(json.loads(line)):
                    sink.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    count += 1
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid scene on line {number}: {error}") from error
    print(f"wrote {count} QA samples to {args.output}")


if __name__ == "__main__":
    main()
