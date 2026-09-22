"""Verify the 3EED frame convention with the camera projection in metadata.

`audit_3eed.py` fixes the lateral axis from caption words (left = +y on all
three platforms) but barely tests the longitudinal sign, because captions
almost never mention front/behind: only a handful of samples carry a
longitudinal word, so front = +x and front = -x score the same.

This script closes that gap without using language at all. Every frame stores
`image_intrinsic`, `image_extrinsic`, and a `bbox_2d_proj` for the referred
object. Projecting the 3D box center with the correct transform has to land
inside the annotated 2D box, which pins the direction of the extrinsic. The
camera axes carried by that transform then name the forward direction of the
box frame.

For waymo the axes are reported twice: in the raw metadata frame and after the
sensor-view rotation the model applies, because the caption test showed the
rotated frame is the one the language agrees with.

    python audit_projection.py --data-root data/3eed --split-dir data/3eed/splits
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np


WAYMO_VIEWS = ["F", "FL", "FR", "SL", "SR"]
WAYMO_VIEW_ANGLES = {"F": 0.0, "FL": -45.0, "FR": 45.0, "SL": -90.0, "SR": 90.0}


def view_rotation_deg(frame):
    if "_" not in frame:
        return 0.0
    index = int(frame.split("_")[-1])
    if not 0 <= index < len(WAYMO_VIEWS):
        return None
    return WAYMO_VIEW_ANGLES[WAYMO_VIEWS[index]]


def rotate_z(vector, angle_deg):
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return np.array([cos_a * x - sin_a * y, sin_a * x + cos_a * y, z])


def project(point_cam, intrinsic):
    """Project a camera-frame point with a 3x3 or 3x4 intrinsic."""
    if intrinsic.shape == (3, 4):
        projected = intrinsic @ np.append(point_cam, 1.0)
    else:
        projected = intrinsic @ point_cam
    if abs(projected[2]) < 1e-9:
        return None
    return projected[:2] / projected[2]


def inside_2d(point, box_2d, mode):
    if point is None or box_2d is None or len(box_2d) != 4:
        return None
    x, y = float(point[0]), float(point[1])
    a, b, c, d = (float(v) for v in box_2d)
    if mode == "xyxy":
        return a <= x <= c and b <= y <= d
    return a <= x <= a + c and b <= y <= b + d


def pearson(pairs):
    if len(pairs) < 3:
        return None
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    if xs.std() < 1e-9 or ys.std() < 1e-9:
        return None
    return round(float(np.corrcoef(xs, ys)[0, 1]), 4)


def describe_axis(vector, tolerance=15.0):
    """Name the axis a vector aligns with, or None when it is oblique."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return "degenerate"
    unit = vector / norm
    index = int(np.argmax(np.abs(unit)))
    name = ("x", "y", "z")[index]
    sign = "+" if unit[index] > 0 else "-"
    angle = math.degrees(math.acos(min(1.0, abs(float(unit[index])))))
    if angle > tolerance:
        return "oblique({}{}, {:.1f}deg)".format(sign, name, angle)
    return "{}{}".format(sign, name)


def audit(data_root, split_dir, dataset, splits, max_frames, seed, examples_wanted):
    platform_dir = os.path.join(data_root, dataset)
    sequences = []
    for split in splits:
        path = os.path.join(split_dir, f"{dataset}_{split}.txt")
        if os.path.exists(path):
            with open(path) as handle:
                sequences += [line.strip() for line in handle if line.strip()]
    frames = sorted({
        (sequence, frame)
        for sequence in set(sequences)
        if os.path.isdir(os.path.join(platform_dir, sequence))
        for frame in os.listdir(os.path.join(platform_dir, sequence))
        if os.path.isdir(os.path.join(platform_dir, sequence, frame))
    })
    if max_frames and len(frames) > max_frames:
        frames = random.Random(seed).sample(frames, max_frames)
        frames.sort()

    hypotheses = ("extrinsic_maps_box_to_camera", "inverse_maps_box_to_camera")
    report = {
        "frames_scanned": len(frames),
        "intrinsic_shapes": Counter(),
        "extrinsic_translation_norm_median": None,
        "skipped": Counter(),
        "hypotheses": {
            name: {"depth_values": [], "inside_xyxy": [], "inside_xywh": []}
            for name in hypotheses
        },
        "camera_axes": defaultdict(lambda: defaultdict(list)),
        "image_axis_correlation": defaultdict(list),
        "examples": [],
    }
    translation_norms = []

    for sequence, frame in frames:
        meta_path = os.path.join(platform_dir, sequence, frame, "meta_info.json")
        if not os.path.exists(meta_path):
            report["skipped"]["missing_meta"] += 1
            continue
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            report["skipped"]["unreadable_meta"] += 1
            continue
        entry = (meta.get("ground_info") or [None])[0]
        if not entry or entry.get("bbox_3d") is None or entry.get("bbox_2d_proj") is None:
            report["skipped"]["no_box"] += 1
            continue
        if meta.get("image_extrinsic") is None or meta.get("image_intrinsic") is None:
            report["skipped"]["no_camera"] += 1
            continue

        extrinsic = np.asarray(meta["image_extrinsic"], dtype=np.float64)
        intrinsic = np.asarray(meta["image_intrinsic"], dtype=np.float64)
        report["intrinsic_shapes"]["{}x{}".format(*intrinsic.shape)] += 1
        if extrinsic.shape != (4, 4) or intrinsic.shape not in ((3, 3), (3, 4)):
            report["skipped"]["bad_shape"] += 1
            continue
        try:
            extrinsic_inverse = np.linalg.inv(extrinsic)
        except np.linalg.LinAlgError:
            report["skipped"]["singular_extrinsic"] += 1
            continue

        center = np.array([float(v) for v in entry["bbox_3d"][:3]])
        box_2d = [float(v) for v in entry["bbox_2d_proj"]]
        homogeneous = np.append(center, 1.0)
        translation_norms.append(float(np.linalg.norm(extrinsic[:3, 3])))

        for name, box_to_camera in (("extrinsic_maps_box_to_camera", extrinsic),
                                    ("inverse_maps_box_to_camera", extrinsic_inverse)):
            point_cam = (box_to_camera @ homogeneous)[:3]
            point_2d = project(point_cam, intrinsic)
            report["hypotheses"][name]["depth_values"].append(float(point_cam[2]))
            for mode in ("xyxy", "xywh"):
                verdict = inside_2d(point_2d, box_2d, mode)
                if verdict is not None:
                    report["hypotheses"][name]["inside_" + mode].append(1.0 if verdict else 0.0)

            rotation = box_to_camera[:3, :3]
            # Rows of an orthonormal transform are the target (camera) basis
            # expressed in the source (box) frame.
            for kind, axis in (("image_right", rotation[0]),
                               ("image_up", rotation[1]),
                               ("optical_axis", rotation[2])):
                report["camera_axes"][name]["raw:" + kind].append(axis)
                angle = view_rotation_deg(frame) if dataset == "waymo" else None
                if angle is not None:
                    report["camera_axes"][name]["view_rotated:" + kind].append(
                        rotate_z(axis, angle))

        # Independent of any extrinsic: how the annotated 2D box moves with the
        # 3D box. For a forward-looking camera, image u (right) grows towards
        # -y and image v (down) grows towards nearer objects, so both
        # correlations below are expected to be negative.
        box_2d_values = [float(v) for v in box_2d]
        u = 0.5 * (box_2d_values[0] + box_2d_values[2])
        v = 0.5 * (box_2d_values[1] + box_2d_values[3])
        angle = view_rotation_deg(frame) if dataset == "waymo" else None
        rotated = rotate_z(center, angle) if angle is not None else None
        report["image_axis_correlation"]["u_vs_y_raw"].append((u, float(center[1])))
        report["image_axis_correlation"]["v_vs_x_raw"].append((v, float(center[0])))
        if rotated is not None:
            report["image_axis_correlation"]["u_vs_y_view_rotated"].append(
                (u, float(rotated[1])))
            report["image_axis_correlation"]["v_vs_x_view_rotated"].append(
                (v, float(rotated[0])))

        if len(report["examples"]) < examples_wanted:
            example = {
                "frame": os.path.join(sequence, frame),
                "class": entry.get("class"),
                "box_3d": [round(float(v), 3) for v in entry["bbox_3d"]],
                "box_2d_proj": [round(v, 2) for v in box_2d],
                "view_rotation_deg": view_rotation_deg(frame) if dataset == "waymo" else None,
            }
            for name, box_to_camera in (("extrinsic_maps_box_to_camera", extrinsic),
                                        ("inverse_maps_box_to_camera", extrinsic_inverse)):
                point_cam = (box_to_camera @ homogeneous)[:3]
                point_2d = project(point_cam, intrinsic)
                example[name] = {
                    "depth": round(float(point_cam[2]), 3),
                    "xy": [round(float(v), 2) for v in point_2d] if point_2d is not None else None,
                    "inside_xyxy": inside_2d(point_2d, box_2d, "xyxy"),
                    "inside_xywh": inside_2d(point_2d, box_2d, "xywh"),
                }
            report["examples"].append(example)

    if translation_norms:
        report["extrinsic_translation_norm_median"] = round(
            float(np.median(translation_norms)), 3)

    for name, values in report["hypotheses"].items():
        depth = values.pop("depth_values")
        if depth:
            values["depth_positive_rate"] = round(sum(1 for d in depth if d > 0) / len(depth), 4)
            values["depth_median"] = round(float(np.median(depth)), 3)
            values["samples"] = len(depth)
        for mode in ("xyxy", "xywh"):
            samples = values.pop("inside_" + mode)
            if samples:
                values["inside_" + mode + "_rate"] = round(sum(samples) / len(samples), 4)

    camera_axes = {}
    for name, kinds in report["camera_axes"].items():
        camera_axes[name] = {}
        for kind, samples in kinds.items():
            stacked = np.stack(samples)
            mean_axis = stacked.mean(axis=0)
            norm = float(np.linalg.norm(mean_axis))
            # The per-sample histogram is reported instead of a sign-aligned
            # mean: if frames disagree about the direction, that has to stay
            # visible rather than be averaged away.
            histogram = Counter(describe_axis(row) for row in stacked)
            agreement = None
            if norm > 1e-6:
                unit_mean = mean_axis / norm
                agreement = round(
                    float(np.mean(np.dot(stacked, unit_mean) > 0)), 4)
            camera_axes[name][kind] = {
                "dominant": describe_axis(mean_axis),
                "mean": [round(float(v), 3) for v in (mean_axis / max(norm, 1e-9))],
                "mean_norm": round(norm / len(stacked), 3),
                "agreement_with_mean": agreement,
                "histogram": dict(histogram.most_common(6)),
                "samples": len(samples),
            }
    report["camera_axes"] = camera_axes

    correlation = {}
    for name, pairs in report["image_axis_correlation"].items():
        value = pearson(pairs)
        if value is None:
            continue
        correlation[name] = {
            "pearson_r": value,
            "samples": len(pairs),
            "reading": (
                "left = +y" if name.startswith("u_vs_y") and value < 0
                else "left = -y" if name.startswith("u_vs_y")
                else "front = +x" if value < 0
                else "front = -x"
            ),
        }
    report["image_axis_correlation"] = correlation
    report["intrinsic_shapes"] = dict(report["intrinsic_shapes"])
    report["skipped"] = dict(report["skipped"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/3eed")
    parser.add_argument("--split-dir", default="data/3eed/splits")
    parser.add_argument("--datasets", nargs="+", default=["waymo", "drone", "quad"])
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=2)
    parser.add_argument("--out", default="projection_report.json")
    args = parser.parse_args()

    report = {}
    for dataset in args.datasets:
        print("projection check:", dataset, flush=True)
        report[dataset] = audit(args.data_root, args.split_dir, dataset, args.splits,
                                args.max_frames, args.seed, args.examples)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    for dataset, info in report.items():
        print("\n===", dataset, "===")
        print("  frames: {} | intrinsic {} | extrinsic |t| median {} | skipped {}".format(
            info["frames_scanned"], info["intrinsic_shapes"],
            info["extrinsic_translation_norm_median"], info["skipped"]))
        for name, values in info["hypotheses"].items():
            print("  {}: depth>0 {} (median {}), inside xyxy {} / xywh {}".format(
                name, values.get("depth_positive_rate"), values.get("depth_median"),
                values.get("inside_xyxy_rate"), values.get("inside_xywh_rate")))
        for name, kinds in info["camera_axes"].items():
            for kind, value in kinds.items():
                print("  {} {}: {} mean {} agree {} hist {}".format(
                    name, kind, value["dominant"], value["mean"],
                    value["agreement_with_mean"], value["histogram"]))
        for name, value in info["image_axis_correlation"].items():
            print("  image/box correlation {}: r={} on {} samples -> {}".format(
                name, value["pearson_r"], value["samples"], value["reading"]))
        for example in info["examples"]:
            print("  example {} {} box2d {} view_rot {}".format(
                example["frame"], example["class"], example["box_2d_proj"],
                example["view_rotation_deg"]))
            print("     box_3d", example["box_3d"])
            for name in ("extrinsic_maps_box_to_camera", "inverse_maps_box_to_camera"):
                print("     {} -> {}".format(name, example[name]))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
