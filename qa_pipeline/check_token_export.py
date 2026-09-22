"""Read-only audit of the exported grounding token NPZ files, plus a join check
against the unified scene JSONL.

Answers, per platform: row/file counts, token dimension, NaN/Inf, token L2-norm
distribution, match_score distribution, center-error and IoU-based success
rates, the observed query-index range, and how many scenes hold two or more
token rows (the pool the two-object QA pairs can come from).

The join check verifies the key that ties a token row to a scene object:
`meta_path` (strip the "data/3eed/" prefix) picks the frame, and the caption
text after the same normalization the dataset applies identifies the object,
because captions are unique within a frame (the audit showed zero duplicate
caption frames). It also confirms that the NPZ `gt_box` lives in the same
coordinate frame as the scene JSONL boxes.

    python check_token_export.py --tokens-root /root/autodl-tmp/3eed_data \
        --scenes /root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl \
        --out /root/autodl-tmp/3eed_data/scenes/token_export_check.json
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/3eedqa/3EED")


def normalize_caption(text):
    """The dataset's _format_caption, minus the sampled ' . not mentioned' tail."""
    return " ".join(str(text).replace(",", " ,").split())


def scene_key_from_meta(meta_path):
    parts = str(meta_path).split("/")
    if parts[:2] == ["data", "3eed"]:
        parts = parts[2:]
    return "/".join(parts[:-1])  # drops meta_info.json, keeps platform/seq/frame


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(array.min()), 4),
        "p05": round(float(np.percentile(array, 5)), 4),
        "median": round(float(np.median(array)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "max": round(float(array.max()), 4),
    }


def iou_fn():
    try:
        from utils.eval_det import iou3d_rotated_vs_aligned
        return iou3d_rotated_vs_aligned
    except Exception as error:  # keep the audit running without the repo import
        print("warning: IoU function unavailable:", error)
        return None


def audit_platform(platform, tokens_root, iou, scenes_index, report):
    pattern = os.path.join(tokens_root, "eval_{}_tokens".format(platform), "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))
    info = {
        "files": len(files), "rows": 0, "token_dim_values": Counter(),
        "tokens_with_nan": 0, "tokens_with_inf": 0,
        "query_index_min": None, "query_index_max": None,
        "scenes": 0, "scenes_with_two_or_more_rows": 0,
        "rows_per_scene": Counter(),
        "join": {"matched_unique": 0, "ambiguous": 0, "no_scene": 0, "no_caption": 0,
                 "center_diff_scene_vs_npz_m": []},
        "scenes_with_two_or_more_matched": 0,
    }
    l2_norms, scores, center_errors, ious = [], [], [], []
    rows_per_scene = Counter()
    matched_scenes = Counter()

    for path in files:
        data = np.load(path, allow_pickle=True)
        tokens = data["object_token"]
        info["rows"] += len(tokens)
        info["token_dim_values"].update([str(tokens.shape[1:])])
        info["tokens_with_nan"] += int(np.isnan(tokens).any(axis=1).sum())
        info["tokens_with_inf"] += int(np.isinf(tokens).any(axis=1).sum())
        l2_norms.extend(np.linalg.norm(tokens, axis=1).tolist())
        scores.extend(data["match_score"].tolist())
        centers = data["pred_center"] - data["gt_box"][:, :3]
        center_errors.extend(np.linalg.norm(centers, axis=1).tolist())
        indices = data["query_index"]
        info["query_index_min"] = int(indices.min()) if info["query_index_min"] is None \
            else min(info["query_index_min"], int(indices.min()))
        info["query_index_max"] = int(indices.max()) if info["query_index_max"] is None \
            else max(info["query_index_max"], int(indices.max()))

        if iou is not None:
            for row in range(len(tokens)):
                gt = torch.tensor(data["gt_box"][row][None, :], dtype=torch.float32)
                pred = torch.tensor(
                    np.concatenate([data["pred_center"][row], data["pred_size"][row]])[None, :],
                    dtype=torch.float32)
                ious.append(float(iou(gt, pred)[0][0, 0]))

        for row in range(len(tokens)):
            key = scene_key_from_meta(data["meta_path"][row])
            rows_per_scene[key] += 1
            caption = normalize_caption(data["utterance"][row])
            if caption.endswith(" . not mentioned"):
                caption = caption[:-len(" . not mentioned")]
            scene = scenes_index.get(key)
            if scene is None:
                info["join"]["no_scene"] += 1
                continue
            matches = [obj for obj in scene
                       if normalize_caption(obj["caption"]) == caption]
            if not matches:
                info["join"]["no_caption"] += 1
                continue
            if len(matches) > 1:
                info["join"]["ambiguous"] += 1
                continue
            info["join"]["matched_unique"] += 1
            matched_scenes[key] += 1
            npz_center = np.asarray(data["gt_box"][row][:3], dtype=np.float64)
            scene_center = np.asarray(matches[0]["center"], dtype=np.float64)
            info["join"]["center_diff_scene_vs_npz_m"].append(
                float(np.linalg.norm(npz_center - scene_center)))

    info["scenes"] = len(rows_per_scene)
    info["scenes_with_two_or_more_rows"] = sum(1 for n in rows_per_scene.values() if n >= 2)
    info["scenes_with_two_or_more_matched"] = sum(1 for n in matched_scenes.values() if n >= 2)
    info["rows_per_scene"] = dict(sorted(rows_per_scene.items(), key=lambda kv: -kv[1])[:1])
    info["rows_per_scene"] = dict(Counter(rows_per_scene.values()))
    info["l2_norm"] = quantiles(l2_norms)
    info["match_score"] = quantiles(scores)
    info["match_score_ge_0.9_rate"] = round(
        float(np.mean(np.asarray(scores) >= 0.9)), 4)
    info["center_error_m"] = quantiles(center_errors)
    info["center_error_le_0.25_rate"] = round(
        float(np.mean(np.asarray(center_errors) <= 0.25)), 4)
    info["center_error_le_0.5_rate"] = round(
        float(np.mean(np.asarray(center_errors) <= 0.5)), 4)
    info["center_error_le_1.0_rate"] = round(
        float(np.mean(np.asarray(center_errors) <= 1.0)), 4)
    if ious:
        info["iou"] = quantiles(ious)
        info["iou_ge_0.25_rate"] = round(float(np.mean(np.asarray(ious) >= 0.25)), 4)
        info["iou_ge_0.5_rate"] = round(float(np.mean(np.asarray(ious) >= 0.5)), 4)
    diffs = info["join"].pop("center_diff_scene_vs_npz_m")
    info["join"]["center_diff_m"] = quantiles(diffs) if diffs else None
    info["token_dim_values"] = dict(info["token_dim_values"])
    info["rows_per_scene"] = {str(k): v for k, v in info["rows_per_scene"].items()}
    report[platform] = info
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens-root", default="/root/autodl-tmp/3eed_data")
    parser.add_argument("--scenes", default="/root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl")
    parser.add_argument("--platforms", nargs="+", default=["waymo", "drone", "quad"])
    parser.add_argument("--out", default="token_export_check.json")
    args = parser.parse_args()

    scenes_index = defaultdict(list)
    with open(args.scenes, encoding="utf-8") as handle:
        for line in handle:
            scene = json.loads(line)
            for obj in scene["objects"]:
                if obj["role"] != "referred":
                    continue
                for expression in obj["referring_expressions"]:
                    scenes_index[scene["metadata_path"].rsplit("/meta_info.json", 1)[0]].append(
                        {"caption": expression["text"], "center": obj["gt_box"]["center"],
                         "object_id": obj["object_id"]})
    print("scene index frames:", len(scenes_index), flush=True)

    iou = iou_fn()
    report = {}
    for platform in args.platforms:
        print("auditing", platform, flush=True)
        info = audit_platform(platform, args.tokens_root, iou, scenes_index, report)
        print("  rows {} | dims {} | nan {} inf {} | scenes {} (>=2 rows: {})".format(
            info["rows"], info["token_dim_values"], info["tokens_with_nan"],
            info["tokens_with_inf"], info["scenes"], info["scenes_with_two_or_more_rows"]))
        print("  score {} | center_err {} | L2 {}".format(
            info["match_score"], info["center_error_m"], info["l2_norm"]))
        print("  join {}".format(info["join"]))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
