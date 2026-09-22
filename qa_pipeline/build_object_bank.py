"""Build the object bank: one record per exported grounding token row.

Joins the token NPZ exports with the normalized scene JSONL, using the key
verified to be exact (center difference < 0.0001 m): the NPZ `meta_path`
without the "data/3eed/" prefix picks the frame, and the cleaned utterance
matches the scene object's caption (captions are unique within a frame).

The 288-D token itself is not inlined in the JSON. Tokens are stacked per
platform into `tokens/{platform}_tokens.npz` in bank order, and each record
points at it with `token_path` + `token_index`.

    python build_object_bank.py \
        --tokens-root /root/autodl-tmp/3eed_data \
        --scenes /root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl \
        --out-dir /root/autodl-tmp/3eed_data/qa_v1
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "/root/3eedqa/3EED")

NOT_MENTIONED = " . not mentioned"


def normalize_caption(text):
    """Match the dataset: commas become separate tokens, whitespace collapsed."""
    return " ".join(str(text).replace(",", " ,").split())


def clean_utterance(text):
    """Turn the NPZ utterance (dataset format + trailing ' . not mentioned')
    back into readable caption text."""
    text = str(text)
    if text.endswith(NOT_MENTIONED):
        text = text[: -len(NOT_MENTIONED)]
    return " ".join(text.replace(" ,", ",").split())


def scene_key_from_meta(meta_path):
    parts = str(meta_path).split("/")
    if parts[:2] == ["data", "3eed"]:
        parts = parts[2:]
    return "/".join(parts[:-1])


def load_scene_index(path):
    """scene key -> {normalized caption -> scene object}"""
    index = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            scene = json.loads(line)
            key = scene["metadata_path"].rsplit("/meta_info.json", 1)[0]
            objects = {}
            for obj in scene["objects"]:
                if obj["role"] != "referred":
                    continue
                for expression in obj["referring_expressions"]:
                    objects[normalize_caption(expression["text"])] = obj
            index[key] = scene | {"_objects_by_caption": objects}
    return index


def load_iou_fn(repo):
    try:
        if repo:
            sys.path.insert(0, repo)
        from utils.eval_det import iou3d_rotated_vs_aligned
        return iou3d_rotated_vs_aligned
    except Exception as error:
        print("warning: IoU function unavailable:", error)
        return None


def build_platform(platform, args, scenes, iou, stats):
    pattern = os.path.join(args.tokens_root, "eval_{}_tokens".format(platform),
                           "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))
    if args.limit_files:
        files = files[: args.limit_files]
    records = []
    tokens = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        for row in range(len(data["query_index"])):
            raw_utterance = str(data["utterance"][row])
            utterance = clean_utterance(raw_utterance)
            key = scene_key_from_meta(data["meta_path"][row])
            scene = scenes.get(key)
            if scene is None:
                stats["join"]["no_scene"] += 1
                continue
            obj = scene["_objects_by_caption"].get(normalize_caption(utterance))
            if obj is None:
                stats["join"]["no_caption"] += 1
                continue
            token = data["object_token"][row]
            if token.shape != (288,):
                raise ValueError("unexpected token shape {} in {}".format(
                    token.shape, path))
            gt_box = data["gt_box"][row][:7].astype(np.float64)
            scene_center = np.asarray(obj["gt_box"]["center"], dtype=np.float64)
            stats["join"]["center_diff_max"] = max(
                stats["join"]["center_diff_max"],
                float(np.linalg.norm(gt_box[:3] - scene_center)))
            pred_center = data["pred_center"][row].astype(np.float64)
            center_error = float(np.linalg.norm(pred_center - gt_box[:3]))
            record = {
                "platform": platform,
                "scene_id": scene["scene_id"],
                "frame_id": scene["scene_id"].split("/")[-1],
                "split": scene["split"],
                "object_id": obj["object_id"],
                "category": obj["category"],
                "utterance": utterance,
                "utterance_raw": raw_utterance,
                "gt_box": [round(float(v), 5) for v in gt_box],
                "pred_center": [round(float(v), 5) for v in data["pred_center"][row]],
                "pred_size": [round(float(v), 5) for v in data["pred_size"][row]],
                "query_index": int(data["query_index"][row]),
                "match_score": round(float(data["match_score"][row]), 5),
                "center_error_m": round(center_error, 5),
                "iou": None,
                "token_dim": int(token.shape[0]),
                "token_path": "tokens/{}_tokens.npz".format(platform),
                "token_index": len(records),
                "source_npz": os.path.relpath(path, args.tokens_root).replace("\\", "/"),
                "metadata_path": scene["metadata_path"],
            }
            if iou is not None:
                gt_box_7 = (list(obj["gt_box"]["center"]) + list(obj["gt_box"]["size"])
                            + [obj["gt_box"]["yaw_rad"]])
                gt = torch.tensor([gt_box_7], dtype=torch.float32)
                pred = torch.tensor(
                    [list(data["pred_center"][row]) + list(data["pred_size"][row])],
                    dtype=torch.float32)
                record["iou"] = round(float(iou(gt, pred)[0][0, 0]), 5)
            records.append(record)
            tokens.append(token)
    return records, tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens-root", default="/root/autodl-tmp/3eed_data")
    parser.add_argument("--scenes", default="/root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl")
    parser.add_argument("--platforms", nargs="+", default=["waymo", "drone", "quad"])
    parser.add_argument("--out-dir", default="/root/autodl-tmp/3eed_data/qa_v1")
    parser.add_argument("--repo", default="/root/3eedqa/3EED")
    parser.add_argument("--limit-files", type=int, default=0,
                        help="use only the first N token files per platform (smoke test)")
    parser.add_argument("--no-iou", action="store_true")
    args = parser.parse_args()

    scenes = load_scene_index(args.scenes)
    print("scene index frames:", len(scenes), flush=True)
    iou = None if args.no_iou else load_iou_fn(args.repo)

    os.makedirs(os.path.join(args.out_dir, "tokens"), exist_ok=True)
    stats = {"scenes": len(scenes), "join": {"no_scene": 0, "no_caption": 0,
                                             "center_diff_max": 0.0}, "platforms": {}}
    all_records = []
    with open(os.path.join(args.out_dir, "object_bank.jsonl"), "w",
              encoding="utf-8") as handle:
        for platform in args.platforms:
            info = {"records": 0, "with_iou": 0}
            records, tokens = build_platform(platform, args, scenes, iou, stats)
            info["records"] = len(records)
            info["with_iou"] = sum(1 for r in records if r["iou"] is not None)
            if records:
                array = np.stack(tokens).astype(np.float32)
                if np.isnan(array).any() or np.isinf(array).any():
                    raise ValueError("NaN/Inf in tokens for " + platform)
                token_file = os.path.join(args.out_dir, "tokens",
                                          "{}_tokens.npz".format(platform))
                np.savez_compressed(token_file, object_token=array,
                                    object_id=np.asarray([r["object_id"] for r in records]))
                info["token_shape"] = list(array.shape)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.extend(records)
            stats["platforms"][platform] = info
            print("{}: {} records, token array {}".format(
                platform, info["records"], info.get("token_shape")), flush=True)

    stats["total_records"] = len(all_records)
    stats["iou_note"] = "iou is 3D rotated-vs-aligned IoU, same metric as the official evaluator"
    with open(os.path.join(args.out_dir, "object_bank_stats.json"), "w",
              encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print("join: no_scene {} | no_caption {} | center_diff_max {:.6f} m".format(
        stats["join"]["no_scene"], stats["join"]["no_caption"],
        stats["join"]["center_diff_max"]))
    print("wrote", os.path.join(args.out_dir, "object_bank.jsonl"),
          "({} records total)".format(len(all_records)))


if __name__ == "__main__":
    main()
