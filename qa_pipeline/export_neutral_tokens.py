"""Re-run 3EED on relation-neutral descriptions and export implicit tokens.

Run from the 3EED repository directory on the dataset/GPU host. The candidate
query is ranked by the first object noun found in the supplied description;
neither its GT category nor its GT box is used for query ranking. GT is used
after prediction only for annotation joining and evaluation.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neutral_query import first_object_noun, neutral_description


def normalize_caption(text):
    return " ".join(str(text).replace(",", " ,").split())


def load_scene_index(path):
    index = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            scene = json.loads(line)
            if scene["platform"] != "waymo":
                continue
            for obj in scene["objects"]:
                if obj["role"] != "referred":
                    continue
                for expr in obj["referring_expressions"]:
                    index[(scene["metadata_path"], normalize_caption(expr["text"]))] = (
                        scene, obj)
    return index


def relative_meta_path(path):
    path = str(path).replace("\\", "/")
    marker = "data/3eed/"
    if marker not in path:
        raise ValueError("metadata path lacks data/3eed/: " + path)
    return path.split(marker, 1)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/root/3eedqa/3EED")
    parser.add_argument("--data-root", default="data/3eed")
    parser.add_argument("--scenes", default="/root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/3eed_data/checkpoints/ckpt_6384.pth")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/3eed_data/qa_v2")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-per-split", type=int, default=0,
                        help="random pilot subset after text filtering; zero means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--span-policy", choices=["noun", "full"], default="noun",
                        help="Text span used to rank decoder queries; full uses the complete neutral phrase")
    parser.add_argument("--diagnose-candidates", action="store_true",
                        help="Offline GT IoU of all candidate boxes; never changes selected token")
    args = parser.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    from src.joint_det_dataset import Joint3DDataset, WAYMO_SYNONYMS, get_positive_map
    from train_dist_mod import TrainTester
    from utils.eval_det import iou3d_rotated_vs_aligned

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 3EED token export")

    terms = sorted({term for words in WAYMO_SYNONYMS.values() for term in words},
                   key=lambda value: (-len(value), value.lower()))
    scene_index = load_scene_index(args.scenes)
    print("indexed referred expressions", len(scene_index), flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TrainTester.get_model(ckpt["config"])
    model.load_state_dict({key.removeprefix("module."): value
                           for key, value in ckpt["model"].items()}, strict=True)
    model.cuda().eval()

    tokens, records = [], []
    stats = {"config": vars(args), "rejections": Counter(), "splits": {},
             "selector": ("full_neutral_phrase_no_gt_for_ranking" if args.span_policy == "full"
                          else "first_object_noun_in_neutral_query_no_gt_for_ranking")}

    for split_no, split in enumerate(args.splits):
        dataset = Joint3DDataset(
            dataset_dict={"waymo": 1}, test_dataset={"waymo": 1}, split=split,
            data_path=args.data_root, split_dir=os.path.join(args.data_root, "splits"),
            use_color=True, detect_intermediate=True, debug=False)
        dataset.augment = False  # no random training geometry transform in the cache
        original_loaded = len(dataset.annos)
        eligible = []
        split_rejections = Counter()
        for anno in dataset.annos:
            source_caption = anno["utterance"]
            phrase, reason = neutral_description(source_caption)
            if phrase is None:
                split_rejections[reason] += 1
                continue
            formatted = " " + normalize_caption(phrase) + " "
            noun = first_object_noun(formatted, terms)
            if noun is None:
                split_rejections["no_noun"] += 1
                continue
            start, end, noun_text = noun
            category_terms = {term.lower() for term in
                              WAYMO_SYNONYMS.get(anno["target"], [anno["target"]])}
            if noun_text.lower() not in category_terms:
                split_rejections["noun_not_target_category"] += 1
                continue
            original_key = (relative_meta_path(anno["meta_path"]),
                            normalize_caption(source_caption))
            joined = scene_index.get(original_key)
            if joined is None:
                split_rejections["no_scene_join"] += 1
                continue
            scene, obj = joined
            if scene["split"] != split:
                raise ValueError("scene split mismatch for " + scene["scene_id"])
            tokenized = dataset.tokenizer.batch_encode_plus(
                [formatted], padding="longest", return_tensors="pt")
            selected_span = (1, len(formatted) - 1) if args.span_policy == "full" else (start, end)
            positive_map = get_positive_map(tokenized, [selected_span])
            if not positive_map.any():
                split_rejections["empty_positive_map"] += 1
                continue
            updated = dict(anno)
            updated["utterance"] = phrase
            updated["pred_pos_map"] = positive_map
            eligible.append((updated, scene, obj, source_caption, noun_text))

        eligible_before_limit = len(eligible)
        if args.limit_per_split and len(eligible) > args.limit_per_split:
            chosen = sorted(random.Random(args.seed + split_no).sample(
                range(len(eligible)), args.limit_per_split))
            eligible = [eligible[i] for i in chosen]
        dataset.annos = [item[0] for item in eligible]
        stats["rejections"].update(split_rejections)
        stats["splits"][split] = {"loaded": original_loaded,
                                  "eligible_before_limit": eligible_before_limit,
                                  "exported": 0,
                                  "rejections": dict(split_rejections)}
        print(split, "export candidates", len(dataset),
              "rejected", dict(split_rejections), flush=True)
        if not eligible:
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        for batch_idx, batch in enumerate(loader):
            base = batch_idx * args.batch_size
            end = min(base + args.batch_size, len(eligible))
            info = eligible[base:end]
            with torch.no_grad():
                output = model({"point_clouds": batch["point_clouds"].float().cuda(),
                                "text": batch["utterances"]},
                               return_query_features=True)
                span = (batch["positive_map"][:, 0, :output["proj_tokens"].shape[1]]
                        > 0).cuda().to(output["proj_tokens"].dtype)
                similarity = torch.matmul(output["last_proj_queries"],
                                          output["proj_tokens"].transpose(-1, -2))
                scores = ((similarity / 0.07).softmax(-1) * span[:, None, :]).sum(-1)
                indices = scores.argmax(-1)
                row = torch.arange(len(indices), device="cuda")
                feature = output["last_query_features"][row, indices].cpu().numpy()
                pred_center = output["last_center"][row, indices].cpu().numpy()
                pred_size = output["last_pred_size"][row, indices].cpu().numpy()
                all_pred_boxes = None
                if args.diagnose_candidates:
                    all_pred_boxes = torch.cat(
                        [output["last_center"], output["last_pred_size"]], dim=-1
                    ).cpu()
                match_scores = scores[row, indices].cpu().numpy()
                query_indices = indices.cpu().numpy()

            for local, (_, scene, obj, source_caption, noun_text) in enumerate(info):
                gt = (list(obj["gt_box"]["center"]) + list(obj["gt_box"]["size"])
                      + [obj["gt_box"]["yaw_rad"]])
                pred = list(pred_center[local]) + list(pred_size[local])
                iou = float(iou3d_rotated_vs_aligned(
                    torch.tensor([gt], dtype=torch.float32),
                    torch.tensor([pred], dtype=torch.float32))[0][0, 0])
                token_index = len(tokens)
                tokens.append(feature[local].astype(np.float32))
                records.append({
                    "platform": "waymo", "split": split,
                    "sequence": scene["sequence"], "scene_id": scene["scene_id"],
                    "object_id": obj["object_id"], "category": obj["category"],
                    "neutral_query": dataset.annos[base + local]["utterance"],
                    "matched_noun": noun_text, "source_caption": source_caption,
                    "gt_box": [round(float(x), 5) for x in gt],
                    "pred_center": [round(float(x), 5) for x in pred_center[local]],
                    "pred_size": [round(float(x), 5) for x in pred_size[local]],
                    "iou": round(iou, 5),
                    "query_index": int(query_indices[local]),
                    "match_score": round(float(match_scores[local]), 6),
                    "token_path": "tokens/waymo_tokens.npz",
                    "token_index": token_index, "token_dim": 288,
                    "selection_policy": stats["selector"],
                    "label_source": "3eed_gt_box",
                })
                if args.diagnose_candidates:
                    candidate_iou, _ = iou3d_rotated_vs_aligned(
                        torch.tensor([gt], dtype=torch.float32),
                        all_pred_boxes[local].float())
                    values = candidate_iou[0].numpy()
                    ranking = np.argsort(-scores[local].detach().cpu().numpy())
                    records[-1]["oracle_best_iou"] = round(float(values.max()), 5)
                    records[-1]["oracle_best_query_index"] = int(values.argmax())
                    records[-1]["top5_best_iou"] = round(float(values[ranking[:5]].max()), 5)
                    records[-1]["top10_best_iou"] = round(float(values[ranking[:10]].max()), 5)
                    records[-1]["oracle_query_rank"] = int(np.flatnonzero(
                        ranking == values.argmax())[0]) + 1
            stats["splits"][split]["exported"] = sum(r["split"] == split for r in records)
            if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(loader):
                print(split, "batch", batch_idx + 1, "/", len(loader),
                      "exported", stats["splits"][split]["exported"], flush=True)

    out = Path(args.out_dir)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    if not tokens:
        raise RuntimeError("no neutral tokens exported")
    stacked = np.stack(tokens).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise ValueError("NaN/Inf in exported token features")
    np.savez_compressed(out / "tokens/waymo_tokens.npz", object_token=stacked,
                        object_id=np.asarray([r["object_id"] for r in records]),
                        selection_policy=stats["selector"])
    with (out / "neutral_bank.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for split in args.splits:
        values = [r["iou"] for r in records if r["split"] == split]
        if values:
            stats["splits"][split].update({
                "iou_mean": round(float(np.mean(values)), 4),
                "acc_025": round(float(np.mean(np.asarray(values) >= 0.25)), 4),
                "acc_05": round(float(np.mean(np.asarray(values) >= 0.5)), 4),
            })
            if args.diagnose_candidates:
                oracle = [r["oracle_best_iou"] for r in records if r["split"] == split]
                stats["splits"][split].update({
                    "oracle_iou_mean": round(float(np.mean(oracle)), 4),
                    "oracle_acc_025": round(float(np.mean(np.asarray(oracle) >= 0.25)), 4),
                    "oracle_acc_05": round(float(np.mean(np.asarray(oracle) >= 0.5)), 4),
                })
                for top in (5, 10):
                    top_values = [r[f"top{top}_best_iou"] for r in records
                                  if r["split"] == split]
                    stats["splits"][split][f"top{top}_acc_025"] = round(
                        float(np.mean(np.asarray(top_values) >= 0.25)), 4)
    stats["total_exported"] = len(records)
    stats["token_shape"] = list(stacked.shape)
    stats["rejections"] = dict(stats["rejections"])
    (out / "neutral_export_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "splits": stats["splits"],
                      "token_shape": stats["token_shape"]},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
