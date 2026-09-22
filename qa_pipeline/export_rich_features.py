"""Export rich multi-layer 3EED query features for implicit-token QA screening.

For every (scene, neutral target description) pair already present in the
implicit token bank this exports, in one frozen-Grounding pass:

- unified candidate query IDs: decoder slot index 0..255 per scene. Every
  decoder layer refines the same slot tensor in place, so slot j is the same
  candidate across all layers (verified by construction; the export also
  asserts the final layer matches the previously exported last_query_features);
- text-ranking top-5 per decoder layer computed only from the neutral
  description and model outputs (contrastive query/text similarity). No GT
  category, GT text span, or GT box is read for ranking;
- 288-D query features for the top-5 candidates of every decoder layer;
- point-cloud scene features: mean/max pooled pre-fusion (fp2) and
  post-fusion (seed) point features;
- offline GT diagnostics computed only after prediction: per-layer IoU of all
  256 candidate boxes against the referred GT box, plus per-layer score
  vectors for all 256 candidates under both ranking policies, so candidate
  recall for any top-k and any policy can be recomputed offline without
  re-running the model.

Ranking policies exported per layer (all 256 scores each):
- "full": softmax over every non-special, non-padding description token;
- "noun": the existing first-object-noun span policy (public lexicon only),
  kept for continuity with the qa_v2 token bank.

Output layout (out-dir):
  rank00/batch%06d.npz   arrays, see export_stats.json for shapes
  records.jsonl          one row per sample, joins back to neutral_bank.jsonl
  export_stats.json      roster checks, dims, per-layer accuracy/recall stats
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

NUM_LAYERS = 6
NUM_CANDIDATES = 256
FEATURE_VERSION = "rich_v1_ckpt6384"


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


def load_bank(path):
    rows, by_object = [], {}
    with open(path, encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            row["__index"] = index
            rows.append(row)
            by_object.setdefault(row["object_id"], []).append(row)
    return rows, by_object


def full_token_mask(text_attention_mask):
    """Mask every real token except the two special tokens (RoBERTa CLS/SEP)."""
    valid = ~text_attention_mask.bool()  # True = real token
    mask = valid.clone()
    if mask.shape[1] == 0:
        return mask
    mask[:, 0] = False
    lengths = valid.sum(-1)
    rows = torch.arange(mask.shape[0], device=mask.device)
    mask[rows, (lengths - 1).clamp(min=0)] = False
    return mask


def span_scores(proj_queries, proj_tokens, mask, temperature=0.07):
    similarity = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
    probs = (similarity / temperature).softmax(-1)
    return (probs * mask[:, None, :].to(probs.dtype)).sum(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/root/3eedqa/3EED")
    parser.add_argument("--data-root", default="data/3eed")
    parser.add_argument("--scenes", default="/root/autodl-tmp/3eed_data/scenes/3eed_scenes.jsonl")
    parser.add_argument("--bank", default="/root/autodl-tmp/3eed_data/qa_v2/neutral_bank.jsonl")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/3eed_data/checkpoints/ckpt_6384.pth")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/3eed_data/qa_features_v1")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="first N eligible samples per split for smoke checks; zero means all")
    parser.add_argument("--seed", type=int, default=42)
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
        raise RuntimeError("CUDA is required for rich feature export")

    terms = sorted({term for words in WAYMO_SYNONYMS.values() for term in words},
                   key=lambda value: (-len(value), value.lower()))
    scene_index = load_scene_index(args.scenes)
    bank_rows, bank_by_object = load_bank(args.bank)
    print("scenes indexed", len(scene_index), "| bank rows", len(bank_rows), flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TrainTester.get_model(ckpt["config"])
    model.load_state_dict({key.removeprefix("module."): value
                           for key, value in ckpt["model"].items()}, strict=True)
    model.cuda().eval()
    print("model loaded; decoder layers", model.num_decoder_layers,
          "| queries", model.num_queries, flush=True)

    out = Path(args.out_dir)
    (out / "rank00").mkdir(parents=True, exist_ok=True)
    record_stream = (out / "records.jsonl").open("w", encoding="utf-8")
    stats = {"version": FEATURE_VERSION, "config": vars(args), "splits": {},
             "ranking": {
                 "full": "softmax over all non-special non-padding neutral-description tokens",
                 "noun": "first object noun of the neutral description (public lexicon)",
                 "no_gt_statement": "ranking uses only the neutral description and model "
                                    "outputs; GT boxes are used after prediction for IoU "
                                    "diagnostics only",
             }}
    total_records = 0

    for split_no, split in enumerate(args.splits):
        dataset = Joint3DDataset(
            dataset_dict={"waymo": 1}, test_dataset={"waymo": 1}, split=split,
            data_path=args.data_root, split_dir=os.path.join(args.data_root, "splits"),
            use_color=True, detect_intermediate=True, debug=False)
        dataset.augment = False
        original_loaded = len(dataset.annos)
        eligible = []
        rejections = Counter()
        for anno in dataset.annos:
            source_caption = anno["utterance"]
            phrase, reason = neutral_description(source_caption)
            if phrase is None:
                rejections[reason] += 1
                continue
            formatted = " " + normalize_caption(phrase) + " "
            noun = first_object_noun(formatted, terms)
            if noun is None:
                rejections["no_noun"] += 1
                continue
            start, end, noun_text = noun
            category_terms = {term.lower() for term in
                              WAYMO_SYNONYMS.get(anno["target"], [anno["target"]])}
            if noun_text.lower() not in category_terms:
                rejections["noun_not_target_category"] += 1
                continue
            original_key = (relative_meta_path(anno["meta_path"]),
                            normalize_caption(source_caption))
            joined = scene_index.get(original_key)
            if joined is None:
                rejections["no_scene_join"] += 1
                continue
            scene, obj = joined
            if scene["split"] != split:
                raise ValueError("scene split mismatch for " + scene["scene_id"])
            tokenized = dataset.tokenizer.batch_encode_plus(
                [formatted], padding="longest", return_tensors="pt")
            positive_map = get_positive_map(tokenized, [(start, end)])
            if not positive_map.any():
                rejections["empty_positive_map"] += 1
                continue
            updated = dict(anno)
            updated["utterance"] = phrase
            updated["pred_pos_map"] = positive_map
            eligible.append((updated, scene, obj, source_caption, noun_text))

        if args.limit and len(eligible) > args.limit:
            eligible = eligible[:args.limit]
        eligible_ids = [item[2]["object_id"] for item in eligible]
        bank_ids = sorted(row["object_id"] for row in bank_rows if row["split"] == split)
        if args.limit:
            missing = [oid for oid in eligible_ids if oid not in bank_by_object]
            if missing:
                raise ValueError(f"smoke subset has object ids outside bank: {missing[:3]}")
            roster_check = "subset_of_bank"
        else:
            if sorted(eligible_ids) != bank_ids:
                only_bank = sorted(set(bank_ids) - set(eligible_ids))
                only_eligible = sorted(set(eligible_ids) - set(bank_ids))
                raise ValueError(
                    f"roster mismatch for {split}: bank-only {len(only_bank)} "
                    f"eligible-only {len(only_eligible)}; first bank-only {only_bank[:3]}")
            roster_check = "identical_to_bank"

        dataset.annos = [item[0] for item in eligible]
        stats["splits"][split] = {"loaded": original_loaded, "eligible": len(eligible),
                                  "roster_check": roster_check,
                                  "rejections": dict(rejections)}
        print(f"[{split}] eligible {len(eligible)} roster {roster_check} "
              f"rejected {dict(rejections)}", flush=True)
        if not eligible:
            continue

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        per_layer_noun_top1_iou = [[] for _ in range(NUM_LAYERS)]
        per_layer_full_top1_iou = [[] for _ in range(NUM_LAYERS)]
        per_layer_top5_best = [[] for _ in range(NUM_LAYERS)]
        per_layer_oracle = [[] for _ in range(NUM_LAYERS)]
        noun_top1_iou = []
        noun_top1_match = 0
        noun_top1_total = 0
        # element layout for one sample row inside a batch npz is described by
        # export_stats["array_shapes"] written at the end.
        for batch_idx, batch in enumerate(loader):
            base = batch_idx * args.batch_size
            end = min(base + args.batch_size, len(eligible))
            info = eligible[base:end]
            batch_n = end - base
            with torch.no_grad():
                output = model({"point_clouds": batch["point_clouds"].float().cuda(),
                                "text": batch["utterances"]},
                               return_query_features=True)
                proj_tokens = output["proj_tokens"]
                noun_mask = (batch["positive_map"][:, 0, :proj_tokens.shape[1]]
                             > 0).cuda().to(proj_tokens.dtype)
                assert torch.all(noun_mask.sum(-1) > 0), "empty noun span in batch"
                full_mask = full_token_mask(output["text_attention_mask"])

                assert output["last_query_features"].shape[1] == NUM_CANDIDATES
                assert output["last_query_features"].shape[-1] == 288
                fp2 = output["fp2_features"]  # (B, 288, N) pre-fusion point features
                seed = output["seed_features"]  # (B, 288, N) post-fusion point features
                assert fp2.shape[1] == 288 and seed.shape[1] == 288

                layer_features, layer_top5_idx, layer_top5_score = [], [], []
                layer_scores_full, layer_scores_noun = [], []
                layer_iou, layer_top5_center, layer_top5_size = [], [], []
                for i in range(NUM_LAYERS):
                    prefix = "last_" if i == model.num_decoder_layers - 1 else f"{i}head_"
                    features = output[f"decoder_{i}_query_features"]  # (B, 256, 288)
                    assert features.shape == (batch_n, NUM_CANDIDATES, 288)
                    if i == model.num_decoder_layers - 1:
                        assert torch.equal(features, output["last_query_features"]), \
                            "final layer features must match last_query_features"
                    proj_q = output[f"{prefix}proj_queries"]  # (B, 256, 64)
                    centers = output[f"{prefix}center"]      # (B, 256, 3)
                    sizes = output[f"{prefix}pred_size"]     # (B, 256, 3)
                    assert proj_q.shape[:2] == (batch_n, NUM_CANDIDATES)
                    assert centers.shape == sizes.shape == (batch_n, NUM_CANDIDATES, 3)

                    score_full = span_scores(proj_q, proj_tokens, full_mask)
                    score_noun = span_scores(proj_q, proj_tokens, noun_mask)
                    top_scores, top_idx = torch.topk(score_full, 5, dim=-1)

                    pred_boxes = torch.cat([centers, sizes], dim=-1)  # (B, 256, 6)
                    ious = torch.zeros(batch_n, NUM_CANDIDATES, device="cuda")
                    for b in range(batch_n):
                        gt = info[b][2]["gt_box"]
                        gt_box = torch.tensor(
                            [list(gt["center"]) + list(gt["size"]) + [gt["yaw_rad"]]],
                            dtype=torch.float32, device="cuda")
                        iou_matrix, _ = iou3d_rotated_vs_aligned(gt_box, pred_boxes[b])
                        ious[b] = iou_matrix[0]

                    layer_features.append(features)
                    layer_top5_idx.append(top_idx)
                    layer_top5_score.append(top_scores)
                    layer_scores_full.append(score_full)
                    layer_scores_noun.append(score_noun)
                    layer_iou.append(ious)
                    gather = top_idx[..., None].expand(-1, -1, 3)
                    layer_top5_center.append(torch.gather(centers, 1, gather))
                    layer_top5_size.append(torch.gather(sizes, 1, gather))

                layer_features = torch.stack(layer_features, 1)          # (B, L, 256, 288)
                layer_top5_idx = torch.stack(layer_top5_idx, 1)          # (B, L, 5)
                layer_top5_score = torch.stack(layer_top5_score, 1)      # (B, L, 5)
                layer_scores_full = torch.stack(layer_scores_full, 1)    # (B, L, 256)
                layer_scores_noun = torch.stack(layer_scores_noun, 1)    # (B, L, 256)
                layer_iou = torch.stack(layer_iou, 1)                    # (B, L, 256)
                layer_top5_center = torch.stack(layer_top5_center, 1)    # (B, L, 5, 3)
                layer_top5_size = torch.stack(layer_top5_size, 1)        # (B, L, 5, 3)
                top5_features = torch.gather(
                    layer_features, 3, layer_top5_idx[..., None].expand(-1, -1, -1, 288))

                noun_idx_last = layer_scores_noun[:, -1].argmax(-1)      # (B,)
                noun_score_last = layer_scores_noun[:, -1].gather(
                    1, noun_idx_last[:, None]).squeeze(1)

                arrays = {
                    "candidate_ids": np.arange(NUM_CANDIDATES, dtype=np.int32),
                    "layer_features_top5": top5_features.cpu().numpy().astype(np.float32),
                    "layer_top5_idx": layer_top5_idx.cpu().numpy().astype(np.int32),
                    "layer_top5_score_full": layer_top5_score.cpu().numpy().astype(np.float32),
                    "layer_scores_full": layer_scores_full.cpu().numpy().astype(np.float32),
                    "layer_scores_noun": layer_scores_noun.cpu().numpy().astype(np.float32),
                    "layer_iou": layer_iou.cpu().numpy().astype(np.float32),
                    "layer_top5_center": layer_top5_center.cpu().numpy().astype(np.float32),
                    "layer_top5_size": layer_top5_size.cpu().numpy().astype(np.float32),
                    "noun_top1_idx_last": noun_idx_last.cpu().numpy().astype(np.int32),
                    "noun_top1_score_last": noun_score_last.cpu().numpy().astype(np.float32),
                    "scene_fp2_mean": fp2.mean(-1).cpu().numpy().astype(np.float32),
                    "scene_fp2_max": fp2.max(-1).values.cpu().numpy().astype(np.float32),
                    "scene_seed_mean": seed.mean(-1).cpu().numpy().astype(np.float32),
                    "scene_seed_max": seed.max(-1).values.cpu().numpy().astype(np.float32),
                }
                for name, arr in arrays.items():
                    if not np.isfinite(arr).all():
                        raise ValueError(f"NaN/Inf in {name} at batch {batch_idx}")

            rel_file = f"rank00/{split}_batch{batch_idx:06d}.npz"
            np.savez_compressed(out / rel_file, **arrays)
            with np.load(out / rel_file) as check:
                if check["layer_features_top5"].shape != (batch_n, NUM_LAYERS, 5, 288):
                    raise ValueError("reloaded npz shape mismatch: " + rel_file)
                if int(check["noun_top1_idx_last"][0]) != int(noun_idx_last[0]):
                    raise ValueError("reloaded npz content mismatch: " + rel_file)

            for local, (_, scene, obj, source_caption, noun_text) in enumerate(info):
                bank_row = bank_by_object[obj["object_id"]][0]
                gt = (list(obj["gt_box"]["center"]) + list(obj["gt_box"]["size"])
                      + [obj["gt_box"]["yaw_rad"]])
                for i in range(NUM_LAYERS):
                    values = layer_iou[local, i].cpu().numpy()
                    per_layer_oracle[i].append(float(values.max()))
                    per_layer_noun_top1_iou[i].append(
                        float(values[layer_scores_noun[local, i].argmax()]))
                    per_layer_full_top1_iou[i].append(
                        float(values[layer_scores_full[local, i].argmax()]))
                    top5_idx = layer_top5_idx[local, i].cpu().numpy()
                    per_layer_top5_best[i].append(float(values[top5_idx].max()))
                noun_idx = int(noun_idx_last[local])
                noun_iou = float(layer_iou[local, -1, noun_idx])
                noun_top1_iou.append(noun_iou)
                noun_top1_total += 1
                noun_top1_match += int(noun_idx == int(bank_row["query_index"]))
                record = {
                    "platform": "waymo", "split": split, "sequence": scene["sequence"],
                    "scene_id": scene["scene_id"], "object_id": obj["object_id"],
                    "category": obj["category"], "neutral_query": bank_row["neutral_query"],
                    "matched_noun": noun_text, "source_caption": source_caption,
                    "gt_box": [round(float(x), 5) for x in gt],
                    "feature_version": FEATURE_VERSION,
                    "feature_file": rel_file, "feature_row": local,
                    "num_layers": NUM_LAYERS, "num_candidates": NUM_CANDIDATES,
                    "bank_index": int(bank_row["__index"]),
                    "bank_query_index": int(bank_row["query_index"]),
                    "noun_top1_idx_last": noun_idx,
                    "noun_top1_iou_last": round(noun_iou, 5),
                    "selection_policy": "text_ranking_no_gt; see export_stats.ranking",
                    "label_source": "3eed_gt_box",
                }
                record_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_records += batch_n
            if (batch_idx + 1) % 50 == 0 or end == len(eligible):
                print(f"[{split}] batch {batch_idx + 1}/{len(loader)} rows {end}", flush=True)

        split_stats = {"exported": len(eligible),
                       "noun_policy_query_index_match_vs_bank": noun_top1_match,
                       "noun_top1_candidates": noun_top1_total}
        if noun_top1_iou:
            values = np.asarray(noun_top1_iou)
            split_stats["noun_policy_last_layer"] = {
                "acc_025": round(float((values >= 0.25).mean()), 4),
                "acc_05": round(float((values >= 0.50).mean()), 4),
                "iou_mean": round(float(values.mean()), 4),
            }
        for i in range(NUM_LAYERS):
            layer_stats = {
                "full_top1_acc_025": round(float((np.asarray(per_layer_full_top1_iou[i]) >= 0.25).mean()), 4),
                "noun_top1_acc_025": round(float((np.asarray(per_layer_noun_top1_iou[i]) >= 0.25).mean()), 4),
                "top5_recall_acc_025": round(float((np.asarray(per_layer_top5_best[i]) >= 0.25).mean()), 4),
                "oracle_acc_025": round(float((np.asarray(per_layer_oracle[i]) >= 0.25).mean()), 4),
                "iou_mean_top1_full": round(float(np.asarray(per_layer_full_top1_iou[i]).mean()), 4),
                "iou_mean_top5_best": round(float(np.asarray(per_layer_top5_best[i]).mean()), 4),
            }
            split_stats[f"layer_{i}"] = layer_stats
        stats["splits"][split].update(split_stats)
        print(f"[{split}] " + json.dumps(split_stats, ensure_ascii=False), flush=True)

    record_stream.close()
    stats["total_exported"] = total_records
    for split in args.splits:
        entry = stats["splits"].get(split)
        if not entry:
            continue
        expected = -(-entry["eligible"] // args.batch_size)
        found = len(list((out / "rank00").glob(f"{split}_batch*.npz")))
        if found != expected:
            raise ValueError(f"{split}: expected {expected} npz shards, found {found}")
        entry["shards"] = found
    stats["array_shapes"] = {
        "layer_features_top5": ["B", NUM_LAYERS, 5, 288],
        "layer_top5_idx": ["B", NUM_LAYERS, 5],
        "layer_top5_score_full": ["B", NUM_LAYERS, 5],
        "layer_scores_full": ["B", NUM_LAYERS, NUM_CANDIDATES],
        "layer_scores_noun": ["B", NUM_LAYERS, NUM_CANDIDATES],
        "layer_iou": ["B", NUM_LAYERS, NUM_CANDIDATES],
        "layer_top5_center": ["B", NUM_LAYERS, 5, 3],
        "layer_top5_size": ["B", NUM_LAYERS, 5, 3],
        "noun_top1_idx_last": ["B"],
        "scene_fp2_mean": ["B", 288],
        "scene_seed_mean": ["B", 288],
    }
    (out / "export_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "total_exported": total_records},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
