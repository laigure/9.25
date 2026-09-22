"""Attach variable-cardinality Grounding tokens to scene reasoning QA.

GT boxes are used only for diagnostics and optional correct-Grounding subsets.
Public QA rows contain no box, center, coordinate, distance or answer key.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.eval_det import iou3d_rotated_vs_aligned


TOKEN_FIELDS = {
    "soft": ("object_token", "pred_center", "pred_size"),
    "contrastive": ("contrastive_object_token", "contrastive_pred_center",
                    "contrastive_pred_size"),
}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_targets(array, sample_idx, count):
    value = array[sample_idx]
    if count == 1 and value.ndim == 1 and value.shape[-1] in (3, 7, 288):
        # A batch containing only single-target samples is exported without a
        # target axis for backward compatibility.
        return value[None, ...]
    return value[:count]


def load_exports(root, threshold, token_variant):
    token_key, center_key, size_key = TOKEN_FIELDS[token_variant]
    groups, count_distribution = {}, Counter()
    for path in sorted(Path(root).rglob("*.npz")):
        with np.load(path) as data:
            for sample_idx, count in enumerate(data["target_count"].tolist()):
                count = int(count)
                ids = json.loads(str(data["object_ids_json"][sample_idx]))
                if count != len(ids) or count < 1:
                    raise ValueError((path, sample_idx, count, ids))
                tokens = sample_targets(data[token_key], sample_idx, count)
                centers = sample_targets(data[center_key], sample_idx, count)
                sizes = sample_targets(data[size_key], sample_idx, count)
                gt = sample_targets(data["gt_box"], sample_idx, count)
                if tokens.shape != (count, 288):
                    raise ValueError((path, sample_idx, tokens.shape, count))
                boxes = np.concatenate((centers, sizes), axis=1)
                ious = []
                for index in range(count):
                    values, _ = iou3d_rotated_vs_aligned(
                        torch.as_tensor(gt[index:index + 1]),
                        torch.as_tensor(boxes[index:index + 1]))
                    ious.append(float(values[0, 0]))
                key = frozenset(ids)
                if key in groups:
                    raise ValueError(f"duplicate scene target set: {ids}")
                groups[key] = {"ids": ids, "tokens": tokens.copy(),
                               "ious": ious,
                               "pass": all(value >= threshold for value in ious)}
                count_distribution[count] += 1
    if not groups:
        raise ValueError(f"no token exports below {root}")
    return groups, count_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--private-gt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--token-variant", choices=sorted(TOKEN_FIELDS),
                        default="contrastive")
    parser.add_argument("--eligibility", choices=("correct", "all"),
                        default="all")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--make-shuffled", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    (output / "joint" / "tokens").mkdir(parents=True, exist_ok=True)
    groups, counts = load_exports(args.exports, args.iou, args.token_variant)
    qa = read_jsonl(args.qa)
    private = {x["qa_id"]: x for x in read_jsonl(args.private_gt)}
    keep_splits = set(args.splits.split(","))

    matched = []
    used_groups = {}
    for row in qa:
        if row["qa_split"] not in keep_splits:
            continue
        key = frozenset(x["object_id"] for x in row["object_refs"])
        group = groups.get(key)
        if group is None or (args.eligibility == "correct" and not group["pass"]):
            continue
        matched.append((row, group, key))
        used_groups[key] = group

    vectors, object_ids, object_splits, object_categories = [], [], [], []
    index = {}
    category_by_id = {ref["object_id"]: ref.get("category", "unknown")
                      for row, _, _ in matched for ref in row["object_refs"]}
    split_by_key = {key: row["qa_split"] for row, _, key in matched}
    for key, group in sorted(used_groups.items(), key=lambda item: sorted(item[0])):
        for local_index, object_id in enumerate(group["ids"]):
            index[(key, object_id)] = len(vectors)
            vectors.append(group["tokens"][local_index])
            object_ids.append(object_id)
            object_splits.append(split_by_key[key])
            object_categories.append(category_by_id.get(object_id, "unknown"))

    joint_rows, private_out = [], {}
    for row, group, key in matched:
        converted = dict(row, token_path="tokens/scene_tokens.npz")
        converted["object_refs"] = [dict(
            ref, token_index=index[(key, ref["object_id"])])
            for ref in row["object_refs"]]
        joint_rows.append(converted)
        diag = dict(private[row["qa_id"]])
        id_to_iou = dict(zip(group["ids"], group["ious"]))
        diag["diagnostic_grounding_ious"] = [id_to_iou[ref["object_id"]]
                                             for ref in row["object_refs"]]
        diag["diagnostic_grounding_pass"] = bool(group["pass"])
        private_out[row["qa_id"]] = diag

    np.savez_compressed(
        output / "joint" / "tokens" / "scene_tokens.npz",
        object_token=np.asarray(vectors, dtype=np.float32),
        object_id=np.asarray(object_ids, dtype=str),
        split=np.asarray(object_splits, dtype=str),
        category=np.asarray(object_categories, dtype=str),
    )
    write_jsonl(output / "joint" / "qa.jsonl", joint_rows)
    write_jsonl(output / "joint" / "private_gt.jsonl",
                [private_out[row["qa_id"]] for row in joint_rows])

    shuffled_rows = []
    if args.make_shuffled:
        pools = defaultdict(list)
        for i, (split, category) in enumerate(zip(object_splits, object_categories)):
            pools[(split, category)].append(i)
        donor = {}
        for i, (split, category, object_id) in enumerate(zip(
                object_splits, object_categories, object_ids)):
            choices = [j for j in pools[(split, category)]
                       if object_ids[j] != object_id]
            if choices:
                donor[i] = choices[(i * 17 + 42) % len(choices)]
        shuffled_vectors = np.asarray(vectors, dtype=np.float32).copy()
        for i, j in donor.items():
            shuffled_vectors[i] = vectors[j]
        shuffled_dir = output / "shuffled" / "tokens"
        shuffled_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shuffled_dir / "scene_tokens.npz", object_token=shuffled_vectors,
            object_id=np.asarray(object_ids, dtype=str),
            split=np.asarray(object_splits, dtype=str),
            category=np.asarray(object_categories, dtype=str))
        for row in joint_rows:
            if all(ref["token_index"] in donor for ref in row["object_refs"]):
                shuffled_rows.append(dict(row, token_path="tokens/scene_tokens.npz"))
        write_jsonl(output / "shuffled" / "qa.jsonl", shuffled_rows)
        write_jsonl(output / "shuffled" / "private_gt.jsonl",
                    [private_out[row["qa_id"]] for row in shuffled_rows])

    summary = {
        "token_variant": args.token_variant, "iou": args.iou,
        "eligibility": args.eligibility,
        "grounding_scene_count": len(groups),
        "grounding_target_count_distribution": dict(sorted(counts.items())),
        "grounding_all_targets_correct": sum(x["pass"] for x in groups.values()),
        "qa_rows": len(joint_rows),
        "qa_split_counts": dict(Counter(x["qa_split"] for x in joint_rows)),
        "qa_scenario_counts": dict(Counter(x["scenario"] for x in joint_rows)),
        "token_entries": len(vectors), "shuffled_qa_rows": len(shuffled_rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
