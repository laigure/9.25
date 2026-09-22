"""Build matched QA diagnostics from correctly grounded joint token exports.

GT boxes are used only to decide diagnostic eligibility. QA inputs contain
question text and implicit 288-D features, with no box or coordinate fields.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.eval_det import iou3d_rotated_vs_aligned


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


TOKEN_FIELDS = {
    "soft": ("object_token", "pred_center", "pred_size"),
    "contrastive": ("contrastive_object_token", "contrastive_pred_center",
                    "contrastive_pred_size"),
}


def load_exports(root, threshold, token_variant):
    token_key, center_key, size_key = TOKEN_FIELDS[token_variant]
    groups = {}
    seen = Counter()
    for path in sorted(Path(root).rglob("*.npz")):
        with np.load(path) as data:
            for sample_idx, count in enumerate(data["target_count"].tolist()):
                ids = json.loads(str(data["object_ids_json"][sample_idx]))
                if count != len(ids) or count < 2:
                    raise ValueError((path, sample_idx, ids))
                boxes = np.concatenate((data[center_key][sample_idx, :count],
                                        data[size_key][sample_idx, :count]), axis=1)
                gt = data["gt_box"][sample_idx, :count]
                ious = []
                for obj_idx in range(count):
                    values, _ = iou3d_rotated_vs_aligned(
                        torch.as_tensor(gt[obj_idx:obj_idx + 1]),
                        torch.as_tensor(boxes[obj_idx:obj_idx + 1]))
                    ious.append(float(values[0, 0]))
                key = frozenset(ids)
                if key in groups:
                    raise ValueError(f"Duplicate target set: {ids}")
                groups[key] = {"ids": ids, "tokens": data[token_key][sample_idx, :count].copy(),
                               "ious": ious, "pass": all(v >= threshold for v in ious)}
                seen[len(ids)] += 1
    if not groups:
        raise ValueError(f"No exports in {root}")
    return groups, seen


def reverse_relation(key):
    parts = key.split("_")
    inverse = {"left": "right", "right": "left",
               "front": "behind", "behind": "front"}
    return "_".join(inverse[part] for part in parts)


def relation_answer(key):
    words = {"left": "to the left of", "right": "to the right of",
             "front": "in front of", "behind": "behind"}
    parts = key.split("_")
    if len(parts) == 2:
        return "Object A is {} and {} Object B.".format(words[parts[0]],
                                                         words[parts[1]])
    return "Object A is {} Object B.".format(words[parts[0]])


def swapped_pair(row, gt):
    """Return a physically equivalent A/B reversal with the correct label."""
    if len(row["object_refs"]) != 2:
        return None
    scenario = row["scenario"]
    if scenario not in ("relative_location", "which_is_left", "closer_to_ego"):
        return None
    first, second = row["object_refs"]
    refs = [dict(second, role="A"), dict(first, role="B")]
    desc_a, desc_b = refs[0]["description"], refs[1]["description"]
    if scenario == "relative_location":
        key = reverse_relation(gt["answer_key"])
        question = ("Object A: {}. Object B: {}. Where is Object A relative "
                    "to Object B?").format(desc_a, desc_b)
        answer = relation_answer(key)
    else:
        key = "B" if gt["answer_key"] == "A" else "A"
        if scenario == "which_is_left":
            question = ("Object A: {}. Object B: {}. Which object is farther "
                        "to the left, Object A or Object B?").format(desc_a, desc_b)
            answer = "Object {} is farther to the left.".format(key)
        else:
            question = ("Object A: {}. Object B: {}. Which object is closer "
                        "to the ego vehicle, Object A or Object B?").format(desc_a, desc_b)
            answer = "Object {} is closer to the ego vehicle.".format(key)
    qa_id = hashlib.sha1((row["qa_id"] + "|ab_swap_v4").encode()).hexdigest()[:16]
    swapped = dict(row, qa_id=qa_id, question=question, answer=answer,
                   object_refs=refs, augmentation="ab_swap")
    swapped_gt = dict(gt, qa_id=qa_id, answer_key=key,
                      gt_geometry=list(reversed(gt.get("gt_geometry", []))),
                      rule=str(gt.get("rule", "")) + "|ab_swap")
    if "diagnostic_grounding_ious" in gt:
        swapped_gt["diagnostic_grounding_ious"] = list(
            reversed(gt["diagnostic_grounding_ious"]))
    return swapped, swapped_gt


def add_missing_train_swaps(rows, private):
    """Complete ordered A/B pairs in train without duplicating existing reversals."""
    signatures = {(r["scenario"], tuple(ref["object_id"] for ref in r["object_refs"]))
                  for r in rows}
    result = list(rows)
    generated_private = dict(private)
    added = Counter()
    for row in rows:
        if row["qa_split"] != "train":
            continue
        pair = swapped_pair(row, private[row["qa_id"]])
        if pair is None:
            continue
        swapped, swapped_gt = pair
        signature = (swapped["scenario"],
                     tuple(ref["object_id"] for ref in swapped["object_refs"]))
        if signature in signatures:
            continue
        signatures.add(signature)
        result.append(swapped)
        generated_private[swapped["qa_id"]] = swapped_gt
        added[swapped["scenario"]] += 1
    return result, generated_private, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--private-gt", required=True)
    parser.add_argument("--neutral-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou", type=float, default=0.25)
    parser.add_argument("--token-variant", choices=sorted(TOKEN_FIELDS),
                        default="contrastive")
    parser.add_argument("--augment-train-swaps", action="store_true")
    parser.add_argument("--eligibility", choices=("correct", "all"),
                        default="correct",
                        help="correct: require every selected box to pass --iou; "
                             "all: keep every exported target pair for end-to-end evaluation")
    parser.add_argument("--keep-without-donor", action="store_true",
                        help="Do not discard rows lacking a shuffled-control donor; "
                             "use for full end-to-end evaluation")
    parser.add_argument("--splits", default="val,test",
                        help="Comma-separated QA splits to keep; add train only for "
                             "training projector/LoRA, never for scoring")
    args = parser.parse_args()
    keep_splits = tuple(args.splits.split(","))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups, counts = load_exports(args.exports, args.iou, args.token_variant)
    qa = read_jsonl(args.qa)
    private = {row["qa_id"]: row for row in read_jsonl(args.private_gt)}
    categories = {row["object_id"]: row["category"] for row in read_jsonl(args.neutral_bank)}

    # Same target set is mapped without using IoU to choose among predictions.
    eligible = []
    for row in qa:
        if len(row["object_refs"]) < 2 or row["qa_split"] not in keep_splits:
            continue
        group = groups.get(frozenset(ref["object_id"] for ref in row["object_refs"]))
        if group is not None and (args.eligibility == "all" or group["pass"]):
            eligible.append((row, group))

    # A token depends on its joint caption, so keep one bank index per
    # (target-set, role), even when the object appears in another caption.
    index = {}
    vectors, ids, splits, cats = [], [], [], []
    for row, group in eligible:
        key = frozenset(group["ids"])
        for obj_idx, oid in enumerate(group["ids"]):
            item = (key, oid)
            if item not in index:
                index[item] = len(vectors)
                vectors.append(group["tokens"][obj_idx])
                ids.append(oid)
                splits.append(row["qa_split"])
                cats.append(categories[oid])
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 288:
        raise ValueError(f"Expected [N,288] tokens, got {vectors.shape}")

    donors = {}
    pools = defaultdict(list)
    for idx, (split, category) in enumerate(zip(splits, cats)):
        pools[(split, category)].append(idx)
    for pool in pools.values():
        for idx in pool:
            other = [j for j in pool if ids[j] != ids[idx]]
            if other:
                donors[idx] = other[(idx * 17 + 42) % len(other)]

    # Keep only rows where every role has a same-category, same-split donor.
    matched = [(row, group) for row, group in eligible
               if args.keep_without_donor or all(
                   index[(frozenset(group["ids"]), ref["object_id"])] in donors
                   for ref in row["object_refs"])]
    joint_rows, original_rows = [], []
    private_with_grounding = {key: dict(value) for key, value in private.items()}
    for row, group in matched:
        joint = dict(row)
        joint["token_path"] = "tokens/waymo_tokens.npz"
        joint["object_refs"] = [dict(ref, token_index=index[(frozenset(group["ids"]), ref["object_id"])])
                                for ref in row["object_refs"]]
        joint_rows.append(joint)
        original_rows.append(row)
        private_with_grounding[row["qa_id"]] = dict(
            private[row["qa_id"]],
            diagnostic_grounding_pass=bool(group["pass"]),
            diagnostic_grounding_ious=[float(value) for value in group["ious"]])

    # Keep a paired validation reversal as a separate consistency test. It is
    # never mixed into training or the ordinary validation score.
    swap_eval_rows, swap_eval_private = [], {}
    for row in joint_rows:
        if row["qa_split"] != "val":
            continue
        pair = swapped_pair(row, private_with_grounding[row["qa_id"]])
        if pair is not None:
            swapped, swapped_gt = pair
            swap_eval_rows.append(swapped)
            swap_eval_private[swapped["qa_id"]] = swapped_gt

    output_private = private_with_grounding
    swaps_added = Counter()
    if args.augment_train_swaps:
        joint_rows, output_private, swaps_added = add_missing_train_swaps(
            joint_rows, private_with_grounding)

    shuffled = vectors.copy()
    for idx, donor in donors.items():
        shuffled[idx] = vectors[donor]
    for variant, bank in (("joint", vectors), ("shuffled", shuffled)):
        directory = output / variant
        (directory / "tokens").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "tokens/waymo_tokens.npz",
                            object_token=bank, object_id=np.asarray(ids, dtype=str))
        write_jsonl(directory / "qa.jsonl", joint_rows)
        write_jsonl(directory / "private_gt.jsonl",
                    [output_private[row["qa_id"]] for row in joint_rows])
    swap_dir = output / "swap_eval"
    (swap_dir / "tokens").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(swap_dir / "tokens/waymo_tokens.npz",
                        object_token=vectors, object_id=np.asarray(ids, dtype=str))
    write_jsonl(swap_dir / "qa.jsonl", swap_eval_rows)
    write_jsonl(swap_dir / "private_gt.jsonl",
                [swap_eval_private[row["qa_id"]] for row in swap_eval_rows])
    original_dir = output / "original"
    original_dir.mkdir(exist_ok=True)
    write_jsonl(original_dir / "qa.jsonl", original_rows)
    write_jsonl(original_dir / "private_gt.jsonl", [private[row["qa_id"]] for row in original_rows])

    summary = {
        "threshold": args.iou,
        "token_variant": args.token_variant,
        "eligibility": args.eligibility,
        "keep_without_donor": args.keep_without_donor,
        "augment_train_swaps": args.augment_train_swaps,
        "train_swaps_added": dict(swaps_added),
        "swap_eval_rows": len(swap_eval_rows),
        "splits": list(keep_splits),
        "grounding_samples": len(groups),
        "grounding_targets": dict(counts),
        "grounding_all_correct": sum(group["pass"] for group in groups.values()),
        "qa_multiref_by_split": dict(Counter(row["qa_split"] for row in qa if len(row["object_refs"]) >= 2)),
        "qa_eligible_before_donor": dict(Counter(row["qa_split"] for row, _ in eligible)),
        "qa_matched_before_augmentation": dict(Counter(row["qa_split"] for row, _ in matched)),
        "qa_matched_after_donor": dict(Counter(row["qa_split"] for row, _ in matched)),
        "qa_rows_after_augmentation": dict(Counter(row["qa_split"] for row in joint_rows)),
        "qa_scenarios": dict(Counter(row["scenario"] for row in joint_rows)),
        "answer_distribution": {"{}:{}:{}".format(*key): count
                                for key, count in Counter(
                                    (r["qa_split"], r["scenario"],
                                     output_private[r["qa_id"]]["answer_key"])
                                    for r in joint_rows).items()},
        "token_entries": len(ids),
        "all_shuffled_ids_differ": all(ids[idx] != ids[donor] for idx, donor in donors.items()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
