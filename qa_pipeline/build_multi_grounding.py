"""Build a joint multi-object 3EED bootstrap set from neutral GT references.

This is a custom dataset, not the missing official 3EED multi-object release.
Each caption contains two or three independently identified objects in one scene.
The ordered target_spans align exactly with bbox3d_obj_N and class_obj_N.
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import pickle
import random


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "qa_pipeline/artifacts/qa_v2/neutral_bank.jsonl"
QA = ROOT / "qa_pipeline/artifacts/qa_v3/scenario_qa/qa.jsonl"
OUTPUT = ROOT / "qa_pipeline/artifacts/multi_grounding"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def format_caption(text):
    return " " + " ".join(text.replace(",", " ,").split()) + " "


def split_sequences():
    mapping = {}
    for row in read_jsonl(QA):
        sequence, split = row["sequence"], row["qa_split"]
        if sequence in mapping and mapping[sequence] != split:
            raise ValueError(f"Sequence crosses QA splits: {sequence}")
        mapping[sequence] = split
    return mapping


def make_sample(scene_id, members):
    labels = "ABC"
    phrases = [" ".join(m["neutral_query"].replace(",", " ,").split()).strip(" .") for m in members]
    caption = " ".join(f"Object {labels[i]}: {phrase}." for i, phrase in enumerate(phrases))
    formatted = format_caption(caption)
    spans = []
    cursor = 0
    for phrase in phrases:
        start = formatted.find(phrase, cursor)
        if start < 0:
            raise ValueError((scene_id, phrase))
        spans.append([start, start + len(phrase)])
        cursor = start + len(phrase)
    _, segment, frame = scene_id.split("/")
    sample = {
        "caption": caption,
        "target_spans": spans,
        "segment_name": segment,
        "frame_name": frame,
        "scene_id": scene_id,
        "object_ids": [m["object_id"] for m in members],
        "source": "qa_neutral_bank_custom_multi",
    }
    for i, member in enumerate(members, 1):
        sample[f"bbox3d_obj_{i}"] = member["gt_box"]
        sample[f"class_obj_{i}"] = member["category"]
    return sample


def build(seed=42, max_pairs=8, max_triples=3):
    rng = random.Random(seed)
    sequence_split = split_sequences()
    scenes = defaultdict(dict)
    for row in read_jsonl(BANK):
        if row["platform"] != "waymo" or row["sequence"] not in sequence_split:
            continue
        if not row["neutral_query"].strip() or len(row["gt_box"]) != 7:
            continue
        scenes[row["scene_id"]][row["object_id"]] = row
    outputs = {split: [] for split in ("train", "val", "test")}
    for scene_id, objects in sorted(scenes.items()):
        # Same text for two different targets would give indistinguishable text spans.
        unique = {}
        for member in sorted(objects.values(), key=lambda m: m["object_id"]):
            key = " ".join(member["neutral_query"].lower().split()).strip(" .")
            unique.setdefault(key, member)
        members = list(unique.values())
        split = sequence_split[members[0]["sequence"]]
        for count, limit in ((2, max_pairs), (3, max_triples)):
            choices = list(combinations(members, count))
            rng.shuffle(choices)
            for group in choices[:limit]:
                outputs[split].append(make_sample(scene_id, group))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outputs = build(seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, rows in outputs.items():
        with (args.output / f"waymo_multi_{split}_info.pkl").open("wb") as handle:
            pickle.dump(rows, handle, protocol=4)
        counts[split] = {"samples": len(rows), "scenes": len({r["scene_id"] for r in rows}),
                         "targets": dict(Counter(len(r["target_spans"]) for r in rows))}
    (args.output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
