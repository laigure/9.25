"""Build two-target 3EED records from ground_info -> others links.

Each retained `others` box must match another `ground_info` object in the same
frame. This gives both boxes an authentic caption and lets us create one
distinct positive text span per target. Context-only `others` entries are
counted but excluded because they have no referring expression in the released
metadata.
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np


SYNONYMS = {
    "car": ["pickup truck", "utility truck", "delivery truck", "automobile", "convertible",
            "hatchback", "minivan", "vehicle", "sedan", "coupe", "pickup", "taxi", "cab",
            "suv", "car", "van"],
    "pedestrian": ["pedestrian", "passerby", "individual", "person", "woman", "people",
                   "walker", "worker", "child", "adult", "lady", "girl", "boy", "man", "guy"],
    "truck": ["concrete mixer truck", "flatbed truck", "cargo truck", "semi-truck",
              "mixer truck", "cement truck", "freight", "lorry", "truck"],
    "bus": ["public transport", "school bus", "minibus", "shuttle", "coach", "bus"],
    "othervehicle": ["machinery", "tractor", "trailer", "vehicle", "jeep"],
    "cyclist": ["person riding", "bike rider", "bicycle", "cyclist", "biker", "rider", "bike"],
}
LABELS = "AB"


def normalize(text):
    return " ".join(str(text).replace(",", " ,").split())


def target_noun_span(caption, category):
    """Return the earliest loader-compatible category mention."""
    matches = []
    for word in SYNONYMS.get(category, [category]):
        for match in re.finditer(rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE):
            matches.append((match.start(), match.end(), word))
    if not matches:
        return None
    return min(matches, key=lambda item: (item[0], -(item[1] - item[0])))


def match_ground_entry(other, entries, source_index, tolerance):
    other_class = str(other.get("class_other", "")).lower()
    other_box = np.asarray(other.get("bbox_3d_other", []), dtype=np.float64)
    if other_box.shape != (7,):
        return None
    candidates = []
    for index, entry in enumerate(entries):
        if index == source_index or str(entry.get("class", "")).lower() != other_class:
            continue
        box = np.asarray(entry.get("bbox_3d", []), dtype=np.float64)
        if box.shape != (7,):
            continue
        center_distance = float(np.linalg.norm(box[:3] - other_box[:3]))
        full_error = float(np.max(np.abs(box - other_box)))
        if center_distance <= tolerance and full_error <= tolerance:
            candidates.append((full_error, center_distance, index))
    if len(candidates) != 1:
        return None
    return min(candidates)[2]


def compose_pair(first, second):
    pieces = [normalize(first["caption"]), normalize(second["caption"])]
    categories = [first["category"], second["category"]]
    text = ""
    spans = []
    span_words = []
    for index, (piece, category) in enumerate(zip(pieces, categories)):
        prefix = ("" if index == 0 else " ") + f"Object {LABELS[index]}: "
        text += prefix
        piece_start = len(text)
        text += piece
        local = target_noun_span(piece, category)
        if local is None:
            return None
        spans.append([piece_start + local[0], piece_start + local[1]])
        span_words.append(local[2])
    return text, spans, span_words


def load_split_map(split_dir):
    result = {}
    for split in ("train", "val"):
        path = split_dir / f"waymo_{split}.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            sequence = line.strip()
            if sequence:
                result[sequence] = split
    return result


def build(data_root, split_dir, tolerance):
    split_map = load_split_map(split_dir)
    outputs = {"train": [], "val": []}
    stats = Counter()
    unmatched_classes = Counter()
    seen_pairs = set()
    for sequence, split in sorted(split_map.items()):
        sequence_dir = data_root / sequence
        if not sequence_dir.is_dir():
            stats["missing_sequences"] += 1
            continue
        for meta_path in sorted(sequence_dir.glob("*/meta_info.json")):
            stats["frames"] += 1
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entries = meta.get("ground_info") or []
            stats["ground_entries"] += len(entries)
            frame_name = meta_path.parent.name
            for source_index, entry in enumerate(entries):
                others = entry.get("others") or []
                stats["others_edges"] += len(others)
                for other_index, other in enumerate(others):
                    target_index = match_ground_entry(other, entries, source_index, tolerance)
                    if target_index is None:
                        stats["unmatched_other_edges"] += 1
                        unmatched_classes[str(other.get("class_other", "")).lower()] += 1
                        continue
                    stats["matched_other_edges"] += 1
                    pair_key = (sequence, frame_name, *sorted((source_index, target_index)))
                    if pair_key in seen_pairs:
                        stats["duplicate_reverse_edges"] += 1
                        continue
                    seen_pairs.add(pair_key)
                    members = []
                    for index in (source_index, target_index):
                        item = entries[index]
                        members.append({
                            "entry_index": index,
                            "category": str(item.get("class", "")).lower(),
                            "caption": str(item.get("caption") or "").strip(),
                            "box": [float(value) for value in item.get("bbox_3d", [])],
                        })
                    if any(len(member["box"]) != 7 or not member["caption"] for member in members):
                        stats["bad_member"] += 1
                        continue
                    composed = compose_pair(*members)
                    if composed is None:
                        stats["missing_target_noun"] += 1
                        continue
                    caption, target_spans, span_words = composed
                    record = {
                        "caption": caption,
                        "target_spans": target_spans,
                        "target_span_sources": ["ground_info_target_noun"] * 2,
                        "target_span_words": span_words,
                        "segment_name": sequence,
                        "frame_name": frame_name,
                        "scene_id": f"waymo/{sequence}/{frame_name}",
                        "object_ids": [
                            f"waymo/{sequence}/{frame_name}#g{member['entry_index']}"
                            for member in members
                        ],
                        "others_link": {
                            "source_ground_index": source_index,
                            "source_other_index": other_index,
                            "matched_ground_index": target_index,
                        },
                        "source": "released_ground_info_others_linked_pair",
                        "coordinate_frame": "source_lidar",
                    }
                    for index, member in enumerate(members, 1):
                        record[f"bbox3d_obj_{index}"] = member["box"]
                        record[f"class_obj_{index}"] = member["category"]
                    outputs[split].append(record)
    stats["unique_pairs"] = sum(map(len, outputs.values()))
    return outputs, stats, unmatched_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("3EED/data/3eed/waymo"))
    parser.add_argument("--split-dir", type=Path, default=Path("3EED/data/splits"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("qa_pipeline/artifacts/multi_grounding_others"))
    parser.add_argument("--box-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    outputs, stats, unmatched_classes = build(args.data_root, args.split_dir, args.box_tolerance)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "main ground_info target plus an others box matched to another ground_info target",
        "targets_per_record": 2,
        "splits": {},
        "counters": dict(stats),
        "unmatched_other_classes": dict(unmatched_classes.most_common()),
    }
    for split, records in outputs.items():
        with (args.out_dir / f"waymo_others_multi_{split}_info.pkl").open("wb") as handle:
            pickle.dump(records, handle, protocol=4)
        summary["splits"][split] = {
            "records": len(records),
            "scenes": len({record["scene_id"] for record in records}),
            "sequences": len({record["segment_name"] for record in records}),
            "class_pairs": dict(Counter(
                "+".join(sorted((record["class_obj_1"], record["class_obj_2"])))
                for record in records
            ).most_common()),
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
