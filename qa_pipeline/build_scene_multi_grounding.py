"""Build scene-level variable-cardinality Grounding records from 3EED Waymo.

Every released ``ground_info`` entry in a frame becomes one supervised target.
The resulting records contain 1-5 targets in the current public data.  Each
target keeps its own real caption, positive character span, GT box and object
ID.  Unmatched ``others`` boxes are retained as context metadata, but are not
given fabricated referring expressions.
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path


SYNONYMS = {
    "car": ["pickup truck", "utility truck", "delivery truck", "automobile",
            "convertible", "hatchback", "minivan", "vehicle", "sedan",
            "license plate", "range rover", "tool cart", "coupe", "pickup",
            "taxi", "cab", "suv", "mpv", "jeep", "car", "truck", "van"],
    "pedestrian": ["pedestrian", "passerby", "individual", "person", "woman",
                   "people", "walker", "worker", "child", "adult", "lady",
                   "girl", "boy", "man", "guy"],
    "truck": ["concrete mixer truck", "flatbed truck", "cargo truck",
              "semi-truck", "mixer truck", "cement truck", "freight",
              "lorry", "truck"],
    "bus": ["public transport", "school bus", "minibus", "shuttle", "coach",
            "bus"],
    "othervehicle": ["camper van", "excavator", "machinery", "tractor",
                     "trailer", "loader", "truck", "vehicle", "jeep"],
    "cyclist": ["person riding", "bike rider", "bicycle", "cyclist", "biker",
                "rider", "bike"],
}
ROLE_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize(text):
    return " ".join(str(text).replace(",", " ,").split())


def target_noun_span(caption, category):
    matches = []
    for word in SYNONYMS.get(category, [category]):
        matches.extend((m.start(), m.end(), word) for m in re.finditer(
            rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE))
    return min(matches, key=lambda item: (item[0], -(item[1] - item[0]))) if matches else None


def load_split_map(split_dir):
    result = {}
    for split in ("train", "val"):
        for line in (split_dir / f"waymo_{split}.txt").read_text(
                encoding="utf-8").splitlines():
            sequence = line.strip()
            if sequence:
                result[sequence] = split
    return result


def compose_scene(entries):
    text, spans, words, sources = "", [], [], []
    for index, entry in enumerate(entries):
        role = ROLE_NAMES[index]
        category = str(entry.get("class", "")).lower()
        caption = normalize(entry.get("caption", ""))
        if not caption:
            return None
        local = target_noun_span(caption, category)
        prefix = ("" if index == 0 else " ") + f"Object {role}: "
        if local is None:
            # Keep every real target while making the positive span explicit.
            # This adds only its released category, never position or GT geometry.
            prefix += f"({category}) "
            span_start = len(text) + len(prefix) - len(category) - 2
            span_end = span_start + len(category)
            source, word = "inserted_released_category", category
        else:
            span_start = len(text) + len(prefix) + local[0]
            span_end = len(text) + len(prefix) + local[1]
            source, word = "ground_info_target_noun", local[2]
        text += prefix + caption
        spans.append([span_start, span_end])
        words.append(word)
        sources.append(source)
    return text, spans, words, sources


def context_objects(entries):
    seen, result = set(), []
    for source_index, entry in enumerate(entries):
        for other in entry.get("others") or []:
            box = other.get("bbox_3d_other") or []
            category = str(other.get("class_other", "")).lower()
            if len(box) != 7:
                continue
            key = (category,) + tuple(round(float(v), 4) for v in box)
            if key in seen:
                continue
            seen.add(key)
            result.append({"category": category,
                           "box": [float(v) for v in box],
                           "source_ground_index": source_index})
    return result


def build(data_root, split_dir):
    split_map = load_split_map(split_dir)
    outputs = {"train": [], "val": []}
    stats = Counter()
    for sequence, split in sorted(split_map.items()):
        sequence_dir = data_root / sequence
        if not sequence_dir.is_dir():
            stats["missing_sequences"] += 1
            continue
        for meta_path in sorted(sequence_dir.glob("*/meta_info.json")):
            stats["frames_seen"] += 1
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entries = meta.get("ground_info") or []
            if not entries:
                stats["frames_without_targets"] += 1
                continue
            if len(entries) > len(ROLE_NAMES):
                raise ValueError(f"too many targets in {meta_path}: {len(entries)}")
            if any(len(entry.get("bbox_3d") or []) != 7 for entry in entries):
                stats["frames_bad_box"] += 1
                continue
            composed = compose_scene(entries)
            if composed is None:
                stats["frames_missing_caption"] += 1
                continue
            caption, spans, span_words, span_sources = composed
            frame_name = meta_path.parent.name
            scene_id = f"waymo/{sequence}/{frame_name}"
            record = {
                "caption": caption,
                "target_spans": spans,
                "target_span_words": span_words,
                "target_span_sources": span_sources,
                "target_count": len(entries),
                "segment_name": sequence,
                "frame_name": frame_name,
                "scene_id": scene_id,
                "object_ids": [f"{scene_id}#g{i}" for i in range(len(entries))],
                "object_descriptions": [str(entry.get("caption", "")).strip()
                                        for entry in entries],
                "object_classes": [str(entry.get("class", "")).lower()
                                   for entry in entries],
                "source": "all_released_ground_info_in_frame",
                "coordinate_frame": "source_lidar",
                "context_objects": context_objects(entries),
                "timestamp": meta.get("timestamp"),
                "pose": meta.get("pose"),
            }
            for index, entry in enumerate(entries, 1):
                record[f"bbox3d_obj_{index}"] = [float(v) for v in entry["bbox_3d"]]
                if entry.get("bbox_2d") is not None:
                    record[f"bbox2d_obj_{index}"] = entry["bbox_2d"]
                record[f"class_obj_{index}"] = str(entry.get("class", "")).lower()
            outputs[split].append(record)
            stats[f"targets_{len(entries)}"] += 1
            stats["targets_total"] += len(entries)
    return outputs, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("3EED/data/3eed/waymo"))
    parser.add_argument("--split-dir", type=Path,
                        default=Path("3EED/data/splits"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("qa_pipeline/artifacts/scene_multi_grounding"))
    args = parser.parse_args()
    outputs, stats = build(args.data_root, args.split_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "all released ground_info targets in each frame",
        "target_count_is_variable": True,
        "max_target_count": max((r["target_count"] for rows in outputs.values()
                                 for r in rows), default=0),
        "splits": {},
        "counters": dict(stats),
    }
    for split, records in outputs.items():
        path = args.out_dir / f"waymo_scene_multi_{split}_info.pkl"
        with path.open("wb") as handle:
            pickle.dump(records, handle, protocol=4)
        summary["splits"][split] = {
            "records": len(records),
            "targets": sum(r["target_count"] for r in records),
            "target_count_distribution": dict(sorted(Counter(
                r["target_count"] for r in records).items())),
            "scenes": len({r["scene_id"] for r in records}),
            "sequences": len({r["segment_name"] for r in records}),
        }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
