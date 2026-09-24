"""Build native-caption multi-target Grounding records from released 3EED.

One output row comes from exactly one released ``ground_info`` entry.  Its
caption is never concatenated with another caption.  The main box is always
kept, and a nested ``others`` box is kept only when its class has exactly one
remaining noun mention and exactly one box in that entry.  This avoids
guessing which of several same-class context boxes a noun denotes.

Rows with fewer than two unambiguously mentioned targets are discarded, so
the resulting dataset contains no single-target Grounding examples.
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

from audit_native_others import category_mentions, secondary_entity_mentions


VIEW_NAMES = {
    "0": "front view",
    "1": "front-left view",
    "2": "front-right view",
    "3": "side-left view",
    "4": "side-right view",
}
SPATIAL_WORDS = re.compile(
    r"\b(left|right|front|behind|back|ahead|near|nearby|far|next|beside|"
    r"adjacent|located|positioned|situated|upper|lower|middle|center|"
    r"above|below|north|south|east|west)\b", re.IGNORECASE)
PREFIX_STOP = {
    "a", "an", "the", "and", "or", "there", "is", "are", "was", "were",
    "to", "from", "at", "in", "on", "of", "by", "while", "that", "which",
    "driving", "walking", "standing", "moving", "facing", "traveling",
}
FOLLOW_STOP = re.compile(
    r"\b(left|right|front|behind|back|ahead|near|nearby|far|next|beside|"
    r"adjacent|located|positioned|situated|above|below|north|south|east|west|view)\b",
    re.IGNORECASE)


def normalize_caption(text):
    return " ".join(str(text).replace(",", " ,").split())


def load_split_map(split_dir):
    result = {}
    for split in ("train", "val"):
        for line in (split_dir / f"waymo_{split}.txt").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                result[line.strip()] = split
    return result


def select_targets(entry, caption):
    """Return (source, source index, class, span, box) without ambiguity."""
    main_category = str(entry.get("class") or "").lower()
    main_mentions = category_mentions(caption, main_category)
    main_box = entry.get("bbox_3d") or []
    if not main_mentions or len(main_box) != 7:
        return None
    main_span = main_mentions[0]
    selected = [("main", 0, main_category, main_span,
                 [float(value) for value in main_box])]
    boxes_by_category = {}
    for index, item in enumerate(entry.get("others") or []):
        category = str(item.get("class_other") or "").lower()
        box = item.get("bbox_3d_other") or []
        if len(box) == 7:
            boxes_by_category.setdefault(category, []).append(
                (index, [float(value) for value in box]))
    for category, boxes in sorted(boxes_by_category.items()):
        mentions = secondary_entity_mentions(caption, category, main_span)
        if len(boxes) == 1 and len(mentions) == 1:
            (source_index, box), span = boxes[0], mentions[0]
            selected.append(("other", source_index, category, span, box))
    selected.sort(key=lambda item: item[3][0])
    if len({item[3][:2] for item in selected}) != len(selected):
        return None
    return selected if len(selected) >= 2 else None


def safe_description(caption, span, category):
    """Extract an appearance noun phrase while dropping spatial clauses."""
    start, end, _ = span
    before = caption[:start]
    after = caption[end:]
    before_tokens = list(re.finditer(r"[A-Za-z0-9-]+", before))
    prefix = []
    boundary = start
    for match in reversed(before_tokens[-5:]):
        word = match.group(0)
        gap = caption[match.end():boundary]
        if any(char in gap for char in ".,;:?!") or word.lower() in PREFIX_STOP:
            break
        if SPATIAL_WORDS.search(word):
            break
        prefix.append(word)
        boundary = match.start()
    prefix.reverse()
    description = " ".join(prefix + [caption[start:end]])

    # Keep a short non-spatial "with/in/wearing ..." appearance clause.
    tail_match = re.match(
        r"\s+(with|wearing|in)\s+([^,.;?!]{1,70})", after,
        flags=re.IGNORECASE)
    if tail_match:
        tail = tail_match.group(0).strip()
        tail = re.split(r"\b(and|while)\b", tail, maxsplit=1,
                        flags=re.IGNORECASE)[0].strip()
        if not FOLLOW_STOP.search(tail):
            description += " " + tail
    description = " ".join(description.split()).strip(" ,.;:")
    if not description or SPATIAL_WORDS.search(description):
        description = category.replace("othervehicle", "special vehicle")
    return description.lower()


def frame_view(frame_name):
    return VIEW_NAMES.get(frame_name.rsplit("_", 1)[-1], "camera view")


def build(data_root, split_dir):
    outputs = {"train": [], "val": []}
    counters = Counter()
    for sequence, split in sorted(load_split_map(split_dir).items()):
        sequence_dir = data_root / sequence
        if not sequence_dir.is_dir():
            counters["missing_sequences"] += 1
            continue
        for meta_path in sorted(sequence_dir.glob("*/meta_info.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame_name = meta_path.parent.name
            scene_id = f"waymo/{sequence}/{frame_name}"
            for ground_index, entry in enumerate(meta.get("ground_info") or []):
                counters["source_ground_entries"] += 1
                caption = normalize_caption(entry.get("caption") or "")
                selected = select_targets(entry, caption)
                if selected is None:
                    counters["rejected_without_unambiguous_mentioned_other"] += 1
                    continue
                descriptions = [safe_description(caption, item[3], item[2])
                                for item in selected]
                if len(set(descriptions)) != len(descriptions):
                    counters["rejected_duplicate_safe_descriptions"] += 1
                    continue
                if any(SPATIAL_WORDS.search(text) for text in descriptions):
                    counters["rejected_description_spatial_leak"] += 1
                    continue
                record_id = f"{scene_id}#g{ground_index}"
                # Joint3DDataset._format_caption adds one leading space before
                # tokenization.  Store spans in that exact formatted string so
                # each positive map covers the intended noun rather than the
                # character immediately before it.
                formatted_caption = " " + caption + " "
                formatted_spans = [[item[3][0] + 1, item[3][1] + 1]
                                   for item in selected]
                if any(formatted_caption[start:end].casefold() != item[3][2].casefold()
                       for (start, end), item in zip(formatted_spans, selected)):
                    raise ValueError(f"formatted span mismatch: {record_id}")
                record = {
                    "caption": caption,
                    "target_spans": formatted_spans,
                    "target_span_words": [item[3][2] for item in selected],
                    "target_span_sources": [
                        "released_main_noun" if item[0] == "main"
                        else "released_others_noun" for item in selected
                    ],
                    "target_count": len(selected),
                    "segment_name": sequence,
                    "frame_name": frame_name,
                    "scene_id": record_id,
                    "source_scene_id": scene_id,
                    "source_ground_index": ground_index,
                    "object_ids": [
                        f"{record_id}:{item[0]}{item[1]}" for item in selected
                    ],
                    "object_descriptions": descriptions,
                    "object_views": [frame_view(frame_name)] * len(selected),
                    "object_classes": [item[2] for item in selected],
                    "source_target_indices": [
                        {"source": item[0], "index": item[1]} for item in selected
                    ],
                    "source": "released_single_caption_unambiguous_mentioned_others",
                    "coordinate_frame": "source_lidar",
                    "timestamp": meta.get("timestamp"),
                    "pose": meta.get("pose"),
                }
                for index, item in enumerate(selected, 1):
                    record[f"bbox3d_obj_{index}"] = item[4]
                    record[f"class_obj_{index}"] = item[2]
                outputs[split].append(record)
                counters["records"] += 1
                counters["targets"] += len(selected)
                counters[f"{split}_N{len(selected)}"] += 1
    return outputs, counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("3EED/data/3eed/waymo"))
    parser.add_argument("--split-dir", type=Path,
                        default=Path("3EED/data/splits"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs, counters = build(args.data_root, args.split_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_version": "waymo_native_multi_v1",
        "source_policy": (
            "one released ground_info caption; main target plus only uniquely "
            "mentioned nested others; no caption concatenation; no N=1"
        ),
        "splits": {},
        "counters": dict(counters),
        "description_policy": "appearance noun phrases with spatial terms removed",
    }
    for split, records in outputs.items():
        path = args.out_dir / f"waymo_native_multi_{split}_info.pkl"
        with path.open("wb") as handle:
            pickle.dump(records, handle, protocol=4)
        summary["splits"][split] = {
            "records": len(records),
            "targets": sum(row["target_count"] for row in records),
            "target_count_distribution": dict(sorted(Counter(
                row["target_count"] for row in records).items())),
            "sequences": len({row["segment_name"] for row in records}),
            "all_single_caption": all("Object A:" not in row["caption"]
                                      for row in records),
            "single_target_records": sum(row["target_count"] == 1
                                         for row in records),
            "description_spatial_leaks": sum(
                bool(SPATIAL_WORDS.search(description))
                for row in records for description in row["object_descriptions"]),
        }
    (args.out_dir / "waymo_native_multi_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    examples = [row for split in ("train", "val")
                for row in outputs[split][:10]]
    (args.out_dir / "waymo_native_multi_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
