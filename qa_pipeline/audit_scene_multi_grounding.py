"""Verify scene-level records against the original 3EED meta_info files."""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


def close_list(left, right, tolerance=1e-6):
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def audit(data_root, records_dir):
    counts = Counter()
    mismatch = Counter()
    inserted = []
    duplicate_caption_scenes = []
    for split in ("train", "val"):
        path = records_dir / f"waymo_scene_multi_{split}_info.pkl"
        with path.open("rb") as handle:
            records = pickle.load(handle)
        for record in records:
            meta_path = (
                data_root
                / record["segment_name"]
                / record["frame_name"]
                / "meta_info.json"
            )
            if not meta_path.exists():
                mismatch["missing_meta"] += 1
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entries = meta.get("ground_info") or []
            count = len(entries)
            counts[f"{split}_scenes_n{count}"] += 1
            counts[f"{split}_targets"] += count
            if count != record["target_count"]:
                mismatch["target_count"] += 1
            source_descriptions = [str(e.get("caption", "")).strip() for e in entries]
            source_classes = [str(e.get("class", "")).lower() for e in entries]
            if source_descriptions != record["object_descriptions"]:
                mismatch["description"] += 1
            if source_classes != record["object_classes"]:
                mismatch["class"] += 1
            normalized = [text.lower() for text in source_descriptions]
            if len(normalized) != len(set(normalized)):
                duplicate_caption_scenes.append(record["scene_id"])
            for index, entry in enumerate(entries):
                if not close_list(
                    record[f"bbox3d_obj_{index + 1}"], entry.get("bbox_3d") or []
                ):
                    mismatch["bbox_3d"] += 1
                begin, end = record["target_spans"][index]
                span_text = record["caption"][begin:end]
                if span_text.lower() != record["target_span_words"][index].lower():
                    mismatch["span_text"] += 1
                if record["target_span_sources"][index] != "ground_info_target_noun":
                    inserted.append(
                        {
                            "split": split,
                            "scene_id": record["scene_id"],
                            "target_index": index,
                            "class": source_classes[index],
                            "description": source_descriptions[index],
                        }
                    )
    return {
        "counts": dict(sorted(counts.items())),
        "mismatches": dict(sorted(mismatch.items())),
        "duplicate_caption_scenes": duplicate_caption_scenes,
        "inserted_positive_span_count": len(inserted),
        "inserted_positive_spans": inserted,
        "exact_source_alignment_pass": not mismatch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_root, args.records_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
