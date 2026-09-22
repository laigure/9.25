"""Audit scene-level records against raw 3EED Waymo ``ground_info``.

This checks provenance separately from positive-span quality.  A scene record
can preserve the raw caption and box exactly while still choosing the wrong
noun inside a caption that mentions surrounding objects.
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def load_split_map(split_dir):
    result = {}
    for split in ("train", "val"):
        for line in (split_dir / f"waymo_{split}.txt").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                result[line.strip()] = split
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_by_scene = {}
    for sequence, split in load_split_map(args.split_dir).items():
        for meta_path in sorted((args.data_root / sequence).glob("*/meta_info.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            entries = meta.get("ground_info") or []
            if entries:
                scene_id = f"waymo/{sequence}/{meta_path.parent.name}"
                raw_by_scene[(split, scene_id)] = entries

    totals = Counter()
    by_cardinality = Counter()
    examples = []
    for split in ("train", "val"):
        path = args.scene_dir / f"waymo_scene_multi_{split}_info.pkl"
        records = pickle.loads(path.read_bytes())
        for record in records:
            totals["records"] += 1
            scene_id = record["scene_id"]
            raw = raw_by_scene.get((split, scene_id))
            if raw is None:
                totals["missing_raw_scene"] += 1
                continue
            n = record["target_count"]
            by_cardinality[(split, n)] += 1
            if len(raw) != n:
                totals["count_mismatch"] += 1
                continue
            for index, entry in enumerate(raw):
                totals["targets"] += 1
                desc = record["object_descriptions"][index]
                if desc != str(entry.get("caption", "")).strip():
                    totals["caption_mismatch"] += 1
                if record["object_classes"][index] != str(entry.get("class", "")).lower():
                    totals["class_mismatch"] += 1
                built_box = record[f"bbox3d_obj_{index + 1}"]
                if not np.allclose(built_box, entry.get("bbox_3d", []), atol=1e-7):
                    totals["box_mismatch"] += 1

                source = record["target_span_sources"][index]
                word = record["target_span_words"][index]
                totals[f"span_source:{source}"] += 1
                offset = desc.lower().find(word.lower())
                anchors = [desc.lower().find(x) for x in ("there is", "there are")]
                context_start = min((x for x in anchors if x >= 0), default=10**9)
                # These clauses explicitly describe surrounding objects in the
                # released caption template. Selecting a noun after the clause
                # is therefore a high-confidence target-span error.
                if offset > context_start:
                    totals["definite_context_span_errors"] += 1
                    if len(examples) < 25:
                        examples.append({
                            "split": split,
                            "scene_id": scene_id,
                            "target_index": index,
                            "class": record["object_classes"][index],
                            "selected_word": word,
                            "description": desc,
                        })
                if source == "inserted_released_category":
                    totals["synthetic_category_spans"] += 1
                if source == "inserted_released_category" or offset > 120:
                    totals["manual_review_candidates"] += 1

    report = {
        "provenance": {
            "raw_scenes_with_ground_info": len(raw_by_scene),
            "records_checked": totals["records"],
            "targets_checked": totals["targets"],
            "missing_raw_scene": totals["missing_raw_scene"],
            "count_mismatch": totals["count_mismatch"],
            "caption_mismatch": totals["caption_mismatch"],
            "class_mismatch": totals["class_mismatch"],
            "box_mismatch": totals["box_mismatch"],
        },
        "positive_span_audit": {
            "raw_noun_spans": totals["span_source:ground_info_target_noun"],
            "synthetic_category_spans": totals["synthetic_category_spans"],
            "definite_context_span_errors": totals["definite_context_span_errors"],
            "manual_review_candidates": totals["manual_review_candidates"],
            "definition": (
                "definite_context_span_errors selects a noun after a released "
                "'There is/There are' surrounding-object clause"
            ),
        },
        "cardinality": {
            f"{split}_N{n}": count
            for (split, n), count in sorted(by_cardinality.items())
        },
        "error_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
