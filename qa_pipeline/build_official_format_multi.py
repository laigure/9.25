"""Rebuild waymo_multi_{split}_info.pkl in the format 3EED's multi loader expects.

The authors never released the official Waymo multi-object annotations
(GitHub issue #2, 2025-11-04: "The Waymo-Multi part will be organized and
released later"; the public data package contains no pkl). This script
reconstructs records from the released single-object annotations: the
per-object captions of two (or three) objects in one frame are merged into
the official "Object A: ... Object B: ..." layout, with exact character
spans so the loader can build one positive map per target.

This is a reconstruction on the released captions, not the official
multi-object annotation. It uses the QA v3 sequence splits so that QA test
sequences never enter grounding training or validation.
"""

import argparse
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "qa_pipeline/artifacts/qa_v2/neutral_bank.jsonl"
QA = ROOT / "qa_pipeline/artifacts/qa_v3/scenario_qa/qa.jsonl"
OUTPUT = ROOT / "qa_pipeline/artifacts/multi_grounding_official_fmt"
LABELS = "ABC"


def format_piece(text):
    """Same whitespace/comma normalization the loader applies to captions."""
    return " ".join(text.replace(",", " ,").split())


def locate_target(piece, phrase, noun):
    """Locate only the referred object phrase inside one full description."""
    phrase = format_piece(phrase).strip(" .")
    if phrase:
        start = piece.casefold().find(phrase.casefold())
        if start >= 0:
            return start, start + len(phrase), "neutral_query"
    for candidate in (noun,):
        candidate = format_piece(candidate).strip(" .")
        if not candidate:
            continue
        match = re.search(rf"\b{re.escape(candidate)}\b", piece, flags=re.IGNORECASE)
        if match:
            return match.start(), match.end(), "matched_noun"
    raise ValueError(f"cannot locate target phrase {phrase!r}/{noun!r} in {piece!r}")


def compose(members):
    """Compose full descriptions with one precise positive span per target."""
    captions = [m["caption"] for m in members]
    pieces = [format_piece(c) for c in captions]
    head = " ".join(
        f"Object {LABELS[i]}: {piece}" for i, piece in enumerate(pieces)
    )
    formatted = " " + head + " "
    cursor = 1
    spans = []
    span_sources = []
    for i, (piece, member) in enumerate(zip(pieces, members)):
        cursor += len(f"Object {LABELS[i]}: ")
        piece_start = cursor
        piece_end = piece_start + len(piece)
        if formatted[piece_start:piece_end] != piece:
            raise ValueError(f"span mismatch for Object {LABELS[i]}")
        local_start, local_end, source = locate_target(
            piece, member["target_phrase"], member["matched_noun"]
        )
        start = piece_start + local_start
        end = piece_start + local_end
        spans.append([start, end])
        span_sources.append(source)
        cursor = piece_end + 1
    return head, spans, span_sources


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def qa_sequence_splits():
    mapping = {}
    for row in read_jsonl(QA):
        sequence, split = row["sequence"], row["qa_split"]
        if mapping.setdefault(sequence, split) != split:
            raise ValueError(f"sequence crosses QA splits: {sequence}")
    return mapping


def scene_objects(bank_path, seq_split, caption_field):
    scenes = defaultdict(list)
    for row in read_jsonl(bank_path):
        if row["platform"] != "waymo" or row["sequence"] not in seq_split:
            continue
        caption = (row.get(caption_field) or "").strip()
        if not caption or len(row["gt_box"]) != 7:
            continue
        scenes[row["scene_id"]].append(
            {
                "object_id": row["object_id"],
                "category": row["category"],
                "caption": caption,
                "target_phrase": row.get("neutral_query", ""),
                "matched_noun": row.get("matched_noun", row["category"]),
                "gt_box": [float(x) for x in row["gt_box"]],
            }
        )
    return scenes


def make_record(scene_id, members):
    _, segment, frame = scene_id.split("/")
    captions = [m["caption"] for m in members]
    if len(set(captions)) != len(captions):
        return None
    try:
        head, spans, span_sources = compose(members)
    except ValueError:
        return None
    record = {
        "caption": head,
        "target_spans": spans,
        "target_span_sources": span_sources,
        "segment_name": segment,
        "frame_name": frame,
        "scene_id": scene_id,
        "object_ids": [m["object_id"] for m in members],
        "source": "released_single_captions_objectABC_rebuild",
        # Boxes and points come directly from the released per-view frame.
        # The multi loader must rotate both to the canonical front view, just
        # like the released single-object Waymo path does.
        "coordinate_frame": "source_lidar",
    }
    for i, member in enumerate(members, 1):
        record[f"bbox3d_obj_{i}"] = member["gt_box"]
        record[f"class_obj_{i}"] = member["category"]
    return record


def build(bank_path, qa_path, caption_field, max_pairs, max_triples, seed):
    seq_split = qa_sequence_splits()
    scenes = scene_objects(bank_path, seq_split, caption_field)
    rng = random.Random(seed)
    outputs = {split: [] for split in ("train", "val", "test")}
    seen = set()
    for scene_id, members in sorted(scenes.items()):
        # One object per distinct caption: identical text cannot identify a target.
        unique = {}
        for member in sorted(members, key=lambda m: m["object_id"]):
            key = format_piece(member["caption"]).casefold()
            unique.setdefault(key, member)
        members = list(unique.values())
        split = seq_split[members[0]["object_id"].split("/")[1]]
        ordered = sorted(members, key=lambda m: m["object_id"])
        candidates = []
        for count, limit in ((2, max_pairs), (3, max_triples)):
            if count > len(ordered) or limit <= 0:
                continue
            groups = list(combinations(ordered, count))
            rng.shuffle(groups)
            candidates.extend(groups[:limit])
        for group in candidates:
            key = tuple(m["object_id"] for m in group)
            if key in seen:
                continue
            record = make_record(scene_id, group)
            if record is None:
                continue
            seen.add(key)
            outputs[split].append(record)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--qa", type=Path, default=QA)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--caption-field",
        choices=["source_caption", "neutral_query"],
        default="source_caption",
    )
    parser.add_argument("--max-pairs-per-scene", type=int, default=4)
    parser.add_argument("--max-triples-per-scene", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outputs = build(
        args.bank,
        args.qa,
        args.caption_field,
        args.max_pairs_per_scene,
        args.max_triples_per_scene,
        args.seed,
    )

    sequences = {}
    summary = {}
    for split, rows in outputs.items():
        for record in rows:
            sequences.setdefault(split, set()).add(
                record["object_ids"][0].split("/")[1]
            )
        counts = Counter(len(r["target_spans"]) for r in rows)
        pair_classes = Counter(
            tuple(sorted(r[f"class_obj_{i}"] for i in range(1, len(r["target_spans"]) + 1)))
            for r in rows
        )
        summary[split] = {
            "records": len(rows),
            "scenes": len({r["scene_id"] for r in rows}),
            "sequences": len(sequences[split]),
            "targets": {str(k): v for k, v in sorted(counts.items())},
            "class_sets": {"+".join(k): v for k, v in pair_classes.most_common(12)},
            "caption_field": args.caption_field,
            "coordinate_frame": "source_lidar",
        }
        for record in rows:
            caption = " " + " ".join(record["caption"].replace(",", " ,").split()) + " "
            for (begin, end) in record["target_spans"]:
                if not (0 <= begin < end <= len(caption)):
                    raise ValueError((record["scene_id"], begin, end, len(caption)))
                if caption[begin:end] != " ".join(
                    caption[begin:end].split()
                ):
                    raise ValueError("span is not whitespace-normalized")

    train_set = sequences.get("train", set())
    for split in ("val", "test"):
        overlap = train_set & sequences.get(split, set())
        if overlap:
            raise ValueError(f"sequence overlap train/{split}: {sorted(overlap)[:3]}")
    if sequences.get("val", set()) & sequences.get("test", set()):
        raise ValueError("sequence overlap val/test")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in outputs.items():
        path = args.out_dir / f"waymo_multi_{split}_info.pkl"
        with path.open("wb") as handle:
            pickle.dump(rows, handle, protocol=4)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
