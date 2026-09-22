"""Generate the v1 Waymo spatial-QA records from pair geometry + object bank.

Answers are computed exclusively from GT box geometry (never from predicted
boxes). Each record carries the structured GT (relations, distances, both GT
boxes) next to the question/answer, plus two token pointers so the backend can
later fetch token A (target) and token B (reference) from the consolidated
per-platform npz.

Expression handling: 3EED captions are full descriptive sentences, which read
badly inside a question. Each record keeps the full caption (`*_expression`)
and a heuristic noun-phrase surface form (`*_expression_short`); the question is
built from the short form. The heuristic cuts at the first location/verb marker
(", located", " is positioned", ...) or first sentence end, converts the
structured "Object type: ...; appearance ..." captions, and strips "This is ".

Leakage audit: nearly every caption contains spatial words, so answers often
overlap the question text. Records keep per-word flags on the question so a
leak-controlled subset can be filtered later without regenerating data.

    python build_qa.py --pairs .../qa_v1/pairs_waymo.jsonl \
        --bank .../qa_v1/object_bank.jsonl --out .../qa_v1/qa_waymo.jsonl
"""

import argparse
import json
import re
from collections import Counter

import numpy as np

LOCATION_MARKERS = [
    ", located", ", positioned", ", situated", ", standing", ", facing",
    " located", " positioned", " situated", " standing", " facing",
    " is located", " is positioned", " is situated", " is standing",
    " stands", " is facing", " faces", " is walking", " walking on",
    ", walking in", ". ",
]


def mentions(text, word):
    return bool(re.search(r"\b{}\b".format(re.escape(word)), text, flags=re.IGNORECASE))


def short_expression(text):
    """Heuristic noun-phrase surface form of a 3EED caption (see module doc)."""
    text = " ".join(str(text).split()).strip().rstrip(".")
    if text.lower().startswith("object type:"):
        match = re.match(r"object type:\s*([^,;]+)[,;]?\s*(.*)", text, flags=re.IGNORECASE)
        kind = match.group(1).strip().lower()
        features = ""
        features_match = re.search(r"appearance features?\s*[:;]\s*([^;]+)",
                                   match.group(2), flags=re.IGNORECASE)
        if features_match:
            features = features_match.group(1).strip().rstrip(",; ")
        short = (kind + " " + features).strip()
    else:
        if text.lower().startswith("this is "):
            text = text[len("this is "):]
        cut = len(text)
        for marker in LOCATION_MARKERS:
            position = text.find(marker)
            if 0 < position < cut:
                cut = position
        short = text[:cut].strip().strip(",; ")
    if len(short) < 3:
        return " ".join(str(text).split()).strip().rstrip(".")
    return short


def in_question_form(short):
    """Lowercase a leading determiner so it fits after 'Where is ...'."""
    for determiner in ("The ", "A ", "An ", "This ", "These ", "Their "):
        if short.startswith(determiner):
            return short[0].lower() + short[1:]
    return short


def direction_phrase(relations):
    """Independent-axis relations [lateral, longitudinal] -> English phrase.

    ["left", "front"] -> "to the left and in front of the reference object"
    ["right", "behind"] -> "to the right and behind the reference object"
    """
    lateral = relations[0]
    longitudinal = "in front of" if relations[1] == "front" else "behind"
    return "to the {} and {} the reference object".format(lateral, longitudinal)


def leak_audit(question, relations):
    answer_words = {
        relations[0]: mentions(question, relations[0]),
        relations[1]: mentions(question, "in front" if relations[1] == "front" else "behind"),
    }
    return {
        "question_relation_words": {w: True for w, hit in answer_words.items() if hit},
        "answer_direction_leaked": any(answer_words.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default="/root/autodl-tmp/3eed_data/qa_v1/pairs_waymo.jsonl")
    parser.add_argument("--bank", default="/root/autodl-tmp/3eed_data/qa_v1/object_bank.jsonl")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = args.pairs.replace("pairs_", "qa_")

    bank = {}
    with open(args.bank, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            bank[record["object_id"]] = record

    records, stats = [], Counter()
    with open(args.pairs, encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            target = bank[pair["target_object_id"]]
            reference = bank[pair["reference_object_id"]]
            if target["object_id"] == reference["object_id"]:
                raise ValueError("pair with identical objects: " + pair["pair_id"])
            target_short = short_expression(target["utterance"])
            reference_short = short_expression(reference["utterance"])
            # Re-derive dx/dy in the QA frame and verify against stored geometry.
            recomputed_dx = round(target["gt_box"][0] - reference["gt_box"][0], 4)
            recomputed_dy = round(target["gt_box"][1] - reference["gt_box"][1], 4)
            if (recomputed_dx, recomputed_dy) != (pair["dx"], pair["dy"]):
                raise ValueError("dx/dy mismatch in " + pair["pair_id"])
            # If both short forms collapse to the same text the question would
            # be ambiguous ("the pedestrian relative to the pedestrian"), so
            # that pair falls back to the full captions and is flagged.
            collision = target_short == reference_short
            if collision:
                stats["identical_short_expressions"] += 1
                target_text = target["utterance"].rstrip(".")
                reference_text = reference["utterance"].rstrip(".")
            else:
                target_text = target_short
                reference_text = reference_short
            question = "Where is {} relative to {}?".format(
                in_question_form(target_text), in_question_form(reference_text))
            distance = pair["distance_3d"]
            answer = ("The target object is located {}, approximately {:.1f} meters away."
                      .format(direction_phrase(pair["relations"]), distance))
            audit = leak_audit(question, pair["relations"])
            record = {
                "qa_id": pair["pair_id"],
                "platform": pair["platform"],
                "scene_id": pair["scene_id"],
                "frame_id": pair["frame_id"],
                "split": target["split"],
                "target_object_id": target["object_id"],
                "reference_object_id": reference["object_id"],
                "target_expression": target["utterance"],
                "reference_expression": reference["utterance"],
                "target_expression_short": target_short,
                "reference_expression_short": reference_short,
                "target_token_path": target["token_path"],
                "reference_token_path": reference["token_path"],
                "target_token_index": target["token_index"],
                "reference_token_index": reference["token_index"],
                "token_dim": target["token_dim"],
                "target_gt_box": target["gt_box"],
                "reference_gt_box": reference["gt_box"],
                "relative_geometry": {
                    "dx": pair["dx"], "dy": pair["dy"], "dz": pair["dz"],
                    "distance_2d": pair["distance_2d"],
                    "distance_3d": pair["distance_3d"],
                    "bearing_deg": pair["bearing_deg"],
                },
                "relations": pair["relations"],
                "bearing_sector": pair["bearing_sector"],
                "question": question,
                "answer": answer,
                "expression_collision": collision,
                "leak_audit": audit,
                "meta": {
                    "target_category": target["category"],
                    "reference_category": reference["category"],
                    "target_iou": target["iou"],
                    "reference_iou": reference["iou"],
                    "min_iou": pair["min_iou"],
                    "quality_bucket": pair["quality_bucket"],
                },
                "label_source": "ground_truth_geometry",
                "answer_distance_kind": "center_to_center_3d",
            }
            if audit["answer_direction_leaked"]:
                stats["answer_direction_leaked"] += 1
            stats["total"] += 1
            records.append(record)

    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    distances = np.asarray([r["relative_geometry"]["distance_3d"] for r in records])
    summary = {
        "total": stats["total"],
        "identical_short_expressions": stats["identical_short_expressions"],
        "answer_direction_leaked": stats["answer_direction_leaked"],
        "answer_direction_leaked_rate": round(
            stats["answer_direction_leaked"] / max(stats["total"], 1), 4),
        "relation_pair_counts": dict(Counter(
            "{}_and_{}".format(*r["relations"]) for r in records)),
        "distance_3d_quantiles": {
            "p05": round(float(np.percentile(distances, 5)), 3),
            "median": round(float(np.median(distances)), 3),
            "p95": round(float(np.percentile(distances, 95)), 3),
            "max": round(float(distances.max()), 3),
        },
        "note": "leak flags are audit metadata; filtering policy is decided downstream",
    }
    stats_path = args.out.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", args.out, "and", stats_path)


if __name__ == "__main__":
    main()
