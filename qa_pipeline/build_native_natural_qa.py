"""Build natural-language spatial QA from native-caption multi-target 3EED.

Public rows contain natural appearance descriptions, view names and opaque
object IDs.  They contain no Object-A/B aliases, GT coordinates, boxes,
categories or answer labels.  Geometry remains private and produces the
canonical full-sentence answers and training-only auxiliary relation labels.
"""

import argparse
import hashlib
import itertools
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np


VIEW_CODES = {"0": "F", "1": "FL", "2": "FR", "3": "SL", "4": "SR"}
ROTATION_DEGREES = {"F": 0.0, "FL": -45.0, "FR": 45.0,
                    "SL": -90.0, "SR": 90.0}
MOVE_VECTORS = {
    "forward": np.array([1.0, 0.0]),
    "backward": np.array([-1.0, 0.0]),
    "left": np.array([0.0, 1.0]),
    "right": np.array([0.0, -1.0]),
}
SPATIAL_LEAK = re.compile(
    r"\b(left|right|front|behind|back|ahead|near|nearby|far|next|beside|"
    r"adjacent|located|positioned|situated|upper|lower|middle|north|south|"
    r"above|below|east|west)\b", re.IGNORECASE)
ROLE_ALIAS = re.compile(r"\bobject\s+[a-z]\b", re.IGNORECASE)
CHOICE_ALIAS = re.compile(r"\b[ABCD][.)]\s*(front|back|left|right)\b",
                          re.IGNORECASE)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def canonical_boxes(record):
    boxes = np.asarray([record[f"bbox3d_obj_{i + 1}"]
                        for i in range(record["target_count"])], dtype=np.float64)
    view = VIEW_CODES[record["frame_name"].rsplit("_", 1)[-1]]
    angle = math.radians(ROTATION_DEGREES[view])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    boxes[:, :2] = boxes[:, :2] @ rotation.T
    boxes[:, 6] += angle
    return boxes


def entity(description, view):
    return f"the {description} in the {view}"


def sentence_entity(description, view):
    text = entity(description, view)
    return text[0].upper() + text[1:]


def cardinal_relation(subject_center, reference_center):
    dx, dy = (subject_center[:2] - reference_center[:2]).tolist()
    if max(abs(dx), abs(dy)) < 1.0:
        return None
    if abs(dx) >= abs(dy):
        return "front" if dx > 0 else "behind"
    return "left" if dy > 0 else "right"


def relation_answer(subject, subject_view, relation, reference, reference_view):
    s = sentence_entity(subject, subject_view)
    o = entity(reference, reference_view)
    if relation == "front":
        return f"{s} is in front of {o}."
    if relation == "behind":
        return f"{s} is behind {o}."
    return f"{s} is to the {relation} of {o}."


def public_refs(record):
    return [
        {"slot": index, "description": description, "view": view,
         "object_id": object_id}
        for index, (description, view, object_id) in enumerate(zip(
            record["object_descriptions"], record["object_views"],
            record["object_ids"]))
    ]


def base_row(record, split, scenario, category, question, answer, key_parts):
    return {
        "qa_id": stable_id(record["scene_id"], scenario, *key_parts),
        "qa_split": split,
        "scenario": scenario,
        "question_category": category,
        "target_count": record["target_count"],
        "scene_id": record["scene_id"],
        "source_scene_id": record["source_scene_id"],
        "sequence": record["segment_name"],
        "question": question,
        "answer": answer,
        "object_refs": public_refs(record),
        "input_policy": "natural_descriptions_plus_implicit_grounding_tokens",
    }


def private_row(row, record, boxes, answer_key):
    return {
        "qa_id": row["qa_id"],
        "evaluation_schema": "natural_svo_v1",
        "canonical_answer": row["answer"],
        "answer_key": answer_key,
        "gt_geometry": boxes.tolist(),
        "grounding_inputs": [
            {"object_id": object_id, "category": category,
             "source_description": description, "view": view}
            for object_id, category, description, view in zip(
                record["object_ids"], record["object_classes"],
                record["object_descriptions"], record["object_views"])
        ],
    }


def add(rows, private, record, split, boxes, scenario, category,
        question, answer, key_parts, answer_key):
    row = base_row(record, split, scenario, category, question, answer, key_parts)
    if row["qa_id"] in private:
        raise ValueError(f"duplicate QA id {row['qa_id']}")
    rows.append(row)
    private[row["qa_id"]] = private_row(row, record, boxes, answer_key)


def build_record(record, split, rows, private, motion_meters):
    boxes = canonical_boxes(record)
    descriptions = record["object_descriptions"]
    views = record["object_views"]
    ids = record["object_ids"]
    count = len(ids)

    # Open-ended cardinal relation, with two natural question templates.
    for subject, reference in itertools.permutations(range(count), 2):
        relation = cardinal_relation(boxes[subject, :3], boxes[reference, :3])
        if relation is None:
            continue
        s, o = entity(descriptions[subject], views[subject]), entity(
            descriptions[reference], views[reference])
        questions = [
            ("The front view corresponds to north. If you stand at the location "
             f"of {o} facing north, where is {s} relative to you? Answer in "
             "one complete sentence."),
            ("The front view corresponds to north. From the perspective of "
             f"{o} while facing north, describe where {s} is located in one "
             "complete sentence."),
        ]
        answer = relation_answer(descriptions[subject], views[subject], relation,
                                 descriptions[reference], views[reference])
        for variant, question in enumerate(questions):
            pair_id = stable_id(record["scene_id"], "inverse_relation",
                                min(subject, reference), max(subject, reference), variant)
            add(rows, private, record, split, boxes, "cardinal_relation",
                "spatial_relation", question, answer,
                [subject, reference, variant], {
                    "type": "spatial_triple", "subject_id": ids[subject],
                    "subject": descriptions[subject], "relation": relation,
                    "object_id": ids[reference], "object": descriptions[reference],
                    "inverse_pair_id": pair_id,
                })

    # Absolute distance from ego for every target.
    for subject in range(count):
        distance = float(np.linalg.norm(boxes[subject, :2]))
        s = entity(descriptions[subject], views[subject])
        question = (f"How far, in meters, is {s} from the ego car? Answer in "
                    "one complete sentence.")
        answer = (f"{sentence_entity(descriptions[subject], views[subject])} is "
                  f"{distance:.1f} meters from the ego car.")
        add(rows, private, record, split, boxes, "ego_absolute_distance",
            "distance", question, answer, [subject], {
                "type": "metric_distance", "subject_id": ids[subject],
                "subject": descriptions[subject], "relation": "distance_from",
                "object": "ego car", "meters": round(distance, 1),
            })

    # Pairwise absolute distance and relative-to-ego comparison.
    for first, second in itertools.combinations(range(count), 2):
        first_entity = entity(descriptions[first], views[first])
        second_entity = entity(descriptions[second], views[second])
        distance = float(np.linalg.norm(boxes[first, :2] - boxes[second, :2]))
        question = (f"How far, in meters, is {first_entity} from {second_entity}? "
                    "Answer in one complete sentence.")
        answer = (f"The distance from {first_entity} to {second_entity} is "
                  f"{distance:.1f} meters.")
        add(rows, private, record, split, boxes, "object_absolute_distance",
            "distance", question, answer, [first, second], {
                "type": "metric_distance", "subject_id": ids[first],
                "subject": descriptions[first], "relation": "distance_from",
                "object_id": ids[second], "object": descriptions[second],
                "meters": round(distance, 1),
            })

        ego_distances = [float(np.linalg.norm(boxes[index, :2]))
                         for index in (first, second)]
        winner, loser = ((first, second) if ego_distances[0] < ego_distances[1]
                         else (second, first))
        question = (f"Is {first_entity} closer to the ego car than "
                    f"{second_entity}? Explain the comparison in one complete "
                    "sentence.")
        answer = (f"{sentence_entity(descriptions[winner], views[winner])} is "
                  f"closer to the ego car than {entity(descriptions[loser], views[loser])} is.")
        add(rows, private, record, split, boxes, "ego_relative_distance",
            "distance", question, answer, [first, second], {
                "type": "relative_distance", "subject_id": ids[winner],
                "subject": descriptions[winner], "relation": "closer_to",
                "object": "ego car", "comparison_id": ids[loser],
                "comparison": descriptions[loser],
            })

    # Object-centric relative distance for every anchor and every candidate pair.
    if count >= 3:
        for anchor in range(count):
            candidates = [index for index in range(count) if index != anchor]
            for first, second in itertools.combinations(candidates, 2):
                d1 = float(np.linalg.norm(boxes[first, :2] - boxes[anchor, :2]))
                d2 = float(np.linalg.norm(boxes[second, :2] - boxes[anchor, :2]))
                if abs(d1 - d2) < 0.5:
                    continue
                winner, loser = (first, second) if d1 < d2 else (second, first)
                question = (f"Which is closer to {entity(descriptions[anchor], views[anchor])}, "
                            f"{entity(descriptions[first], views[first])} or "
                            f"{entity(descriptions[second], views[second])}? Answer "
                            "in one complete sentence.")
                answer = (f"{sentence_entity(descriptions[winner], views[winner])} is "
                          f"closer to {entity(descriptions[anchor], views[anchor])} "
                          f"than {entity(descriptions[loser], views[loser])} is.")
                add(rows, private, record, split, boxes,
                    "object_relative_distance", "distance", question, answer,
                    [anchor, first, second], {
                        "type": "relative_distance", "subject_id": ids[winner],
                        "subject": descriptions[winner], "relation": "closer_to",
                        "object_id": ids[anchor], "object": descriptions[anchor],
                        "comparison_id": ids[loser],
                        "comparison": descriptions[loser],
                    })

    # Ego counterfactual motion: all four directions prevent a fixed label prior.
    origin = np.zeros(2, dtype=np.float64)
    for target in range(count):
        before = float(np.linalg.norm(boxes[target, :2] - origin))
        for direction, unit in MOVE_VECTORS.items():
            moved_ego = origin + motion_meters * unit
            after = float(np.linalg.norm(boxes[target, :2] - moved_ego))
            closer = after < before
            target_entity = entity(descriptions[target], views[target])
            question = ("The front view corresponds to north. If the ego car "
                        f"moves {motion_meters:g} meters {direction} while all "
                        f"other objects remain stationary, does it get closer to "
                        f"{target_entity}? Answer in one complete sentence.")
            predicate = "gets closer to" if closer else "does not get closer to"
            answer = (f"After moving {motion_meters:g} meters {direction}, the "
                      f"ego car {predicate} {target_entity}.")
            add(rows, private, record, split, boxes, "ego_motion_reasoning",
                "ego_motion", question, answer, [target, direction], {
                    "type": "motion_relation", "subject": "ego car",
                    "relation": "gets_closer_to" if closer else "does_not_get_closer_to",
                    "object_id": ids[target], "object": descriptions[target],
                    "motion_direction": direction, "motion_meters": motion_meters,
                })

    # Object counterfactual motion for every ordered object pair.
    for mover, reference in itertools.permutations(range(count), 2):
        before = float(np.linalg.norm(boxes[mover, :2] - boxes[reference, :2]))
        for direction, unit in MOVE_VECTORS.items():
            moved = boxes[mover, :2] + motion_meters * unit
            after = float(np.linalg.norm(moved - boxes[reference, :2]))
            closer = after < before
            mover_entity = entity(descriptions[mover], views[mover])
            reference_entity = entity(descriptions[reference], views[reference])
            question = ("The front view corresponds to north. If "
                        f"{mover_entity} moves {motion_meters:g} meters "
                        f"{direction} while all other objects remain stationary, "
                        f"does it get closer to {reference_entity}? Answer in one "
                        "complete sentence.")
            predicate = "gets closer to" if closer else "does not get closer to"
            answer = (f"After moving {motion_meters:g} meters {direction}, "
                      f"{mover_entity} {predicate} {reference_entity}.")
            answer = answer[0].upper() + answer[1:]
            add(rows, private, record, split, boxes, "object_motion_reasoning",
                "object_motion", question, answer,
                [mover, reference, direction], {
                    "type": "motion_relation", "subject_id": ids[mover],
                    "subject": descriptions[mover],
                    "relation": "gets_closer_to" if closer else "does_not_get_closer_to",
                    "object_id": ids[reference], "object": descriptions[reference],
                    "motion_direction": direction, "motion_meters": motion_meters,
                })


def audit(rows, private):
    counters = Counter()
    seen_inputs = set()
    label_by_scenario_direction = Counter()
    for row in rows:
        counters["rows"] += 1
        counters[f"split:{row['qa_split']}"] += 1
        counters[f"category:{row['question_category']}"] += 1
        counters[f"scenario:{row['scenario']}"] += 1
        counters[f"N:{row['target_count']}"] += 1
        if ROLE_ALIAS.search(row["question"] + " " + row["answer"]):
            counters["object_alias_leaks"] += 1
        if CHOICE_ALIAS.search(row["question"]):
            counters["multiple_choice_leaks"] += 1
        if any(SPATIAL_LEAK.search(ref["description"])
               for ref in row["object_refs"]):
            counters["description_spatial_leaks"] += 1
        if re.search(r"[-+]?\d+\.\d+\s*,\s*[-+]?\d+\.\d+", row["question"]):
            counters["coordinate_leaks"] += 1
        if not row["answer"].endswith("."):
            counters["non_sentence_answers"] += 1
        signature = (row["qa_split"], row["question"],
                     tuple(ref["object_id"] for ref in row["object_refs"]))
        if signature in seen_inputs:
            counters["duplicate_model_inputs"] += 1
        seen_inputs.add(signature)
        key = private[row["qa_id"]]["answer_key"]
        if key["type"] == "motion_relation":
            label_by_scenario_direction[(row["scenario"],
                                         key["motion_direction"],
                                         key["relation"])] += 1
    required_zero = ["object_alias_leaks", "multiple_choice_leaks",
                     "description_spatial_leaks", "coordinate_leaks",
                     "non_sentence_answers", "duplicate_model_inputs"]
    for name in required_zero:
        if counters[name]:
            raise ValueError(f"QA audit failed: {name}={counters[name]}")
    return {
        "counts": dict(counters),
        "motion_label_distribution": {
            "|".join(key): value
            for key, value in sorted(label_by_scenario_direction.items())
        },
        "strict_zero_checks": required_zero,
    }


def balance_categories(rows, private):
    """Balance the four requested QA categories within split and target count.

    Motion templates naturally outnumber relation and distance templates.  A
    stable hash subset prevents that template count from becoming a shortcut
    while preserving both ego-motion and object-motion supervision.
    """
    categories = ("spatial_relation", "distance", "ego_motion", "object_motion")
    grouped = {}
    for row in rows:
        grouped.setdefault((row["qa_split"], row["target_count"],
                            row["question_category"]), []).append(row)
    selected = []
    report = {}
    split_counts = sorted({(row["qa_split"], row["target_count"]) for row in rows})
    for split, target_count in split_counts:
        groups = [grouped.get((split, target_count, category), [])
                  for category in categories]
        if any(not group for group in groups):
            raise ValueError(
                f"missing QA category for {split} N={target_count}: "
                f"{dict(zip(categories, map(len, groups)))}")
        quota = min(map(len, groups))
        report[f"{split}_N{target_count}"] = {
            "quota_per_category": quota,
            "before": dict(zip(categories, map(len, groups))),
        }
        for category, group in zip(categories, groups):
            selected.extend(sorted(
                group, key=lambda row: stable_id("balance", row["qa_id"]))[:quota])
    selected.sort(key=lambda row: (row["qa_split"], row["scene_id"], row["qa_id"]))
    kept_ids = {row["qa_id"] for row in selected}
    return selected, {qa_id: value for qa_id, value in private.items()
                      if qa_id in kept_ids}, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-meters", type=float, default=5.0)
    args = parser.parse_args()
    rows, private = [], {}
    source_counts = Counter()
    for split in ("train", "val"):
        path = args.annotations / f"waymo_native_multi_{split}_info.pkl"
        with path.open("rb") as handle:
            records = pickle.load(handle)
        for record in records:
            source_counts[f"{split}_N{record['target_count']}"] += 1
            build_record(record, split, rows, private, args.motion_meters)
    raw_rows = len(rows)
    rows, private, balance_report = balance_categories(rows, private)
    report = audit(rows, private)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "qa.jsonl", rows)
    write_jsonl(args.output / "private_gt.jsonl",
                [private[row["qa_id"]] for row in rows])
    examples = {}
    for row in rows:
        examples.setdefault(row["scenario"], row)
    (args.output / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary = {
        "dataset_version": "native_natural_qa_v1",
        "source_grounding_records": dict(source_counts),
        "rows": len(rows),
        "rows_before_category_balance": raw_rows,
        "category_balance": balance_report,
        "motion_meters": args.motion_meters,
        "public_input": (
            "natural non-spatial descriptions + view names + ordered implicit "
            "Grounding tokens; no Object-A/B aliases and no explicit geometry"
        ),
        "answer_format": "one canonical complete sentence with private SVO key",
        "audit": report,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
