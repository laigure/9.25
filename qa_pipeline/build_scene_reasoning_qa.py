"""Build balanced scene QA from private 3D annotations.

Public examples contain generic questions, ordered object roles, and opaque
object IDs used to attach 288D Grounding tokens. Captions, categories, boxes,
coordinates, relations, distances, and answer keys remain private.

Questions are divided into distance, direction, and obstacle avoidance. A
three-object scene receives several paraphrases and every object is rotated
through the subject/anchor role to reduce the N=2 versus N=3 imbalance.
"""

import argparse
import hashlib
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROLES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATASET_VERSION = "scene_reasoning_qa_v3_balanced_categories"
INVERSE = {"left": "right", "right": "left", "front": "behind",
           "behind": "front"}
QUESTION_CATEGORIES = ("distance", "direction", "obstacle_avoidance")
DISTANCE_THRESHOLD_M = 10.0

PAIR_DIRECTION_TEMPLATES = (
    "Considering all grounded objects, where is Object {subject} relative to "
    "Object {reference}? Answer in exactly one sentence.",
    "Taking Object {reference} as the reference, describe the direction of "
    "Object {subject} in exactly one sentence.",
    "Use the grounding tokens to state Object {subject}'s spatial position with "
    "respect to Object {reference}, using exactly one sentence.",
)
MULTI_DIRECTION_TEMPLATES = (
    "State where Object {subject} is relative to every other grounded object, "
    "using exactly one sentence.",
    "Taking Object {subject} as the subject, report its directional relationship "
    "to each other grounded object in exactly one sentence.",
    "Describe Object {subject}'s spatial position with respect to all remaining "
    "grounded objects in exactly one sentence.",
)
DISTANCE_BAND_TEMPLATES = (
    "Is Object {subject} within 10 meters of the ego vehicle? State the result "
    "in exactly one sentence.",
    "Determine whether Object {subject} is at most 10 meters from the ego vehicle "
    "or farther away, and answer in exactly one sentence.",
    "Classify Object {subject}'s distance from the ego vehicle using the 10-meter "
    "boundary, with exactly one sentence.",
)
ANCHOR_DISTANCE_TEMPLATES = (
    "Which other grounded object is {extreme} Object {anchor}? Answer in exactly "
    "one sentence.",
    "Taking Object {anchor} as the distance reference, identify the {extreme} "
    "other object in exactly one sentence.",
    "Among the remaining grounded objects, state which one is {extreme} Object "
    "{anchor}, using exactly one sentence.",
)
EGO_RANK_TEMPLATES = (
    "Which grounded object is {rank_phrase} the ego vehicle? Answer in exactly "
    "one sentence.",
    "Rank the grounded objects by distance to the ego vehicle and identify the "
    "one that is {rank_phrase} it, using exactly one sentence.",
    "Using the grounding tokens, state which object is {rank_phrase} the ego "
    "vehicle in exactly one sentence.",
)


def relation_atoms(first, second, deadband=1.0):
    dx = float(first[0]) - float(second[0])
    dy = float(first[1]) - float(second[1])
    atoms = []
    if abs(dy) >= deadband:
        atoms.append("left" if dy > 0 else "right")
    if abs(dx) >= deadband:
        atoms.append("front" if dx > 0 else "behind")
    if not atoms:
        atoms.append("front" if dx >= 0 else "behind")
    return atoms


def relation_phrase(subject, atoms, target):
    words = {"left": "to the left of", "right": "to the right of",
             "front": "in front of", "behind": "behind"}
    body = words[atoms[0]] if len(atoms) == 1 else (
        words[atoms[0]] + " and " + words[atoms[1]])
    target_text = "the ego vehicle" if target == "EGO" else f"Object {target}"
    return f"Object {subject} is {body} {target_text}"


def triples(subject, atoms, target):
    return [[subject, atom, target] for atom in atoms]


def refs(record, order):
    return [{"role": ROLES[new_index],
             "object_id": record["object_ids"][old_index]}
            for new_index, old_index in enumerate(order)]


def role_preamble(object_refs):
    roles = ", ".join(x["role"] for x in object_refs)
    return f"Grounding tokens are provided for Objects {roles}."


def stable_order(record, count):
    """Assign roles independently of geometry and annotation order."""
    order = list(range(count))
    seed = int(hashlib.sha1(record["scene_id"].encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(order)
    return order


def qa_id(scene_id, scenario, suffix):
    return hashlib.sha1(f"{scene_id}|{scenario}|{suffix}".encode()).hexdigest()[:16]


def template_count(target_count):
    """Oversample scarce N=3 scenes with genuine question paraphrases."""
    if target_count == 2:
        return 1
    if target_count == 3:
        return 3
    return 2


def add_row(rows, private, record, split, category, scenario, suffix, order,
            question_tail, answer, required_triples, answer_key=None,
            planning_key=None, swap_pair_id=None):
    if category not in QUESTION_CATEGORIES:
        raise ValueError(category)
    object_refs = refs(record, order)
    ident = qa_id(record["scene_id"], scenario, suffix)
    if ident in private:
        raise ValueError(f"duplicate qa_id {ident}")
    row = {
        "qa_id": ident, "qa_split": split, "question_category": category,
        "scenario": scenario, "scene_id": record["scene_id"],
        "sequence": record["segment_name"], "target_count": len(order),
        "question": role_preamble(object_refs) + " " + question_tail,
        "answer": answer, "object_refs": object_refs,
        "dataset_version": DATASET_VERSION,
        "input_policy": "implicit_grounding_tokens_only",
        "answer_format": "exactly_one_complete_sentence",
        "task_scope": category,
    }
    rows.append(row)
    expected_keywords = sorted(
        relation for relation in INVERSE
        if re.search(rf"\b{relation}\b", answer.lower()))
    private[ident] = {
        "qa_id": ident, "question_category": category,
        "canonical_answer": answer, "required_triples": required_triples,
        "expected_relation_keywords": expected_keywords,
        "answer_key": answer_key,
        "gt_geometry": [record[f"bbox3d_obj_{i + 1}"] for i in order],
        "grounding_inputs": [{
            "role": ROLES[new_index],
            "object_id": record["object_ids"][old_index],
            "category": record["object_classes"][old_index],
            "description": record["object_descriptions"][old_index],
        } for new_index, old_index in enumerate(order)],
        "planning_key": planning_key, "swap_pair_id": swap_pair_id,
    }


def distance_2d(first, second=(0.0, 0.0)):
    return math.hypot(float(first[0]) - float(second[0]),
                      float(first[1]) - float(second[1]))


def rank_label(rank, count):
    if rank == 0:
        return "closest", "to"
    if rank == count - 1:
        return "farthest", "from"
    ordinal = {1: "second", 2: "third", 3: "fourth", 4: "fifth"}[rank]
    return f"{ordinal} closest", "to"


def local_plan(boxes):
    """Evaluate fixed local paths toward x=20 m against all boxes ahead."""
    ahead = [(i, b) for i, b in enumerate(boxes) if 1.0 < float(b[0]) < 22.0]
    if not ahead:
        return None
    blocker_index, blocker = min(ahead, key=lambda item: distance_2d(item[1]))
    candidate_specs = [("straight", 0.0), ("left", 3.0), ("right", -3.0)]
    candidates = []
    for steer, offset in candidate_specs:
        clearance = float("inf")
        for _, box in ahead:
            x, y = float(box[0]), float(box[1])
            path_y = offset * min(max(x / 10.0, 0.0), 1.0)
            half_width = max(float(box[4]) / 2.0, 0.3)
            clearance = min(clearance, abs(y - path_y) - half_width - 1.2)
        candidates.append({"maneuver": steer, "offset_m": offset,
                           "clearance_m": clearance,
                           "safe": clearance >= -0.5})
    priority = {"straight": 1, "left": 0, "right": 0}
    best = max(candidates, key=lambda x: (x["clearance_m"],
                                          priority[x["maneuver"]]))
    if best["clearance_m"] < -0.5:
        steer, pass_side = "stop", None
    else:
        steer = best["maneuver"]
        pass_side = "left" if best["offset_m"] > float(blocker[1]) else "right"
    return {"steer": steer, "pass_side": pass_side,
            "blocker_index": blocker_index, "lookahead_m": 20.0,
            "candidate_offsets_m": [0.0, 3.0, -3.0],
            "candidates": candidates}


def add_direction_questions(rows, private, record, split, order, role_boxes):
    count = len(order)
    variants = template_count(count)
    swap_pair_id = qa_id(record["scene_id"], "pairwise_direction", "ab_pair")
    for source in range(count):
        for target in range(count):
            if source == target:
                continue
            atoms = relation_atoms(role_boxes[source], role_boxes[target])
            subject, reference = ROLES[source], ROLES[target]
            answer = relation_phrase(subject, atoms, reference) + "."
            for variant, template in enumerate(PAIR_DIRECTION_TEMPLATES[:variants]):
                pair_id = (swap_pair_id if source == 0 and target == 1 and
                           variant == 0 else None)
                add_row(
                    rows, private, record, split, "direction",
                    "pairwise_direction", f"{subject}_{reference}_v{variant}",
                    order, template.format(subject=subject, reference=reference),
                    answer, triples(subject, atoms, reference),
                    answer_key={"type": "directed_relation", "subject": subject,
                                "reference": reference, "relations": atoms},
                    swap_pair_id=pair_id)

    base_atoms = relation_atoms(role_boxes[0], role_boxes[1])
    swapped_order = [order[1], order[0]] + order[2:]
    swapped_atoms = [INVERSE[x] for x in base_atoms]
    add_row(
        rows, private, record, split, "direction", "pairwise_direction",
        "AB_physical_swap_control", swapped_order,
        PAIR_DIRECTION_TEMPLATES[0].format(subject="A", reference="B"),
        relation_phrase("A", swapped_atoms, "B") + ".",
        triples("A", swapped_atoms, "B"),
        answer_key={"type": "directed_relation", "subject": "A",
                    "reference": "B", "relations": swapped_atoms},
        swap_pair_id=swap_pair_id)

    if count >= 3:
        multi_variants = min(variants, len(MULTI_DIRECTION_TEMPLATES))
        for source in range(count):
            subject = ROLES[source]
            clauses, required = [], []
            for target in range(count):
                if source == target:
                    continue
                atoms = relation_atoms(role_boxes[source], role_boxes[target])
                clauses.append(relation_phrase(subject, atoms, ROLES[target]))
                required.extend(triples(subject, atoms, ROLES[target]))
            answer = "; ".join(clauses) + "."
            for variant, template in enumerate(
                    MULTI_DIRECTION_TEMPLATES[:multi_variants]):
                add_row(
                    rows, private, record, split, "direction",
                    "multi_reference_direction", f"{subject}_all_v{variant}",
                    order, template.format(subject=subject), answer, required,
                    answer_key={"type": "multi_directed_relation",
                                "subject": subject})


def add_distance_questions(rows, private, record, split, order, role_boxes):
    count = len(order)
    variants = template_count(count)
    for subject_index, box in enumerate(role_boxes):
        subject = ROLES[subject_index]
        within = distance_2d(box) <= DISTANCE_THRESHOLD_M
        answer = (f"Object {subject} is within 10 meters of the ego vehicle."
                  if within else
                  f"Object {subject} is farther than 10 meters from the ego vehicle.")
        for variant, template in enumerate(DISTANCE_BAND_TEMPLATES[:variants]):
            add_row(
                rows, private, record, split, "distance", "ego_distance_band",
                f"{subject}_v{variant}", order,
                template.format(subject=subject), answer, [],
                answer_key={"type": "ego_distance_band", "subject": subject,
                            "within_10m": within})

    ranked = sorted(range(count), key=lambda index: distance_2d(role_boxes[index]))
    for rank, object_index in enumerate(ranked):
        label, preposition = rank_label(rank, count)
        role = ROLES[object_index]
        answer = (f"Object {role} is the {label} grounded object "
                  f"{preposition} the ego vehicle.")
        rank_phrase = f"the {label} to" if preposition == "to" else "farthest from"
        for variant, template in enumerate(EGO_RANK_TEMPLATES[:variants]):
            add_row(
                rows, private, record, split, "distance", "ego_distance_rank",
                f"rank{rank}_{role}_v{variant}", order,
                template.format(rank_phrase=rank_phrase), answer, [],
                answer_key={"type": "ego_distance_rank", "rank": rank,
                            "role": role, "label": label})

    if count >= 3:
        for anchor_index in range(count):
            anchor = ROLES[anchor_index]
            candidates = [index for index in range(count) if index != anchor_index]
            candidates.sort(key=lambda index: distance_2d(
                role_boxes[index], role_boxes[anchor_index]))
            for extreme, object_index in (("closest to", candidates[0]),
                                          ("farthest from", candidates[-1])):
                role = ROLES[object_index]
                adjective = "closest" if extreme.startswith("closest") else "farthest"
                preposition = "to" if adjective == "closest" else "from"
                answer = (f"Object {role} is the {adjective} grounded object "
                          f"{preposition} Object {anchor}.")
                for variant, template in enumerate(
                        ANCHOR_DISTANCE_TEMPLATES[:variants]):
                    add_row(
                        rows, private, record, split, "distance",
                        "anchor_distance_extreme",
                        f"{anchor}_{adjective}_{role}_v{variant}", order,
                        template.format(extreme=extreme, anchor=anchor), answer, [],
                        answer_key={"type": "anchor_distance_extreme",
                                    "anchor": anchor, "extreme": adjective,
                                    "role": role})


def add_obstacle_questions(rows, private, record, split, order, role_boxes):
    plan = local_plan(role_boxes)
    if plan is None:
        return
    blocker = plan["blocker_index"]
    blocker_role = ROLES[blocker]
    blocker_atoms = relation_atoms(role_boxes[blocker], [0, 0, 0, 0, 0, 0, 0])
    blocker_phrase = relation_phrase(blocker_role, blocker_atoms, "EGO")
    if plan["steer"] == "stop":
        action = f"the ego vehicle should stop before Object {blocker_role}"
    else:
        action = (f"the ego vehicle should steer {plan['steer']} and continue "
                  f"forward while passing to the {plan['pass_side']} of Object "
                  f"{blocker_role}")
    answer = blocker_phrase + ", so " + action + "."
    planning_key = {
        "kind": "global_plan", "steer": plan["steer"],
        "pass_side": plan["pass_side"], "blocker_role": blocker_role,
        "action": "stop" if plan["steer"] == "stop" else "continue_forward",
        "lookahead_m": plan["lookahead_m"],
    }
    add_row(
        rows, private, record, split, "obstacle_avoidance", "local_path_plan",
        "global", order,
        "Identify the nearest grounded obstacle ahead and give the safest local "
        "maneuver toward the fixed forward goal; answer in exactly one sentence.",
        answer, triples(blocker_role, blocker_atoms, "EGO"),
        answer_key={"type": "local_path_plan", "blocker_role": blocker_role,
                    "steer": plan["steer"], "pass_side": plan["pass_side"]},
        planning_key=planning_key)

    maneuver_name = {"left": "left-steering", "straight": "straight",
                     "right": "right-steering"}
    for candidate in plan["candidates"]:
        maneuver, safe = candidate["maneuver"], candidate["safe"]
        label = maneuver_name[maneuver]
        answer = (f"The {label} candidate maneuver is "
                  f"{'safe' if safe else 'unsafe'} for the ego vehicle.")
        add_row(
            rows, private, record, split, "obstacle_avoidance",
            "candidate_maneuver_safety", maneuver, order,
            f"Is the fixed {label} candidate maneuver collision-safe when all "
            "grounded objects are considered? Answer in exactly one sentence.",
            answer, [],
            answer_key={"type": "candidate_maneuver_safety",
                        "maneuver": maneuver, "safe": safe},
            planning_key={"kind": "candidate_maneuver_safety",
                          "maneuver": maneuver, "safe": safe})


def build_record(record, split, rows, private):
    count = int(record["target_count"])
    if count < 2:
        return
    boxes = [record[f"bbox3d_obj_{i + 1}"] for i in range(count)]
    order = stable_order(record, count)
    role_boxes = [boxes[index] for index in order]
    add_direction_questions(rows, private, record, split, order, role_boxes)
    add_distance_questions(rows, private, record, split, order, role_boxes)
    add_obstacle_questions(rows, private, record, split, order, role_boxes)


def audit(rows, private):
    """Fail generation if public content exposes private grounding labels."""
    source_caption_hits = 0
    forbidden_public_fields = 0
    coordinate_hits = 0
    direct_relation_label_hits = 0
    swap_groups = defaultdict(list)
    subject_distribution = Counter()
    for row in rows:
        gt = private[row["qa_id"]]
        assert row["question_category"] in QUESTION_CATEGORIES
        assert row["target_count"] == len(row["object_refs"])
        assert gt["question_category"] == row["question_category"]
        forbidden_public_fields += sum(
            any(key in ref for key in ("description", "category", "bbox", "center"))
            for ref in row["object_refs"])
        question_norm = row["question"].lower()
        source_caption_hits += sum(
            item["description"].strip().lower() in question_norm
            for item in gt["grounding_inputs"] if item["description"].strip())
        coordinate_hits += bool(re.search(
            r"(?:x|y|z)\s*[:=]\s*-?\d|\[\s*-?\d+(?:\.\d+)?\s*,",
            row["question"], re.I))
        if row["question_category"] != "obstacle_avoidance":
            direct_relation_label_hits += bool(re.search(
                r"\b(?:left|right|front|behind)\b", row["question"], re.I))
        key = gt.get("answer_key") or {}
        if key.get("subject"):
            subject_distribution[key["subject"]] += 1
        if gt["swap_pair_id"]:
            swap_groups[gt["swap_pair_id"]].append(row)
    assert source_caption_hits == 0
    assert forbidden_public_fields == 0
    assert coordinate_hits == 0
    assert direct_relation_label_hits == 0
    assert len({row["qa_id"] for row in rows}) == len(rows)
    for pair_id, pair in swap_groups.items():
        assert len(pair) == 2, (pair_id, len(pair))
        base, swapped = pair
        base_ids = [item["object_id"] for item in base["object_refs"]]
        swapped_ids = [item["object_id"] for item in swapped["object_refs"]]
        assert swapped_ids == [base_ids[1], base_ids[0]] + base_ids[2:], pair_id
    return {
        "public_source_caption_hits": source_caption_hits,
        "public_forbidden_object_fields": forbidden_public_fields,
        "public_coordinate_pattern_hits": coordinate_hits,
        "public_answer_relation_label_hits_outside_planning": direct_relation_label_hits,
        "subject_role_distribution": dict(sorted(subject_distribution.items())),
        "ab_swap_pairs_checked": len(swap_groups),
        "model_text_input_fields": ["question"],
        "model_non_text_input_fields": ["ordered_288d_grounding_tokens"],
    }


def nested_counts(rows, first, second):
    values = defaultdict(Counter)
    for row in rows:
        values[str(row[first])][str(row[second])] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(values.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Directory containing waymo_scene_multi_*_info.pkl")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, private = [], {}
    for split in ("train", "val"):
        with (args.annotations / f"waymo_scene_multi_{split}_info.pkl").open("rb") as f:
            records = pickle.load(f)
        for record in records:
            build_record(record, split, rows, private)
    audit_result = audit(rows, private)
    scene_targets = {(row["qa_split"], row["scene_id"]): row["target_count"]
                     for row in rows}
    scene_target_counts = defaultdict(Counter)
    for (split, _), target_count in scene_targets.items():
        scene_target_counts[split][str(target_count)] += 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "qa.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "private_gt.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(private[row["qa_id"]], ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "splits": dict(sorted(Counter(r["qa_split"] for r in rows).items())),
        "question_categories": dict(sorted(Counter(
            r["question_category"] for r in rows).items())),
        "target_counts": dict(sorted(Counter(r["target_count"] for r in rows).items())),
        "split_target_counts": nested_counts(rows, "qa_split", "target_count"),
        "unique_scenes": len(scene_targets),
        "scene_target_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(scene_target_counts.items())
        },
        "category_target_counts": nested_counts(
            rows, "question_category", "target_count"),
        "scenarios": dict(sorted(Counter(r["scenario"] for r in rows).items())),
        "all_answers_one_sentence": all(
            r["answer"].count(".") == 1 and r["answer"].endswith(".")
            for r in rows),
        "distance_threshold_m": DISTANCE_THRESHOLD_M,
        "planning_scope": "deterministic local obstacle avoidance to a fixed forward goal",
        "full_route_planning": False,
        "dataset_version": DATASET_VERSION,
        "input_policy": "implicit_grounding_tokens_only",
        "leakage_audit": audit_result,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
