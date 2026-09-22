"""Generate hard, one-sentence scene QA from variable-target Grounding records.

Public QA rows contain descriptions, object IDs and questions.  GT geometry,
relations and deterministic local-planning labels are written only to the
private file.  Planning is explicitly a local obstacle-avoidance task with a
fixed forward goal; it is not an HD-map route-planning label.
"""

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path


ROLES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
INVERSE = {"left": "right", "right": "left", "front": "behind",
           "behind": "front"}


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
    words = {
        "left": "to the left of", "right": "to the right of",
        "front": "in front of", "behind": "behind",
    }
    if len(atoms) == 1:
        body = words[atoms[0]]
    else:
        body = words[atoms[0]] + " and " + words[atoms[1]]
    return f"Object {subject} is {body} Object {target}"


def refs(record, order):
    return [{
        "role": ROLES[new_index],
        "description": record["object_descriptions"][old_index],
        "category": record["object_classes"][old_index],
        "object_id": record["object_ids"][old_index],
    } for new_index, old_index in enumerate(order)]


def preamble(object_refs):
    return " ".join(f"Object {x['role']}: {x['description']}." for x in object_refs)


def qa_id(scene_id, scenario, suffix):
    return hashlib.sha1(f"{scene_id}|{scenario}|{suffix}".encode()).hexdigest()[:16]


def triples(subject, atoms, target):
    return [[subject, atom, target] for atom in atoms]


def add_row(rows, private, record, split, scenario, suffix, order,
            question_tail, answer, required_triples, planning_key=None,
            swap_pair_id=None):
    object_refs = refs(record, order)
    ident = qa_id(record["scene_id"], scenario, suffix)
    row = {
        "qa_id": ident,
        "qa_split": split,
        "scenario": scenario,
        "scene_id": record["scene_id"],
        "sequence": record["segment_name"],
        "question": preamble(object_refs) + " " + question_tail,
        "answer": answer,
        "object_refs": object_refs,
        "answer_format": "exactly_one_complete_sentence",
        "task_scope": ("local_obstacle_avoidance" if planning_key else
                       "multi_object_spatial_reasoning"),
    }
    rows.append(row)
    expected_keywords = sorted({relation for _, relation, _ in required_triples} |
                               ({planning_key["steer"]} if planning_key and
                                planning_key["steer"] in ("left", "right") else set()) |
                               ({planning_key["pass_side"]} if planning_key and
                                planning_key.get("pass_side") else set()))
    private[ident] = {
        "qa_id": ident,
        "canonical_answer": answer,
        "required_triples": required_triples,
        "expected_relation_keywords": expected_keywords,
        "gt_geometry": [record[f"bbox3d_obj_{i + 1}"] for i in order],
        "planning_key": planning_key,
        "swap_pair_id": swap_pair_id,
    }


def local_plan(boxes):
    """Choose a deterministic ego-centric corridor toward x=20 m.

    front=+x and left=+y.  Candidate paths end at lateral offsets +3/0/-3 m.
    The score is the minimum lateral clearance from annotated boxes ahead.
    """
    ahead = [(i, b) for i, b in enumerate(boxes) if 1.0 < float(b[0]) < 22.0]
    if not ahead:
        return None
    blocker_index, blocker = min(ahead, key=lambda item: math.hypot(
        float(item[1][0]), float(item[1][1])))
    candidates = [("straight", 0.0), ("left", 3.0), ("right", -3.0)]
    scores = []
    for steer, offset in candidates:
        clearance = float("inf")
        for _, box in ahead:
            x, y = float(box[0]), float(box[1])
            path_y = offset * min(max(x / 10.0, 0.0), 1.0)
            half_width = max(float(box[4]) / 2.0, 0.3)
            clearance = min(clearance, abs(y - path_y) - half_width - 1.2)
        scores.append((clearance, 1 if steer == "straight" else 0, steer, offset))
    clearance, _, steer, offset = max(scores)
    if clearance < -0.5:
        steer = "stop"
        pass_side = None
    else:
        pass_side = "left" if offset > float(blocker[1]) else "right"
    return {"steer": steer, "pass_side": pass_side,
            "blocker_index": blocker_index,
            "lookahead_m": 20.0, "candidate_offsets_m": [0.0, 3.0, -3.0]}


def build_record(record, split, rows, private):
    count = int(record["target_count"])
    boxes = [record[f"bbox3d_obj_{i + 1}"] for i in range(count)]
    if count < 2:
        return

    # Relation amid all scene distractors, plus a physically equivalent A/B swap.
    order = list(range(count))
    atoms = relation_atoms(boxes[0], boxes[1])
    pair_id = qa_id(record["scene_id"], "distractor_pair_relation", "pair")
    answer = relation_phrase("A", atoms, "B") + "."
    add_row(rows, private, record, split, "distractor_pair_relation", "base",
            order, "Considering every listed object, where is Object A relative to "
            "Object B? Answer in exactly one sentence.", answer,
            triples("A", atoms, "B"), swap_pair_id=pair_id)

    swapped_order = [1, 0] + order[2:]
    swapped_atoms = [INVERSE[x] for x in atoms]
    swapped_answer = relation_phrase("A", swapped_atoms, "B") + "."
    add_row(rows, private, record, split, "distractor_pair_relation", "swap",
            swapped_order, "Considering every listed object, where is Object A "
            "relative to Object B? Answer in exactly one sentence.",
            swapped_answer, triples("A", swapped_atoms, "B"),
            swap_pair_id=pair_id)

    if count >= 3:
        atoms_ab = relation_atoms(boxes[0], boxes[1])
        atoms_ca = relation_atoms(boxes[2], boxes[0])
        answer = (relation_phrase("A", atoms_ab, "B") + "; " +
                  relation_phrase("C", atoms_ca, "A").lower() + ".")
        add_row(rows, private, record, split, "two_step_relation_chain", "abc",
                order, "State both where Object A is relative to Object B and where "
                "Object C is relative to Object A, using exactly one sentence.",
                answer, triples("A", atoms_ab, "B") + triples("C", atoms_ca, "A"))

        ranked = sorted(range(count), key=lambda i: math.hypot(
            float(boxes[i][0]), float(boxes[i][1])))
        closest_old, second_old = ranked[:2]
        ranked_order = [second_old, closest_old] + [i for i in order
                                                    if i not in ranked[:2]]
        rank_atoms = relation_atoms(boxes[second_old], boxes[closest_old])
        answer = ("Object A is the second closest object to the ego vehicle and is " +
                  relation_phrase("A", rank_atoms, "B").split(" is ", 1)[1] + ".")
        add_row(rows, private, record, split, "rank_then_relation", "rank2",
                ranked_order, "Which object is second closest to the ego vehicle, "
                "and where is it relative to the closest object? Answer in exactly "
                "one sentence.", answer, triples("A", rank_atoms, "B"))

    plan = local_plan(boxes)
    if plan is not None:
        blocker = plan.pop("blocker_index")
        plan_order = [blocker] + [i for i in order if i != blocker]
        blocker_atoms = relation_atoms(boxes[blocker], [0, 0, 0, 0, 0, 0, 0])
        blocker_phrase = relation_phrase("A", blocker_atoms, "EGO").replace(
            "Object EGO", "the ego vehicle")
        if plan["steer"] == "stop":
            action = "the ego vehicle should stop before Object A"
        else:
            action = (f"the ego vehicle should steer {plan['steer']} and continue "
                      f"forward while passing to the {plan['pass_side']} of Object A")
        answer = blocker_phrase + ", so " + action + "."
        planning_key = dict(plan, blocker_role="A", action=(
            "stop" if plan["steer"] == "stop" else "continue_forward"))
        add_row(rows, private, record, split, "local_path_planning", "plan",
                plan_order, "Identify the nearest grounded obstacle ahead and give "
                "the safest local maneuver toward the fixed forward goal; answer in "
                "exactly one sentence.", answer,
                triples("A", blocker_atoms, "EGO"), planning_key=planning_key)


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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "qa.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "private_gt.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(private[row["qa_id"]], ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "splits": {split: sum(r["qa_split"] == split for r in rows)
                   for split in ("train", "val")},
        "scenarios": {scenario: sum(r["scenario"] == scenario for r in rows)
                      for scenario in sorted({r["scenario"] for r in rows})},
        "all_answers_one_sentence": all(r["answer"].count(".") == 1
                                        and r["answer"].endswith(".") for r in rows),
        "planning_scope": "deterministic local obstacle avoidance to a fixed forward goal",
        "full_route_planning": False,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
