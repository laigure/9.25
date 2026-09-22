"""Expand v2 with distinct spatial tasks while keeping input token-only.

The extra QA rows reuse existing objects. Summary reports both row count and
independent scene/object counts so repetitions are not misrepresented.
"""

import argparse
import collections
import itertools
import json
import math
import os
import random

from build_scenario_qa import build, center, direction, make_row, sentence, write_jsonl


def distinct(*objects):
    return len({x["neutral_query"].strip().casefold() for x in objects}) == len(objects)


def balanced_order(a, b, rng):
    return (b, a) if rng.randrange(2) else (a, b)


def far_enough(d1, d2):
    return abs(d1 - d2) >= max(3.0, 0.15 * min(d1, d2))


def add(public, private, counts, scene, split, scenario, objects, question,
        answer, key, rule, serial=0):
    row, gt = make_row(scene, split, scenario, objects, question, answer,
                      key, rule, serial)
    public.append(row)
    private.append(gt)
    counts[(split, scenario)] += 1


def expand(rows, seed=42):
    public, private, mapping, counts = build(rows, seed)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    for scene in sorted(grouped):
        objects = sorted(grouped[scene], key=lambda r: r["object_id"])
        split = mapping[objects[0]["sequence"]]
        rng = random.Random(scene + "|v3|" + str(seed))
        for a in objects:
            x, y, _ = center(a)
            desc = a["neutral_query"]
            if abs(y) >= 2.0:
                side = "left" if y > 0 else "right"
                add(public, private, counts, scene, split, "ego_side", [a],
                    "Object A: {}. On which side of the ego vehicle is Object A?".format(desc),
                    "Object A is on the {} side of the ego vehicle.".format(side),
                    side, "abs(y)>=2m")
            distance = math.hypot(x, y)
            if distance <= 15.0 or distance >= 30.0:
                grade = "near" if distance <= 15.0 else "far"
                add(public, private, counts, scene, split, "ego_near_far", [a],
                    "Object A: {}. Is Object A near or far from the ego vehicle?".format(desc),
                    "Object A is {} the ego vehicle.".format(
                        "near" if grade == "near" else "far from"),
                    grade, "distance<=15m or >=30m")

        for first, second in itertools.combinations(objects, 2):
            if not distinct(first, second):
                continue
            a, b = first, second
            # Paired reversal is a distinct direction query, but uses the
            # same scene and must never be treated as independent evidence.
            reverse = direction(center(b)[0] - center(a)[0],
                                center(b)[1] - center(a)[1])
            if reverse:
                add(public, private, counts, scene, split, "relative_location",
                    [b, a],
                    "Object A: {}. Object B: {}. Where is Object A relative to Object B?"
                    .format(b["neutral_query"], a["neutral_query"]),
                    "Object A is {}.".format(sentence(reverse, "Object B")),
                    reverse, "reverse_pair", 1)

            a, b = balanced_order(first, second, rng)
            ya, yb = center(a)[1], center(b)[1]
            if abs(ya - yb) >= 2.0:
                winner = "A" if ya > yb else "B"
                add(public, private, counts, scene, split, "which_is_left", [a, b],
                    "Object A: {}. Object B: {}. Which object is farther to the left, Object A or Object B?"
                    .format(a["neutral_query"], b["neutral_query"]),
                    "Object {} is farther to the left.".format(winner),
                    winner, "abs(delta_y)>=2m")
            da, db = math.hypot(*center(a)[:2]), math.hypot(*center(b)[:2])
            if far_enough(da, db):
                winner = "A" if da < db else "B"
                add(public, private, counts, scene, split, "closer_to_ego", [a, b],
                    "Object A: {}. Object B: {}. Which object is closer to the ego vehicle, Object A or Object B?"
                    .format(a["neutral_query"], b["neutral_query"]),
                    "Object {} is closer to the ego vehicle.".format(winner),
                    winner, "distance_gap")

        for triple in itertools.combinations(objects, 3):
            if not distinct(*triple):
                continue
            # Each of the three objects may serve as the reference. Preserve
            # the v2 instance; duplicate QA ids are removed below.
            for anchor in triple:
                candidates = [x for x in triple if x is not anchor]
                a, b = balanced_order(*candidates, rng)
                da, db = math.dist(center(a)[:2], center(anchor)[:2]), \
                         math.dist(center(b)[:2], center(anchor)[:2])
                if not far_enough(da, db):
                    continue
                winner = "A" if da < db else "B"
                add(public, private, counts, scene, split, "closer_of_two",
                    [a, b, anchor],
                    "Object A: {}. Object B: {}. Object C: {}. Which object is closer to Object C, Object A or Object B?"
                    .format(a["neutral_query"], b["neutral_query"], anchor["neutral_query"]),
                    "Object {} is closer to Object C.".format(winner),
                    winner, "alternate_anchor")
            ordered = list(triple)
            rng.shuffle(ordered)
            distance = [math.hypot(*center(x)[:2]) for x in ordered]
            ranking = sorted(range(3), key=lambda i: distance[i])
            if far_enough(distance[ranking[0]], distance[ranking[1]]):
                winner = "ABC"[ranking[0]]
                add(public, private, counts, scene, split, "closest_of_three_to_ego",
                    ordered,
                    "Object A: {}. Object B: {}. Object C: {}. Which object is closest to the ego vehicle?"
                    .format(*(x["neutral_query"] for x in ordered)),
                    "Object {} is closest to the ego vehicle.".format(winner),
                    winner, "nearest_runner_up_gap")

    # A v2 triple may coincide with the expanded anchor/candidate ordering.
    # Deduplicate by stable qa_id rather than counting repeated identical rows.
    unique = {}
    for row, gt in zip(public, private):
        if row["qa_id"] in unique:
            prior, prior_gt = unique[row["qa_id"]]
            assert prior["question"] == row["question"] and prior_gt["answer_key"] == gt["answer_key"]
        else:
            unique[row["qa_id"]] = (row, gt)
    public = [pair[0] for pair in unique.values()]
    private = [pair[1] for pair in unique.values()]
    counts = collections.Counter((r["qa_split"], r["scenario"]) for r in public)
    assert len(public) == len(private) == len(unique)
    assert not ({r["sequence"] for r in public if r["qa_split"] == "train"} &
                {r["sequence"] for r in public if r["qa_split"] == "val"})
    for row in public:
        assert not any(k in row for k in ("gt_box", "gt_center", "pred_center",
                                       "pred_size", "distance", "answer_key"))
    return public, private, mapping, counts, grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(args.bank, encoding="utf-8")]
    public, private, mapping, counts, grouped = expand(rows, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "qa.jsonl"), public)
    write_jsonl(os.path.join(args.out_dir, "private_gt.jsonl"), private)
    summary = {"total_qa_rows": len(public),
               "source_object_annotations": len(rows),
               "source_scenes": len(grouped),
               "qa_distinct_objects": len({o["object_id"] for r in public for o in r["object_refs"]}),
               "qa_distinct_scenes": len({r["scene_id"] for r in public}),
               "multi_object_scenes": sum(len(v) >= 2 for v in grouped.values()),
               "three_object_scenes": sum(len(v) >= 3 for v in grouped.values()),
               "by_split_scenario": {"{}:{}".format(*k): v for k, v in sorted(counts.items())},
               "sequences": {s: sum(v == s for v in mapping.values())
                             for s in ("train", "val", "test")},
               "input_policy": "question plus implicit 288D Grounding tokens only",
               "label_policy": "GT geometry offline only; no predicted-IoU filtering",
               "planning": "no future trajectory labels in current 3EED conversion"}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
