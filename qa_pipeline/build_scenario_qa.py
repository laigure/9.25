"""Build sequence-disjoint spatial QA from neutral 3EED object tokens.

Ground-truth boxes are read here only to create answers. The public QA JSONL
contains descriptions, token references and answers, never geometry or a
precomputed spatial relation in its question/feature inputs.
"""

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import random
import re


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def center(row):
    return row["gt_box"][:3]


def direction(dx, dy):
    # Waymo vehicle coordinates: +x forward, +y left. Omit boundary cases.
    lon = "front" if dx >= 3.0 else "behind" if dx <= -3.0 else None
    lat = "left" if dy >= 1.5 else "right" if dy <= -1.5 else None
    if lon and lat:
        return lat + "_" + lon
    if lon and abs(dy) <= 0.75:
        return lon
    if lat and abs(dx) <= 1.5:
        return lat
    return None


def sentence(label, anchor):
    parts = label.split("_")
    terms = {"left": "to the left of", "right": "to the right of",
             "front": "in front of", "behind": "behind"}
    if len(parts) == 2:
        side, long = parts
        return "to the {} and {} {}".format(side, terms[long], anchor)
    return "{} {}".format(terms[parts[0]], anchor)


def ref(row, role):
    return {"role": role, "description": row["neutral_query"],
            "object_id": row["object_id"], "token_index": row["token_index"]}


def make_row(scene, split, scenario, objects, question, answer, answer_key,
             label, serial):
    key = "{}|{}|{}|{}".format(scene, scenario,
                              ",".join(x["object_id"] for x in objects), serial)
    qa_id = hashlib.sha1(key.encode()).hexdigest()[:16]
    return ({"qa_id": qa_id, "qa_split": split, "scenario": scenario,
             "scene_id": scene, "sequence": objects[0]["sequence"],
             "question": question, "answer": answer,
             "token_path": objects[0]["token_path"],
             "object_refs": [ref(x, chr(65 + i)) for i, x in enumerate(objects)]},
            {"qa_id": qa_id, "answer_key": answer_key,
             "gt_geometry": [x["gt_box"] for x in objects], "rule": label})


def split_sequences(rows, seed=42):
    train = {x["sequence"] for x in rows if x["split"] == "train"}
    held = sorted({x["sequence"] for x in rows if x["split"] == "val"})
    assert not train.intersection(held)
    random.Random(seed).shuffle(held)
    midpoint = len(held) // 2
    mapping = {s: "train" for s in train}
    mapping.update({s: "val" for s in held[:midpoint]})
    mapping.update({s: "test" for s in held[midpoint:]})
    return mapping


def build(rows, seed=42):
    mapping = split_sequences(rows, seed)
    grouped = collections.defaultdict(list)
    for row in rows:
        assert re.fullmatch(r"[a-z0-9_./#-]+", row["object_id"], re.I)
        grouped[row["scene_id"]].append(row)
    public, private = [], []
    counts = collections.Counter()
    for scene in sorted(grouped):
        objects = sorted(grouped[scene], key=lambda r: r["object_id"])
        split = mapping[objects[0]["sequence"]]
        # An object is usable only if its grounded phrase differs from every
        # other phrase in the same question. No prediction-IoU filtering.
        for a in objects:
            x, y, _ = center(a)
            label = direction(x, y)
            if not label:
                continue
            desc = a["neutral_query"].strip()
            question = "Object A: {}. Where is Object A relative to the ego vehicle?".format(desc)
            answer = "Object A is {}.".format(sentence(label, "the ego vehicle"))
            item, gt = make_row(scene, split, "ego_location", [a], question,
                                answer, label, label, 0)
            public.append(item); private.append(gt); counts[(split, "ego_location")] += 1
        for a, b in itertools.combinations(objects, 2):
            if a["neutral_query"].strip().casefold() == b["neutral_query"].strip().casefold():
                continue
            dx = center(a)[0] - center(b)[0]
            dy = center(a)[1] - center(b)[1]
            label = direction(dx, dy)
            if not label:
                continue
            question = ("Object A: {}. Object B: {}. Where is Object A relative to Object B?"
                        .format(a["neutral_query"], b["neutral_query"]))
            answer = "Object A is {}.".format(sentence(label, "Object B"))
            item, gt = make_row(scene, split, "relative_location", [a, b],
                                question, answer, label, label, 0)
            public.append(item); private.append(gt); counts[(split, "relative_location")] += 1
        for trio in itertools.combinations(objects, 3):
            a, b, c = trio
            # Annotation order often follows depth; reverse candidates in a
            # deterministic half of scenes to avoid a spurious A/B prior.
            if int(hashlib.sha1((scene + a["object_id"] + b["object_id"]).encode()).hexdigest(), 16) % 2:
                a, b = b, a
            phrases = [x["neutral_query"].strip().casefold() for x in trio]
            if len(set(phrases)) < 3:
                continue
            da = math.dist(center(a)[:2], center(c)[:2])
            db = math.dist(center(b)[:2], center(c)[:2])
            if abs(da - db) < max(3.0, 0.15 * min(da, db)):
                continue
            winner = "A" if da < db else "B"
            question = ("Object A: {}. Object B: {}. Object C: {}. "
                        "Which object is closer to Object C, Object A or Object B?"
                        .format(a["neutral_query"], b["neutral_query"], c["neutral_query"]))
            answer = "Object {} is closer to Object C.".format(winner)
            item, gt = make_row(scene, split, "closer_of_two", [a, b, c],
                                question, answer, winner, "distance_gap", 0)
            public.append(item); private.append(gt); counts[(split, "closer_of_two")] += 1
    ids = [x["qa_id"] for x in public]
    assert len(ids) == len(set(ids))
    sets = {name: {x["sequence"] for x in public if x["qa_split"] == name}
            for name in ("train", "val", "test")}
    assert not any(sets[a] & sets[b] for a, b in itertools.combinations(sets, 2))
    assert len(public) == len(private)
    # Key audit: features are indexed only by token refs. Coordinates live in
    # private rows; never in public sample fields or textual questions.
    for row in public:
        assert "gt_box" not in row and "center" not in row and "answer_key" not in row
        assert len(row["object_refs"]) == (1 if row["scenario"] == "ego_location" else
                                         2 if row["scenario"] == "relative_location" else 3)
    return public, private, mapping, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(args.bank, encoding="utf-8")]
    os.makedirs(args.out_dir, exist_ok=True)
    public, private, mapping, counts = build(rows, args.seed)
    write_jsonl(os.path.join(args.out_dir, "qa.jsonl"), public)
    write_jsonl(os.path.join(args.out_dir, "private_gt.jsonl"), private)
    summary = {"total": len(public), "by_split_scenario":
               {"{}:{}".format(*key): value for key, value in sorted(counts.items())},
               "sequences": {s: sum(v == s for v in mapping.values()) for s in ("train", "val", "test")},
               "coordinate_convention": "Waymo ego +x forward, +y left",
               "input_policy": "question plus implicit 288D Grounding tokens only; GT geometry offline labels",
               "planning": "not generated: no future ego trajectory or drivable-area labels"}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
