"""Strict all-conditions evaluator for scene-level spatial QA and planning."""

import argparse
import collections
import json
import re


RELATIONS = ("left", "right", "front", "behind")
INVERSE = {"left": "right", "right": "left", "front": "behind",
           "behind": "front"}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def read_predictions(path):
    with open(path, encoding="utf-8") as handle:
        try:
            value = json.load(handle)
            if isinstance(value, dict) and "rows" in value:
                return [{"qa_id": row["qa_id"],
                         "prediction": row.get("prediction", "")}
                        for row in value["rows"]]
        except json.JSONDecodeError:
            pass
    return read_jsonl(path)


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def relation_keywords(text):
    value = norm(text)
    return {word for word in RELATIONS if re.search(rf"\b{word}\b", value)}


def extract_triples(text):
    """Parse atomic relations from the required canonical one-sentence grammar."""
    triples = set()
    pattern = re.compile(
        r"object\s+([A-Z])\s+is\s+(.+?)\s+(?:object\s+([A-Z])|the\s+ego\s+vehicle)"
        r"(?=\s*[,;.]|\s+and\s+object|$)", re.IGNORECASE)
    for match in pattern.finditer(text):
        subject = match.group(1).upper()
        target = match.group(3).upper() if match.group(3) else "EGO"
        phrase = norm(match.group(2))
        for relation in RELATIONS:
            if re.search(rf"\b{relation}\b", phrase):
                triples.add((subject, relation, target))
    return triples


def one_sentence(text):
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(".") and len(
        re.findall(r"[.!?]", stripped)) == 1 and "\n" not in stripped


def planning_key(text, gt):
    if not gt:
        return None
    value = norm(text)
    if gt.get("kind") == "candidate_maneuver_safety":
        safe = bool(re.search(r"\bis safe\b", value))
        unsafe = bool(re.search(r"\bis unsafe\b", value))
        return safe and not unsafe if gt["safe"] else unsafe
    if gt["steer"] == "stop":
        return bool(re.search(r"\bshould stop\b", value))
    steer = bool(re.search(rf"\bsteer {gt['steer']}\b", value))
    forward = bool(re.search(r"\bcontinue forward\b", value))
    passing = bool(re.search(
        rf"\bpassing to the {gt['pass_side']} of object {gt['blocker_role'].lower()}\b",
        value))
    return steer and forward and passing


def evaluate(qa_rows, predictions, private_rows):
    private = {x["qa_id"]: x for x in private_rows}
    pred = {x["qa_id"]: x.get("prediction", "") for x in predictions}
    details = []
    for row in qa_rows:
        gt = private[row["qa_id"]]
        text = pred.get(row["qa_id"], "")
        expected_triples = {tuple(x) for x in gt["required_triples"]}
        parsed_triples = extract_triples(text)
        keyword_ok = relation_keywords(text) == set(gt["expected_relation_keywords"])
        triple_ok = parsed_triples == expected_triples
        sentence_ok = norm(text) == norm(gt["canonical_answer"])
        one_sentence_ok = one_sentence(text)
        planning_applicable = gt.get("planning_key") is not None
        plan_ok = planning_key(text, gt.get("planning_key"))
        if plan_ok is None:
            plan_ok = True
        details.append({
            "qa_id": row["qa_id"], "scenario": row["scenario"],
            "question_category": row.get("question_category", "legacy"),
            "target_count": row.get("target_count", len(row.get("object_refs", []))),
            "prediction": text, "keyword_ok": keyword_ok,
            # Candidate maneuvers can contain words such as "left-steering"
            # without expressing an object-object relation.  Count relation
            # keyword accuracy only when a spatial triple is required.
            "relation_keyword_applicable": bool(expected_triples),
            "triple_ok": triple_ok, "sentence_ok": sentence_ok,
            "triple_applicable": bool(expected_triples),
            "one_sentence_ok": one_sentence_ok, "planning_ok": plan_ok,
            "planning_applicable": planning_applicable,
            "parsed_triples": sorted(parsed_triples),
            "expected_triples": sorted(expected_triples),
            "swap_pair_id": gt.get("swap_pair_id"),
            "grounding_pass": gt.get("diagnostic_grounding_pass"),
        })

    swap_groups = collections.defaultdict(list)
    for item in details:
        if item["swap_pair_id"]:
            swap_groups[item["swap_pair_id"]].append(item)
    swap_ok_by_id = {}
    for pair_id, items in swap_groups.items():
        ok = len(items) == 2 and all(x["triple_ok"] for x in items)
        if ok:
            first, second = items
            # The physical objects were exchanged but relabelled A/B in the
            # swapped question, so the role names stay A/B and the relation
            # itself must invert.
            inverted = {(s, INVERSE[r], o)
                        for s, r, o in map(tuple, first["parsed_triples"])}
            ok = inverted == set(map(tuple, second["parsed_triples"]))
        swap_ok_by_id[pair_id] = ok

    for item in details:
        item["swap_ok"] = (swap_ok_by_id[item["swap_pair_id"]]
                           if item["swap_pair_id"] else True)
        item["language_all_ok"] = all(item[key] for key in (
            "keyword_ok", "triple_ok", "sentence_ok", "one_sentence_ok",
            "planning_ok", "swap_ok"))
        item["end_to_end_ok"] = (item["language_all_ok"] and
                                 item["grounding_pass"] is True)

    n = len(details)
    grounding_rows = [x for x in details if x["grounding_pass"] is True]
    known_grounding = [x for x in details if x["grounding_pass"] is not None]
    planning_rows = [x for x in details if x["planning_applicable"]]
    relation_rows = [x for x in details if x["relation_keyword_applicable"]]
    triple_rows = [x for x in details if x["triple_applicable"]]
    rate = lambda key, values=details: (sum(x[key] for x in values) / len(values)
                                        if values else None)
    def grouped(field):
        groups = collections.defaultdict(list)
        for item in details:
            groups[str(item[field])].append(item)
        return {
            key: {
                "n": len(values),
                "strict_language_all_pass": rate("language_all_ok", values),
                "complete_sentence_match_rate": rate("sentence_ok", values),
                "one_sentence_rate": rate("one_sentence_ok", values),
            }
            for key, values in sorted(groups.items())
        }
    result = {
        "n": n,
        "relation_keyword_n": len(relation_rows),
        "relation_keyword_accuracy": rate("keyword_ok", relation_rows),
        "strict_triple_n": len(triple_rows),
        "strict_triple_accuracy": rate("triple_ok", triple_rows),
        "complete_sentence_match_rate": rate("sentence_ok"),
        "one_sentence_rate": rate("one_sentence_ok"),
        "planning_action_n": len(planning_rows),
        "planning_action_accuracy": rate("planning_ok", planning_rows),
        "ab_swap_consistency": (sum(swap_ok_by_id.values()) / len(swap_ok_by_id)
                                if swap_ok_by_id else None),
        "strict_language_all_pass": rate("language_all_ok"),
        "grounding_condition_n": len(grounding_rows),
        "grounding_conditional_accuracy": rate("language_all_ok", grounding_rows),
        "end_to_end_n": len(known_grounding),
        "end_to_end_accuracy": rate("end_to_end_ok", known_grounding),
        "by_question_category": grouped("question_category"),
        "by_target_count": grouped("target_count"),
        "pass_rule": ("keyword AND exact triples AND canonical sentence AND one sentence "
                      "AND planning action when applicable AND A/B swap when applicable; "
                      "end-to-end additionally requires all referenced Grounding targets"),
        "rows": details,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True)
    parser.add_argument("--private-gt", required=True)
    parser.add_argument("--predictions", required=True,
                        help="JSONL rows with qa_id and prediction")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None,
                        help="Optional qa_split filter, for example val")
    args = parser.parse_args()
    qa_rows = read_jsonl(args.qa)
    if args.split:
        qa_rows = [row for row in qa_rows if row.get("qa_split") == args.split]
    keep = {row["qa_id"] for row in qa_rows}
    private_rows = [row for row in read_jsonl(args.private_gt)
                    if row["qa_id"] in keep]
    result = evaluate(qa_rows, read_predictions(args.predictions), private_rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
