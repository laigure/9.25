"""Strict evaluator for natural-description spatial QA.

The private answer key names the expected subject, predicate and object.  A
prediction passes only when all required entities occur in order, the expected
predicate is present, contradictory predicates are absent, and metric answers
contain the canonical one-decimal value.  Canonical sentence match is reported
separately and is also part of ``strict_all_pass``.
"""

import collections
import json
import re


RELATION_PHRASES = {
    "left": "to the left of",
    "right": "to the right of",
    "front": "in front of",
    "behind": "behind",
    "closer_to": "closer to",
    "gets_closer_to": "gets closer to",
    "does_not_get_closer_to": "does not get closer to",
    "distance_from": "distance from",
}
OPPOSITE = {"left": "right", "right": "left", "front": "behind",
            "behind": "front", "gets_closer_to": "does not get closer to",
            "does_not_get_closer_to": "gets closer to"}


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()))


def phrase_present(text, phrase):
    return norm(phrase) in norm(text)


def entities_in_order(text, entities):
    value = norm(text)
    cursor = 0
    for entity in entities:
        needle = norm(entity)
        index = value.find(needle, cursor)
        if index < 0:
            return False
        cursor = index + len(needle)
    return True


def predicate_ok(prediction, key):
    relation = key["relation"]
    expected = RELATION_PHRASES[relation]
    if not phrase_present(prediction, expected):
        return False
    opposite = OPPOSITE.get(relation)
    if opposite and phrase_present(prediction, RELATION_PHRASES.get(opposite, opposite)):
        return False
    if relation == "gets_closer_to" and phrase_present(
            prediction, RELATION_PHRASES["does_not_get_closer_to"]):
        return False
    return True


def svo_ok(prediction, key):
    key_type = key["type"]
    relation = key["relation"]
    subject = key["subject"]
    object_name = key["object"]
    if key_type == "metric_distance":
        entities = [subject, object_name]
        if not entities_in_order(prediction, entities):
            return False
        values = [float(value) for value in re.findall(
            r"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)\s*meters?\b",
            prediction.lower())]
        return len(values) == 1 and abs(values[0] - float(key["meters"])) < 0.051
    if key_type == "relative_distance":
        return (entities_in_order(
            prediction, [subject, object_name, key["comparison"]]) and
                predicate_ok(prediction, key))
    if key_type == "motion_relation":
        return (entities_in_order(prediction, [subject, object_name]) and
                phrase_present(prediction, key["motion_direction"]) and
                predicate_ok(prediction, key))
    if key_type == "spatial_triple":
        return (entities_in_order(prediction, [subject, object_name]) and
                predicate_ok(prediction, key))
    raise ValueError(f"unknown answer key type: {key_type}")


def one_sentence(text):
    value = text.strip()
    if not value or value[-1] not in ".!?":
        return False
    # Decimal points do not count as sentence boundaries.
    stripped = re.sub(r"(?<=\d)\.(?=\d)", "", value)
    return len(re.findall(r"[.!?]", stripped)) == 1


def rate(rows, field):
    return sum(bool(row[field]) for row in rows) / max(len(rows), 1)


def grouped(rows, field, group_field):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(row[group_field])].append(row)
    return {key: {"n": len(group), "accuracy": rate(group, field)}
            for key, group in sorted(groups.items())}


def evaluate(records, prediction_rows, private_rows):
    predictions = {row["qa_id"]: row["prediction"] for row in prediction_rows}
    private = {row["qa_id"]: row for row in private_rows}
    rows = []
    inverse_groups = collections.defaultdict(list)
    for record in records:
        qa_id = record["qa_id"]
        prediction = predictions[qa_id].strip()
        truth = private[qa_id]
        key = truth["answer_key"]
        exact = norm(prediction) == norm(truth["canonical_answer"])
        structured = svo_ok(prediction, key)
        sentence = one_sentence(prediction)
        strict = bool(exact and structured and sentence)
        grounding_pass = bool(truth.get("diagnostic_grounding_pass", True))
        row = {
            "qa_id": qa_id, "scenario": record["scenario"],
            "question_category": record["question_category"],
            "target_count": record["target_count"], "prediction": prediction,
            "canonical_answer": truth["canonical_answer"],
            "answer_key": key, "svo_ok": bool(structured),
            "canonical_sentence_ok": bool(exact),
            "one_sentence_ok": bool(sentence), "strict_all_ok": strict,
            "grounding_pass": grounding_pass,
            "end_to_end_ok": bool(strict and grounding_pass),
        }
        rows.append(row)
        if key.get("inverse_pair_id"):
            inverse_groups[key["inverse_pair_id"]].append(row)
    grounding_rows = [row for row in rows if row["grounding_pass"]]
    inverse_complete = [group for group in inverse_groups.values()
                        if len(group) == 2]
    result = {
        "n": len(rows),
        "strict_svo_accuracy": rate(rows, "svo_ok"),
        "canonical_sentence_match_rate": rate(rows, "canonical_sentence_ok"),
        "one_sentence_rate": rate(rows, "one_sentence_ok"),
        "strict_all_pass": rate(rows, "strict_all_ok"),
        "inverse_relation_pair_n": len(inverse_complete),
        "inverse_relation_consistency": (
            sum(all(row["strict_all_ok"] for row in group)
                for group in inverse_complete) / max(len(inverse_complete), 1)),
        "grounding_condition_n": len(grounding_rows),
        "grounding_conditional_accuracy": rate(grounding_rows, "strict_all_ok"),
        "end_to_end_accuracy": rate(rows, "end_to_end_ok"),
        "by_question_category": grouped(rows, "strict_all_ok",
                                         "question_category"),
        "by_scenario": grouped(rows, "strict_all_ok", "scenario"),
        "by_target_count": grouped(rows, "strict_all_ok", "target_count"),
        "pass_rule": (
            "canonical full sentence AND one sentence AND strict private "
            "subject-predicate-object/value key"
        ),
        "rows": rows,
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--private-gt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in open(path, encoding="utf-8")
                         if line.strip()]
    result = evaluate(read(args.qa), read(args.predictions), read(args.private_gt))
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
