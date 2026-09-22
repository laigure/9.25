"""Shared answer parser and metrics for every QA experiment (A, B, C).

Model output is free text; the canonical answer template written by
build_qa.py is

    "The target object is located to the left and in front of the reference
     object, approximately 8.5 meters away."

Experiments A (GT baseline), B (projector) and C (projector+LoRA) all read
their predictions through parse_answer() and score them with evaluate(), so
the comparison uses byte-identical evaluation code.

Definitions (computed over every evaluated row; a missing field counts as a
wrong answer on that axis):
    format_validity       lateral + longitudinal + distance all parsed
    left_right_accuracy   lateral component correct
    front_behind_accuracy longitudinal component correct
    exact_accuracy        both components correct
    distance MAE / median |pred - GT|, over rows with a parsed distance

The module is numpy-free so the same file runs locally and on the training
host.
"""

import re
import statistics

LATERAL_PRIMARY = re.compile(r"\bto the (left|right)\b", re.IGNORECASE)
LATERAL_ANY = re.compile(r"\b(left|right)\b", re.IGNORECASE)
FRONT = re.compile(r"\bin front of\b|\bin front\b", re.IGNORECASE)
BEHIND = re.compile(r"\bbehind\b|\bin back of\b", re.IGNORECASE)
DISTANCE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m(?:eters?)?\b", re.IGNORECASE)


def _unique(values):
    """Single distinct value, else None (ambiguous text fails the parse)."""
    distinct = {value.lower() for value in values}
    return distinct.pop() if len(distinct) == 1 else None


def parse_answer(text):
    text = "" if text is None else str(text)
    lateral_matches = LATERAL_PRIMARY.findall(text)
    if not lateral_matches:
        lateral_matches = LATERAL_ANY.findall(text)
    lateral = _unique(lateral_matches)

    front_hit = bool(FRONT.search(text))
    behind_hit = bool(BEHIND.search(text))
    longitudinal = None
    if front_hit != behind_hit:
        longitudinal = "front" if front_hit else "behind"

    distance_match = DISTANCE.search(text)
    distance = float(distance_match.group(1)) if distance_match else None

    return {
        "lateral": lateral,
        "longitudinal": longitudinal,
        "distance": distance,
        "format_ok": (lateral is not None and longitudinal is not None
                      and distance is not None),
    }


def evaluate(records, predictions, tag="all"):
    """Score free-text predictions against the clean QA records."""
    rows = []
    for record, text in zip(records, predictions):
        parsed = parse_answer(text)
        gt_lateral = record["relations"][0]
        gt_longitudinal = record["relations"][1]
        gt_distance = record["relative_geometry"]["distance_3d"]
        rows.append({
            "qa_id": record["qa_id"],
            "qa_split": record.get("qa_split"),
            "bucket": record["grounding_quality"]["bucket"],
            "question": record["question"],
            "prediction": text,
            "gt_lateral": gt_lateral,
            "gt_longitudinal": gt_longitudinal,
            "gt_distance": gt_distance,
            "pred_lateral": parsed["lateral"],
            "pred_longitudinal": parsed["longitudinal"],
            "pred_distance": parsed["distance"],
            "format_ok": parsed["format_ok"],
            "lateral_ok": parsed["lateral"] == gt_lateral,
            "longitudinal_ok": parsed["longitudinal"] == gt_longitudinal,
            "exact_ok": (parsed["lateral"] == gt_lateral
                         and parsed["longitudinal"] == gt_longitudinal),
            "distance_abs_error": (abs(parsed["distance"] - gt_distance)
                                   if parsed["distance"] is not None else None),
        })
    return {"tag": tag, "n": len(rows), "rows": rows, "metrics": _aggregate(rows)}


def _aggregate(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def rate(flags_subset, flags):
        values = list(flags)
        return round(sum(1 for flag in values if flag) / len(values), 4) if values else None

    format_ok_rows = [row for row in rows if row["format_ok"]]
    distances = [row["distance_abs_error"] for row in rows
                 if row["distance_abs_error"] is not None]

    per_relation = {}
    for lateral in ("left", "right"):
        for longitudinal in ("front", "behind"):
            subset = [row for row in rows if row["gt_lateral"] == lateral
                      and row["gt_longitudinal"] == longitudinal]
            if subset:
                per_relation["{}_and_{}".format(lateral, longitudinal)] = {
                    "n": len(subset),
                    "exact_accuracy": rate(subset, (row["exact_ok"] for row in subset)),
                }

    per_bucket = {}
    for name in ("high_high", "high_low", "low_low"):
        subset = [row for row in rows if row["bucket"] == name]
        if subset:
            per_bucket[name] = {
                "n": len(subset),
                "exact_accuracy": rate(subset, (row["exact_ok"] for row in subset)),
                "left_right_accuracy": rate(subset, (row["lateral_ok"] for row in subset)),
                "front_behind_accuracy": rate(subset, (row["longitudinal_ok"] for row in subset)),
            }

    return {
        "n": n,
        "format_validity": rate(rows, (row["format_ok"] for row in rows)),
        "left_right_accuracy": rate(rows, (row["lateral_ok"] for row in rows)),
        "front_behind_accuracy": rate(rows, (row["longitudinal_ok"] for row in rows)),
        "exact_accuracy": rate(rows, (row["exact_ok"] for row in rows)),
        "conditional_on_format": {
            "n": len(format_ok_rows),
            "exact_accuracy": rate(format_ok_rows, (row["exact_ok"] for row in format_ok_rows)),
        },
        "distance": {
            "n_scored": len(distances),
            "mae": round(sum(distances) / len(distances), 4) if distances else None,
            "median_abs_error": round(float(statistics.median(distances)), 4) if distances else None,
        },
        "per_relation": per_relation,
        "per_bucket": per_bucket,
    }
