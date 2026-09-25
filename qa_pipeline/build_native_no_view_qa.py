"""Remove public view names from native-caption QA without changing Grounding.

The source QA IDs, object IDs, private geometry and answer keys are preserved.
Only the public natural-language question/answer and public object references
are changed.  Private source view metadata remains available for auditing but
never enters the model prompt.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from strict_natural_qa_eval import evaluate


FRAME_SENTENCE = re.compile(
    r"The front view corresponds to north\.\s*", re.IGNORECASE)
PUBLIC_VIEW = re.compile(r"\bview\b", re.IGNORECASE)
ROLE_ALIAS = re.compile(r"\bobject\s+[a-z]\b", re.IGNORECASE)
COORDINATE = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?")


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")
            if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def remove_view_phrases(text, views):
    text = FRAME_SENTENCE.sub(
        "Use the ego vehicle's current heading as north. ", text)
    for view in sorted(set(views), key=len, reverse=True):
        text = re.sub(r"\s+in\s+the\s+" + re.escape(view), "", text,
                      flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert(public_rows, private_rows):
    private_by_id = {row["qa_id"]: dict(row) for row in private_rows}
    converted = []
    for source in public_rows:
        row = dict(source)
        views = [ref["view"] for ref in row["object_refs"]]
        descriptions = [ref["description"] for ref in row["object_refs"]]
        if len(set(descriptions)) != len(descriptions):
            raise ValueError(
                f"ambiguous descriptions after hiding views: {row['qa_id']}")
        row["question"] = remove_view_phrases(row["question"], views)
        row["answer"] = remove_view_phrases(row["answer"], views)
        row["object_refs"] = [
            {key: value for key, value in ref.items() if key != "view"}
            for ref in row["object_refs"]
        ]
        row["input_policy"] = "natural_descriptions_and_implicit_tokens"
        if PUBLIC_VIEW.search(row["question"] + " " + row["answer"]):
            raise ValueError(f"public view leakage remains: {row['qa_id']}")
        converted.append(row)
        private_by_id[row["qa_id"]]["canonical_answer"] = row["answer"]
    private = [private_by_id[row["qa_id"]] for row in converted]
    return converted, private


def audit(rows, private):
    counters = Counter()
    seen = set()
    for row in rows:
        counters["rows"] += 1
        counters[f"split:{row['qa_split']}"] += 1
        counters[f"category:{row['question_category']}"] += 1
        counters[f"scenario:{row['scenario']}"] += 1
        counters[f"N:{row['target_count']}"] += 1
        public_text = row["question"] + " " + row["answer"]
        if PUBLIC_VIEW.search(public_text):
            counters["public_view_text_leaks"] += 1
        if any("view" in ref for ref in row["object_refs"]):
            counters["public_view_field_leaks"] += 1
        if ROLE_ALIAS.search(public_text):
            counters["object_alias_leaks"] += 1
        if COORDINATE.search(row["question"]):
            counters["coordinate_leaks"] += 1
        signature = (row["qa_split"], row["question"],
                     tuple(ref["object_id"] for ref in row["object_refs"]))
        if signature in seen:
            counters["duplicate_model_inputs"] += 1
        seen.add(signature)
    required_zero = [
        "public_view_text_leaks", "public_view_field_leaks",
        "object_alias_leaks", "coordinate_leaks", "duplicate_model_inputs",
    ]
    for key in required_zero:
        if counters[key]:
            raise ValueError(f"no-view QA audit failed: {key}={counters[key]}")

    predictions = [{"qa_id": row["qa_id"], "prediction": row["answer"]}
                   for row in rows]
    oracle = evaluate(rows, predictions, private)
    for key in ("strict_svo_accuracy", "canonical_sentence_match_rate",
                "one_sentence_rate", "strict_all_pass"):
        if oracle[key] != 1.0:
            raise ValueError(f"oracle {key}={oracle[key]}")
    return {
        "counts": dict(counters),
        "strict_zero_checks": required_zero,
        "oracle": {key: oracle[key] for key in (
            "strict_svo_accuracy", "canonical_sentence_match_rate",
            "one_sentence_rate", "strict_all_pass",
            "inverse_relation_consistency")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, private = convert(
        read_jsonl(args.source / "qa.jsonl"),
        read_jsonl(args.source / "private_gt.jsonl"))
    report = audit(rows, private)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "qa.jsonl", rows)
    write_jsonl(args.output / "private_gt.jsonl", private)
    examples = {}
    for row in rows:
        examples.setdefault(row["scenario"], row)
    (args.output / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    summary = {
        "dataset_version": "native_natural_qa_v2_no_view",
        "source_dataset": str(args.source),
        "grounding_change": "none; object IDs and token bindings are preserved",
        "rows": len(rows),
        "public_input": (
            "natural non-spatial descriptions + ordered implicit Grounding "
            "tokens; no view names, Object-A/B aliases or explicit geometry"),
        "private_only": "GT geometry, source view metadata and answer keys",
        "audit": report,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
