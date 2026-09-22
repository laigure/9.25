"""Aggregate T0-T9 screening runs into one comparison table.

Reads screening/runs/*/metrics.json and prints a markdown table plus the
candidate-recall table produced by compute_candidate_recall.py when present.
Screening numbers only; they are not LLM QA results.
"""

import argparse
import json
import os

RELATION_SCENARIOS = ("which_is_left", "closer_to_ego", "closer_of_two",
                      "closest_of_three_to_ego", "relative_location")


def relation_acc(split_report):
    total = 0
    correct = 0.0
    for scenario in RELATION_SCENARIOS:
        entry = split_report["by_scenario"].get(scenario)
        if not entry:
            continue
        total += entry["n"]
        correct += entry["acc"] * entry["n"]
    return round(correct / total, 5) if total else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="/root/autodl-tmp/3eed_data/screening/runs")
    parser.add_argument("--out", default="/root/autodl-tmp/3eed_data/screening/comparison.md")
    parser.add_argument("--recall",
                        default="/root/autodl-tmp/3eed_data/screening/candidate_recall.json")
    args = parser.parse_args()

    lines = ["# T0-T9 screening comparison", "",
             "Lightweight screening models (`screen_token_groups.py`); these are",
             "screening signals only, not LLM QA results.",
             "X rows are extra attribution diagnostics, not part of the directive's",
             "T0-T9 comparison.", "",
             "| group | visual dim | val acc | test acc | relation val | relation test | grounded val acc | missed val acc |",
             "|---|---|---|---|---|---|---|---|"]
    for group in sorted(os.listdir(args.runs) if os.path.isdir(args.runs) else [],
                        key=lambda g: (0 if g.startswith("T") else 1,
                                       int(g[1:]) if g[1:].isdigit() else 99)):
        path = os.path.join(args.runs, group, "metrics.json")
        if not os.path.exists(path):
            continue
        report = json.load(open(path, encoding="utf-8"))
        grounding = report.get("val", {}).get("grounding", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            group, report.get("visual_dim", "-"),
            report.get("val", {}).get("acc"), report.get("test", {}).get("acc"),
            relation_acc(report.get("val", {})), relation_acc(report.get("test", {})),
            grounding.get("acc_when_grounded", "-"),
            grounding.get("acc_when_missed", "-")))

    if os.path.exists(args.recall):
        recall = json.load(open(args.recall, encoding="utf-8"))
        lines += ["", "## Candidate recall (text ranking vs GT, offline diagnostic)",
                  "", "| qa split | layer | recall@1 (full) | recall@3 (full) | recall@5 (full) | recall@1 (noun) |",
                  "|---|---|---|---|---|---|"]
        for split in ("train", "val", "test"):
            for layer in range(6):
                entry = recall.get(split, {}).get(f"layer_{layer}")
                if not entry:
                    continue
                lines.append("| {} | {} | {} | {} | {} | {} |".format(
                    split, layer, entry.get("recall@1_full"), entry.get("recall@3_full"),
                    entry.get("recall@5_full"), entry.get("recall@1_noun")))

    table = "\n".join(lines) + "\n"
    open(args.out, "w", encoding="utf-8").write(table)
    print(table)


if __name__ == "__main__":
    main()
