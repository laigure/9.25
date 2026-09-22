"""Aggregate Qwen LoRA group runs (T1/T6/T9) into comparison tables.

Reads runs/group_{G}/eval_{val,test}{,_swap,_shuffle42}.json written by
train_group_qa.py and prints:
  1) main table: full val/test accuracy (test = main score), relation subset,
     grounding-grouped diagnostics, parse rate;
  2) probe table: A/B swap (flip/content rates) and shuffled-feature accuracy.

Missing files are reported as "missing" instead of failing, so partial runs
are still inspectable.
"""

import argparse
import json
import os


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fmt(entry, key, digits=4):
    if entry is None:
        return "missing"
    value = entry
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return "-"
        value = value[part]
    if value is None:
        return "-"
    if isinstance(value, float):
        return round(value, digits)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="/root/autodl-tmp/3eed_data/qa_v3/runs")
    parser.add_argument("--groups", default="T1,T6,T9")
    parser.add_argument("--out", default="/root/autodl-tmp/3eed_data/qa_v3/group_comparison.md")
    args = parser.parse_args()

    lines = ["# Qwen2.5-7B LoRA group comparison (full val/test)", "",
             "Same questions, same hyper-parameters (v3 settings), same eval script;",
             "only the visual input group differs. Test is the main score.",
             "Full val/test = no per-scenario cap (`--eval-per-scenario 0`).",
             "Probe rows use multi-object questions only (`--only-multi-ref`).", ""]

    main_rows, probe_rows, shuffle_rows = [], [], []
    for group in args.groups.split(","):
        run_dir = os.path.join(args.runs, "group_" + group)
        val = load(os.path.join(run_dir, "eval_val.json"))
        test = load(os.path.join(run_dir, "eval_test.json"))
        val_swap = load(os.path.join(run_dir, "eval_val_swap.json"))
        test_swap = load(os.path.join(run_dir, "eval_test_swap.json"))
        val_shuf = load(os.path.join(run_dir, "eval_val_shuffle42.json"))
        test_shuf = load(os.path.join(run_dir, "eval_test_shuffle42.json"))
        main_rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            group,
            fmt(val, "n"), fmt(val, "accuracy"), fmt(test, "accuracy"),
            fmt(val, "relation.accuracy"), fmt(test, "relation.accuracy"),
            fmt(val, "grounding.acc_when_grounded"),
            fmt(val, "grounding.acc_when_missed"),
            fmt(val, "parse_rate"), fmt(test, "parse_rate")))
        for split, entry in (("val", val_swap), ("test", test_swap)):
            probe_rows.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                group, split,
                fmt(entry, "swap.n_applicable"),
                fmt(entry, "swap.acc_normal"), fmt(entry, "swap.acc_swapped"),
                fmt(entry, "swap.flip_rate"), fmt(entry, "swap.content_rate")))
        shuffle_rows.append("| {} | {} | {} | {} | {} | {} |".format(
            group, fmt(val_shuf, "n"), fmt(val_shuf, "accuracy"),
            fmt(test_shuf, "accuracy"), fmt(val_shuf, "relation.accuracy"),
            fmt(test_shuf, "relation.accuracy")))

    lines += ["| group | val n | val acc | test acc | val relation | test relation | val grounded | val missed | val parse | test parse |",
              "|---|---|---|---|---|---|---|---|---|---|"] + main_rows
    lines += ["", "## A/B swap probe (multi-object questions only)", "",
              "acc normal / acc A/B: same rows, blocks of roles A and B exchanged;",
              "flip: swapped prediction mirrors the normal prediction (order-driven);",
              "content: swapped prediction mirrors the GT key (feature-content-driven).", "",
              "| group | split | swap n | acc normal | acc A/B | flip | content |",
              "|---|---|---|---|---|---|---|"] + probe_rows
    lines += ["", "## Shuffled-feature probe (multi-object questions only)", "",
              "Features replaced by those of a same (split, category) donor object, seed 42;",
              "compare with the main table's relation columns.", "",
              "| group | val n | val acc | test acc | val relation | test relation |",
              "|---|---|---|---|---|---|"] + shuffle_rows

    table = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(table)
    print(table)


if __name__ == "__main__":
    main()
