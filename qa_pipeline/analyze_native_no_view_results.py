"""Create compact JSON/Markdown reports for native no-view QA experiments."""

import argparse
import json
import math
from pathlib import Path


RUNS = {
    "noaux_projector": "native_no_view_pairwise_noaux_projector",
    "noaux_lora": "native_no_view_pairwise_noaux_lora",
    "noaux_shuffled": "native_no_view_pairwise_noaux_shuffled",
    "aux_projector": "native_no_view_pairwise_aux_projector",
    "aux_lora": "native_no_view_pairwise_aux_lora",
    "aux_shuffled": "native_no_view_pairwise_aux_shuffled",
}
SCALARS = (
    "n", "strict_svo_accuracy", "canonical_sentence_match_rate",
    "one_sentence_rate", "strict_all_pass", "inverse_relation_pair_n",
    "inverse_relation_consistency", "grounding_condition_n",
    "grounding_conditional_accuracy", "end_to_end_accuracy",
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(result):
    value = {key: result[key] for key in SCALARS}
    for key in ("by_question_category", "by_scenario", "by_target_count",
                "pass_rule"):
        value[key] = result[key]
    return value


def paired(correct, shuffled):
    left = {row["qa_id"]: bool(row["strict_all_ok"])
            for row in correct["rows"]}
    right = {row["qa_id"]: bool(row["strict_all_ok"])
             for row in shuffled["rows"]}
    if left.keys() != right.keys():
        raise ValueError("correct/shuffled QA IDs differ")
    counts = {"both_correct": 0, "correct_only": 0,
              "shuffled_only": 0, "both_wrong": 0}
    for qa_id, a in left.items():
        b = right[qa_id]
        key = ("both_correct" if a and b else "correct_only" if a else
               "shuffled_only" if b else "both_wrong")
        counts[key] += 1
    n = len(left)
    delta = (counts["correct_only"] - counts["shuffled_only"]) / n
    discordant = (counts["correct_only"] + counts["shuffled_only"]) / n
    se = math.sqrt(max(discordant - delta * delta, 0.0) / n)
    return {
        "n": n, **counts, "accuracy_delta": delta,
        "paired_standard_error": se,
        "normal_95ci": [delta - 1.96 * se, delta + 1.96 * se],
        "z_score": delta / se if se else 0.0,
    }


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounding-selection", type=Path)
    args = parser.parse_args()

    raw, metrics, training = {}, {}, {}
    for label, directory in RUNS.items():
        run = args.runs_root / directory
        raw[label] = load(run / "eval_val.json")
        metrics[label] = compact(raw[label])
        history = run / "history.json"
        if history.exists():
            value = load(history)
            training[label] = {
                key: value[key] for key in
                ("val_loss", "epoch_seconds", "total_steps")
            }

    report = {
        "dataset": "native_natural_qa_v2_no_view",
        "evaluation_rows": metrics["aux_lora"]["n"],
        "public_input": (
            "natural non-spatial descriptions + implicit Grounding tokens; "
            "no public view names or explicit geometry"),
        "models": metrics,
        "token_ablation": {
            "noaux": paired(raw["noaux_lora"], raw["noaux_shuffled"]),
            "aux": paired(raw["aux_lora"], raw["aux_shuffled"]),
        },
        "aux_lora_minus_noaux_lora": {
            key: metrics["aux_lora"][key] - metrics["noaux_lora"][key]
            for key in ("strict_all_pass", "inverse_relation_consistency",
                        "grounding_conditional_accuracy", "end_to_end_accuracy")
        },
        "training": training,
    }
    if args.grounding_selection and args.grounding_selection.exists():
        report["unchanged_grounding"] = load(args.grounding_selection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_metrics_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = [
        "# Native no-view QA final results", "",
        "The public prompt contains natural non-spatial descriptions and "
        "implicit Grounding tokens. View names and explicit geometry are absent.",
        "", "## Main metrics", "",
        "| Run | Strict | Grounding conditional | End-to-end | Inverse consistency |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in RUNS:
        item = metrics[label]
        lines.append("| {} | {} | {} | {} | {} |".format(
            label, pct(item["strict_all_pass"]),
            pct(item["grounding_conditional_accuracy"]),
            pct(item["end_to_end_accuracy"]),
            pct(item["inverse_relation_consistency"])))
    lines += ["", "## Token ablation", ""]
    for arm in ("noaux", "aux"):
        value = report["token_ablation"][arm]
        lines.append(
            f"- **{arm}:** correct minus shuffled = "
            f"{100 * value['accuracy_delta']:.2f} pp; paired 95% CI "
            f"[{100 * value['normal_95ci'][0]:.2f}, "
            f"{100 * value['normal_95ci'][1]:.2f}] pp; "
            f"correct-only/shuffled-only = {value['correct_only']}/"
            f"{value['shuffled_only']}.")
    lines += ["", "## Auxiliary LoRA by scenario", "",
              "| Scenario | Correct token | Shuffled token | Gain |",
              "|---|---:|---:|---:|"]
    correct = metrics["aux_lora"]["by_scenario"]
    shuffled = metrics["aux_shuffled"]["by_scenario"]
    for scenario in sorted(correct):
        a, b = correct[scenario]["accuracy"], shuffled[scenario]["accuracy"]
        lines.append(f"| {scenario} | {pct(a)} | {pct(b)} | "
                     f"{100 * (a - b):+.2f} pp |")
    lines += ["", "## Interpretation", "",
              "- Without auxiliary relation supervision, correct tokens do not "
              "significantly outperform shuffled tokens.",
              "- Auxiliary relation supervision produces a large, statistically "
              "clear token-dependent gain, especially for cardinal relations, "
              "relative distance and object motion.",
              "- Exact metric-distance generation remains near zero and needs a "
              "separate design; the overall distance category must not hide this.",
              "- Ego-motion remains mostly answerable from language priors because "
              "its correct/shuffled gap is small.", ""]
    (args.output_dir / "FINAL_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "strict": {key: metrics[key]["strict_all_pass"] for key in metrics},
        "token_ablation": report["token_ablation"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
