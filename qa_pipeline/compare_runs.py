"""Compare Experiment B (projector only) and Experiment C (projector + LoRA).

Reads the run directories produced by train_qa.py:
    <run>/history.json        train-loss points + per-epoch val loss
    <run>/eval_val.json       val metrics + per-row predictions
    <run>/samples20_val.json  20 fixed val samples (identical qa_ids in both)
Baselines: the random baseline json from Experiment A and, if given, the
text-only run directory (train_qa.py --mode eval --no-tokens).

Writes a markdown report + json next to it with:
  * metric table and delta exact accuracy = LoRA - projector-only
  * train-loss drop (% from first to last logged point)
  * the directive's reference targets as diagnostic flags (not gates)
  * the 20 identical samples side by side (before/after)
  * configuration equality check (same data, same eval code, same prompt)

Usage:
    python compare_runs.py --projector-dir data/runs/projector \
        --lora-dir data/runs/lora --random-a data/experiment_a_gt_baseline.json \
        --text-only-dir data/runs/text_only --out-dir reports
"""

import argparse
import json
import os


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projector-dir", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--random-a", default=None)
    parser.add_argument("--text-only-dir", default=None)
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--name", default="b_vs_c_report")
    args = parser.parse_args()

    projector_eval = load_json(os.path.join(args.projector_dir, "eval_val.json"))
    lora_eval = load_json(os.path.join(args.lora_dir, "eval_val.json"))
    projector_history = load_json(os.path.join(args.projector_dir, "history.json"))
    lora_history = load_json(os.path.join(args.lora_dir, "history.json"))
    text_only_eval = (load_json(os.path.join(args.text_only_dir, "eval_val.json"))
                      if args.text_only_dir else None)
    random_a = load_json(args.random_a) if args.random_a else None

    rows = []
    table = {
        "n": ("n", lambda m: m["n"]),
        "format_validity": ("format validity", lambda m: m["format_validity"]),
        "left_right_accuracy": ("left/right acc", lambda m: m["left_right_accuracy"]),
        "front_behind_accuracy": ("front/behind acc", lambda m: m["front_behind_accuracy"]),
        "exact_accuracy": ("exact acc", lambda m: m["exact_accuracy"]),
        "high_high_exact": ("high-high exact", lambda m: m["per_bucket"]["high_high"]["exact_accuracy"]),
        "high_low_exact": ("high-low exact", lambda m: m["per_bucket"]["high_low"]["exact_accuracy"]),
        "low_low_exact": ("low-low exact", lambda m: m["per_bucket"]["low_low"]["exact_accuracy"]),
        "distance_mae": ("distance MAE (m)", lambda m: m["distance"]["mae"]),
        "distance_median_ae": ("distance median err (m)", lambda m: m["distance"]["median_abs_error"]),
    }

    def metrics_of(eval_json):
        return eval_json["metrics"] if eval_json else None

    print("{:<24} {:>10} {:>10} {:>10}".format("metric", "projector", "lora", "delta"))

    def value_of(source_metrics, extractor):
        if source_metrics is None:
            return None
        try:
            return extractor(source_metrics)
        except (KeyError, TypeError):
            return None

    projector_metrics = metrics_of(projector_eval)
    lora_metrics = metrics_of(lora_eval)
    for _, (label, extractor) in table.items():
        proj_value = value_of(projector_metrics, extractor)
        lora_value = value_of(lora_metrics, extractor)
        delta = (None if proj_value is None or lora_value is None
                 else round(lora_value - proj_value, 4))
        print("{:<24} {:>10} {:>10} {:>10}".format(
            label, str(proj_value), str(lora_value), str(delta)))
        rows.append({"metric": label, "projector": proj_value,
                     "lora": lora_value, "delta": delta})

    def loss_drop(history):
        if not history or not history.get("train"):
            return None
        losses = [entry["loss"] for entry in history["train"]]
        head = sum(losses[:3]) / min(3, len(losses))
        tail = sum(losses[-3:]) / min(3, len(losses))
        return {"first": round(head, 4), "last": round(tail, 4),
                "drop": round((head - tail) / head, 4) if head else None}

    proj_drop = loss_drop(projector_history)
    lora_drop = loss_drop(lora_history)

    baselines = {}
    if random_a:
        baselines["random"] = random_a.get("random_baseline")
    if text_only_eval:
        baselines["text_only"] = text_only_eval["metrics"]

    def target_flags(metrics, drop):
        if metrics is None:
            return None
        flags = {
            "train_loss_drop_ge_0.40": (drop or {}).get("drop") is not None
                                       and drop["drop"] >= 0.40,
            "left_right_ge_0.80": metrics["left_right_accuracy"] >= 0.80,
            "front_behind_ge_0.80": metrics["front_behind_accuracy"] >= 0.80,
            "exact_ge_0.65": metrics["exact_accuracy"] >= 0.65,
            "high_high_exact_ge_0.75": (
                metrics["per_bucket"]["high_high"]["exact_accuracy"] >= 0.75),
            "format_validity_ge_0.95": metrics["format_validity"] >= 0.95,
        }
        best_baseline = 0.0
        for name in ("random", "text_only"):
            value = (baselines.get(name) or {}).get("exact_accuracy")
            if value is not None:
                best_baseline = max(best_baseline, value)
        flags["exact_ge_baseline_plus_0.20"] = (
            metrics["exact_accuracy"] >= best_baseline + 0.20 and best_baseline > 0)
        return flags

    projector_samples = load_json(os.path.join(args.projector_dir,
                                               "samples20_val.json"))
    lora_samples = load_json(os.path.join(args.lora_dir, "samples20_val.json"))
    side_by_side = []
    if projector_samples and lora_samples:
        lora_by_id = {row["qa_id"]: row for row in lora_samples}
        for row in projector_samples:
            other = lora_by_id.get(row["qa_id"])
            if other is None:
                continue
            side_by_side.append({
                "qa_id": row["qa_id"],
                "question": row["question"],
                "gt": {"lateral": row["gt_lateral"],
                       "longitudinal": row["gt_longitudinal"],
                       "distance": round(row["gt_distance"], 2)},
                "projector": {"prediction": row["prediction"],
                              "exact_ok": row["exact_ok"]},
                "lora": {"prediction": other["prediction"],
                         "exact_ok": other["exact_ok"]},
            })

    config_check = {}
    if projector_eval and lora_eval:
        proj_cfg, lora_cfg = projector_eval.get("config", {}), lora_eval.get("config", {})
        for key in ("data", "eval_splits", "seed", "no_tokens", "max_eval"):
            config_check[key] = {"projector": proj_cfg.get(key),
                                 "lora": lora_cfg.get(key),
                                 "equal": proj_cfg.get(key) == lora_cfg.get(key)}
        config_check["projector_init"] = lora_cfg.get("init_projector")

    report = {
        "metric_table": rows,
        "loss": {"projector": proj_drop, "lora": lora_drop},
        "targets": {"projector": target_flags(projector_metrics, proj_drop),
                    "lora": target_flags(lora_metrics, lora_drop)},
        "baselines": baselines,
        "config_check": config_check,
        "samples": side_by_side,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.name + ".json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    md = ["# Experiment B vs C (projector-only vs projector+LoRA)", ""]
    md.append("| metric | projector | lora | delta |")
    md.append("|---|---|---|---|")
    for row in rows:
        md.append("| {} | {} | {} | {} |".format(row["metric"], row["projector"],
                                                 row["lora"], row["delta"]))
    md.append("")
    md.append("Loss drop (first->last logged): projector {}, lora {}".format(
        proj_drop, lora_drop))
    md.append("")
    md.append("Baselines: {}".format(json.dumps(baselines)))
    md.append("")
    md.append("Config equality: {}".format(json.dumps(config_check)))
    md.append("")
    md.append("## 20 identical validation samples")
    for sample in side_by_side:
        md.append("- Q: {}".format(sample["question"]))
        md.append("  GT: {} {} {:.1f} m".format(
            sample["gt"]["lateral"], sample["gt"]["longitudinal"],
            sample["gt"]["distance"]))
        md.append("  projector (exact={}): {}".format(
            sample["projector"]["exact_ok"], sample["projector"]["prediction"]))
        md.append("  lora (exact={}): {}".format(
            sample["lora"]["exact_ok"], sample["lora"]["prediction"]))
    with open(os.path.join(args.out_dir, args.name + ".md"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")

    print()
    print("loss drop", {"projector": proj_drop, "lora": lora_drop})
    print("baselines", json.dumps(baselines))
    print("targets", json.dumps(report["targets"], indent=2))
    print("wrote", json_path, "and", os.path.join(args.out_dir, args.name + ".md"))


if __name__ == "__main__":
    main()
