"""C/D/E/oracle/relative-geometry ablation table from the run directories.

Reads only what train_qa.py already wrote (eval_<split>.json, history.json,
samples20_<split>.json) so C's baseline stays untouched. Everything the D
directive asks for that is not in the eval metrics is recomputed here from the
per-sample rows: per-GT-distance-bin predicted means and MAE, Pearson
correlation of predicted vs GT distance, predicted-distance std, and a
collapse flag for the "everything ~10 m" failure mode. D's success criteria
(any two of five) are evaluated against the baseline run.

Usage:
    python qa_pipeline/ablation_report.py --run C=data/runs/lora \
        --run D=data/runs/d_center --run oracle=data/runs/oracle_gt_center \
        --baseline C --out reports/ablation_c_d_e
"""

import argparse
import json
import os

import numpy as np

BINS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, float("inf"))]
BIN_LABELS = ["0-5m", "5-10m", "10-15m", "15m+"]
# A run counts as "collapsed toward ~10 m" when the predicted bin means cover
# less than half of the GT bin-mean span: C itself sits at 0.41 (6.5 m of a
# 15.9 m GT span), which is exactly the compression the ablation targets.
SPREAD_RATIO_MIN = 0.5


def load_json(path):
    with open(path, encoding="utf-8") as fobj:
        return json.load(fobj)


def distance_diagnostics(rows):
    gt = np.array([r["gt_distance"] for r in rows], dtype=np.float64)
    pred = np.array([r["pred_distance"] if r["pred_distance"] is not None
                     else np.nan for r in rows], dtype=np.float64)
    scored = ~np.isnan(pred)
    out = {"n_scored": int(scored.sum())}
    if scored.sum() == 0:
        return out
    g, p = gt[scored], pred[scored]
    out["pred_mean"] = round(float(p.mean()), 4)
    out["pred_std"] = round(float(p.std()), 4)
    out["gt_std"] = round(float(g.std()), 4)
    out["std_ratio"] = round(float(p.std() / g.std()), 4) \
        if g.std() > 0 else None
    out["pearson"] = round(float(np.corrcoef(p, g)[0, 1]), 4) \
        if p.std() > 0 and g.std() > 0 else None
    bins = {}
    means = []
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        mask = (g >= lo) & (g < hi)
        if mask.sum() == 0:
            bins[label] = {"n": 0}
            means.append(None)
            continue
        err = np.abs(p[mask] - g[mask])
        bins[label] = {"n": int(mask.sum()),
                       "gt_mean": round(float(g[mask].mean()), 4),
                       "pred_mean": round(float(p[mask].mean()), 4),
                       "mae": round(float(err.mean()), 4)}
        means.append(bins[label]["pred_mean"])
    out["bins"] = bins
    out["bin_pred_means_monotonic"] = bool(
        all(a is not None and b is not None and b > a
            for a, b in zip(means, means[1:])))
    gt_present = [bins[l]["gt_mean"] for l in BIN_LABELS if bins[l].get("n")]
    pred_present = [bins[l]["pred_mean"] for l in BIN_LABELS if bins[l].get("n")]
    out["bin_spread_ratio"] = round(
        (max(pred_present) - min(pred_present))
        / (max(gt_present) - min(gt_present)), 4) \
        if len(gt_present) > 1 and max(gt_present) > min(gt_present) else None
    out["collapse_suspect"] = bool(
        out["bin_spread_ratio"] is not None
        and out["bin_spread_ratio"] < SPREAD_RATIO_MIN)
    return out


def run_summary(run_dir, split):
    path = os.path.join(run_dir, "eval_{}.json".format(split))
    if not os.path.exists(path):
        return {"status": "not run"}
    data = load_json(path)
    metrics = data["metrics"]
    summary = {
        "status": "ok",
        "dir": run_dir.replace("\\", "/"),
        "n": metrics["n"],
        "format_validity": metrics["format_validity"],
        "left_right": metrics["left_right_accuracy"],
        "front_behind": metrics["front_behind_accuracy"],
        "exact": metrics["exact_accuracy"],
        "buckets": {k: v["exact_accuracy"]
                    for k, v in metrics["per_bucket"].items()},
        "distance_mae": metrics["distance"]["mae"],
        "distance_median": metrics["distance"]["median_abs_error"],
        "distance_diag": distance_diagnostics(data["rows"]),
    }
    hist = os.path.join(run_dir, "history.json")
    if os.path.exists(hist):
        summary["history"] = load_json(hist)
    return summary


def success_criteria(base, cand):
    """D success = at least two of the five criteria in the directive."""
    checks = {}
    if base["status"] != "ok" or cand["status"] != "ok":
        return checks, 0
    checks["fb_plus_5pp"] = bool(cand["front_behind"] - base["front_behind"]
                                 >= 0.05)
    checks["exact_plus_5pp"] = bool(cand["exact"] - base["exact"] >= 0.05)
    checks["dist_mae_down_20pct"] = bool(
        cand["distance_mae"] <= 0.8 * base["distance_mae"])
    pearson = cand["distance_diag"].get("pearson")
    checks["pearson_ge_0.5"] = bool(pearson is not None and pearson >= 0.5)
    spread = cand["distance_diag"].get("bin_spread_ratio")
    checks["bin_means_monotonic_and_spread"] = bool(
        cand["distance_diag"].get("bin_pred_means_monotonic", False)
        and spread is not None and spread >= SPREAD_RATIO_MIN)
    return checks, sum(1 for v in checks.values() if v)


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "{:.{}f}".format(value, digits)
    return str(value)


def sample_table(run_dirs, split):
    samples = {}
    for label, path in run_dirs.items():
        fpath = os.path.join(path, "samples20_{}.json".format(split))
        if os.path.exists(fpath):
            samples[label] = {r["qa_id"]: r for r in load_json(fpath)}
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        metavar="LABEL=DIR")
    parser.add_argument("--baseline", default="C")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="reports/ablation_c_d_e")
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        label, _, path = item.partition("=")
        runs[label] = path
    summaries = {label: run_summary(path, args.split)
                 for label, path in runs.items()}

    base = summaries.get(args.baseline, {"status": "not run"})
    report = {"split": args.split, "baseline": args.baseline,
              "runs": summaries, "success_criteria": {}}
    for label, summary in summaries.items():
        if label == args.baseline or summary["status"] != "ok":
            continue
        checks, count = success_criteria(base, summary)
        report["success_criteria"][label] = {
            "checks": checks, "passed": count, "of": len(checks),
            "verdict": count >= 2}

    samples = sample_table(runs, args.split)
    lines = ["# Ablation C/D/E/oracle/relative-geometry (split {})".format(
        args.split), "",
        "Baseline for comparisons: {} ({}).".format(
            args.baseline, runs[args.baseline]), ""]

    header = ["run", "format", "L/R", "F/B", "exact", "hh", "hl", "ll",
              "dist MAE", "dist med", "pearson", "std ratio", "spread ratio",
              "collapse?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for label, summary in summaries.items():
        if summary["status"] != "ok":
            lines.append("| {} | not run | | | | | | | | | | | | |".format(label))
            continue
        diag = summary["distance_diag"]
        row = [label, fmt(summary["format_validity"], 3),
               fmt(summary["left_right"]), fmt(summary["front_behind"]),
               fmt(summary["exact"]),
               fmt(summary["buckets"].get("high_high")),
               fmt(summary["buckets"].get("high_low")),
               fmt(summary["buckets"].get("low_low")),
               fmt(summary["distance_mae"], 3),
               fmt(summary["distance_median"], 3),
               fmt(diag.get("pearson"), 3), fmt(diag.get("std_ratio"), 3),
               fmt(diag.get("bin_spread_ratio"), 3),
               fmt(bool(diag.get("collapse_suspect")))]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Distance by GT bin (predicted mean / MAE)", ""]
    lines.append("| run | " + " | ".join(BIN_LABELS) + " |")
    lines.append("|" + "---|" * (len(BIN_LABELS) + 1))
    for label, summary in summaries.items():
        if summary["status"] != "ok":
            continue
        diag = summary["distance_diag"]
        cells = []
        for bname in BIN_LABELS:
            cell = diag.get("bins", {}).get(bname, {})
            if cell.get("n"):
                cells.append("{} / {} (n={})".format(
                    fmt(cell["pred_mean"], 2), fmt(cell["mae"], 2),
                    cell["n"]))
            else:
                cells.append("-")
        lines.append("| " + " | ".join([label] + cells) + " |")

    lines += ["", "## D success criteria vs {}".format(args.baseline), "",
              "Any two of: F/B +5pp, exact +5pp, distance MAE -20%, "
              "Pearson >= 0.5, bin means monotonic and spread ratio >= "
              "{:.1f}.".format(SPREAD_RATIO_MIN), ""]
    for label, info in report["success_criteria"].items():
        flags = ", ".join("{}={}".format(k, fmt(v))
                          for k, v in info["checks"].items())
        lines.append("- {}: passed {}/{} -> {} | {}".format(
            label, info["passed"], info["of"],
            "SUCCESS" if info["verdict"] else "not yet", flags))

    if samples:
        lines += ["", "## 20 identical {} samples (baseline = {})".format(
            args.split, args.baseline), ""]
        base_samples = samples.get(args.baseline, {})
        other_labels = [l for l in samples if l != args.baseline]
        for qa_id, base_row in base_samples.items():
            lines.append("### {}".format(base_row["question"]))
            lines.append("")
            lines.append("- {} GT: {} {} / {} m | pred: {} {} / {} m".format(
                args.baseline, base_row["gt_lateral"],
                base_row["gt_longitudinal"], base_row["gt_distance"],
                base_row["pred_lateral"], base_row["pred_longitudinal"],
                base_row["pred_distance"]))
            for label in other_labels:
                row = samples[label].get(qa_id)
                if row is None:
                    continue
                changed = row["prediction"] != base_row["prediction"]
                lines.append("- {}{}: {} {} / {} m | \"{}\"".format(
                    label, " [changed]" if changed else " [same]",
                    row["pred_lateral"], row["pred_longitudinal"],
                    row["pred_distance"], row["prediction"][:160]))
            lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines) + "\n")
    with open(args.out + ".json", "w", encoding="utf-8") as fobj:
        json.dump(report, fobj, indent=1, ensure_ascii=False)
    print("wrote {}.md and {}.json".format(args.out, args.out))
    for label, summary in summaries.items():
        if summary["status"] == "ok":
            diag = summary["distance_diag"]
            print("{}: exact {} | FB {} | MAE {} | pearson {} | "
                  "pred std {} | collapse {}".format(
                      label, summary["exact"], summary["front_behind"],
                      summary["distance_mae"], diag.get("pearson"),
                      diag.get("pred_std"), diag.get("collapse_suspect")))
        else:
            print("{}: not run".format(label))


if __name__ == "__main__":
    main()
