"""Select a retained native N=2/3 Grounding checkpoint from its train log."""

import argparse
import json
import re
from pathlib import Path


def validation_metrics(log_text):
    text = log_text.replace("\r", "\n")
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    marks = list(re.finditer(r"Eval epoch (\d+): 100%", text))
    results = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        block = text[mark.start():end]
        overall = re.search(r"JointAll-bbf@0\.25 = ([0-9.]+)", block)
        groups = {int(count): float(score) for count, score in re.findall(
            r"bbf targets=(\d+): n=\d+\s+JointAcc@0\.25=([0-9.]+)", block)}
        if overall and all(count in groups for count in (2, 3)):
            epoch = int(mark.group(1))
            results[epoch] = {
                "joint_all_bbf_025": float(overall.group(1)),
                "joint_by_target_count_025": groups,
                "macro_n2_n3_025": (groups[2] + groups[3]) / 2.0,
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--selection-out", type=Path, required=True)
    args = parser.parse_args()
    metrics = validation_metrics(args.log.read_text(errors="ignore"))
    candidates = []
    for path in args.run_dir.glob("ckpt_epoch_*.pth"):
        match = re.fullmatch(r"ckpt_epoch_(\d+)\.pth", path.name)
        if match and int(match.group(1)) in metrics:
            candidates.append((int(match.group(1)), path))
    last = args.run_dir / "ckpt_epoch_last.pth"
    if last.exists() and metrics:
        candidates.append((max(metrics), last))
    if not candidates:
        raise SystemExit("no retained native checkpoint has matching validation metrics")
    epoch, checkpoint = max(
        candidates,
        key=lambda item: (metrics[item[0]]["macro_n2_n3_025"],
                          metrics[item[0]]["joint_all_bbf_025"]))
    result = {
        "selected_checkpoint": str(checkpoint),
        "selected_epoch": epoch,
        "selection_metric": "macro JointAll-bbf Acc@0.25 over native N=2,3",
        "selected_metrics": metrics[epoch],
        "candidate_metrics": {str(candidate_epoch): metrics[candidate_epoch]
                              for candidate_epoch, _ in sorted(candidates)},
    }
    args.selection_out.parent.mkdir(parents=True, exist_ok=True)
    args.selection_out.write_text(json.dumps(result, indent=2) + "\n")
    print(checkpoint)


if __name__ == "__main__":
    main()
