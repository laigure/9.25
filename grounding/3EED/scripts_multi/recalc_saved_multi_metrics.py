"""Recalculate per-target and joint metrics from saved multi predictions."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("predictions", type=Path)
args = parser.parse_args()
records = [json.loads(path.read_text()) for path in args.predictions.rglob("prediction.json")]
if not records:
    raise SystemExit(f"no prediction.json under {args.predictions}")
ious = [float(value) for record in records for value in record["ious"]]
result = {
    "samples": len(records),
    "targets": len(ious),
    "per_target_acc025": sum(value > 0.25 for value in ious) / len(ious),
    "per_target_acc05": sum(value > 0.5 for value in ious) / len(ious),
    "joint_acc025": sum(all(value > 0.25 for value in record["ious"]) for record in records) / len(records),
    "joint_acc05": sum(all(value > 0.5 for value in record["ious"]) for record in records) / len(records),
    "miou": sum(ious) / len(ious),
}
print(json.dumps(result, indent=2))
