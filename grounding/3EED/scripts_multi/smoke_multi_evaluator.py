"""Check that multi-object evaluator separates per-target and joint accuracy."""

import torch

from src.grounding_evaluator import GroundingEvaluator


evaluator = GroundingEvaluator(prefixes=["last_"])
batch = {
    "gt_bboxes": torch.tensor([[
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ]]),
    "meta_path": ["synthetic"],
    "utterances": ["Object A: car. Object B: car."],
    "class_ids": ["0,0"],
}
pred_bbox = torch.tensor([[
    [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],       # target A: correct
    [30.0, 0.0, 0.0, 2.0, 2.0, 2.0],      # target B: wrong
    [50.0, 0.0, 0.0, 2.0, 2.0, 2.0],
]])
scores = torch.tensor([
    [10.0, 0.0, -1.0],
    [0.0, 10.0, -1.0],
])

for mode in ("bbs", "bbf"):
    evaluator._evaluate_multi(batch, 0, "last_", mode, scores, pred_bbox)
    assert evaluator.dets[("last_", 0.25, 1, mode)] == 1
    assert evaluator.gts[("last_", 0.25, 1, mode)] == 2
    assert evaluator.dets[("multi_all", mode, 0.25, 1)] == 0
    assert evaluator.gts[("multi_all", mode, 0.25, 1)] == 1

assert evaluator.dets[("total_acc", 0.25, "bbf")] == 1
assert abs(evaluator.gts[("total_acc", 0.25, "bbf")] - 2) < 1e-10
print("multi evaluator denominator smoke OK")
