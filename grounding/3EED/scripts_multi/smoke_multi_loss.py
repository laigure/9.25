"""CPU checks for two-target Hungarian and point-instance supervision."""

import torch

from models.losses import (
    HungarianMatcher,
    SetCriterion,
    compute_points_obj_cls_loss_hard_topk,
)


# Two language spans and two boxes must match two distinct queries.
pred_logits = torch.zeros(1, 4, 8, requires_grad=True)
with torch.no_grad():
    pred_logits[0, 0, 1] = 10
    pred_logits[0, 1, 3] = 10
pred_boxes = torch.tensor(
    [[[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
      [8.0, 0.0, 0.0, 2.0, 2.0, 2.0],
      [30.0, 0.0, 0.0, 2.0, 2.0, 2.0],
      [40.0, 0.0, 0.0, 2.0, 2.0, 2.0]]],
    requires_grad=True,
)
positive_map = torch.zeros(2, 8)
positive_map[0, 1] = 1
positive_map[1, 3] = 1
targets = [{
    "labels": torch.zeros(2, dtype=torch.long),
    "boxes": pred_boxes.detach()[0, :2].clone(),
    "positive_map": positive_map,
}]
matcher = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=2, soft_token=True)
indices = matcher({"pred_logits": pred_logits, "pred_boxes": pred_boxes}, targets)
assert indices[0][0].tolist() == [0, 1]
assert indices[0][1].tolist() == [0, 1]
criterion = SetCriterion(matcher, losses=["boxes", "labels"])
losses, _ = criterion({"pred_logits": pred_logits, "pred_boxes": pred_boxes}, targets)
sum(losses.values()).backward()
assert torch.isfinite(sum(losses.values()))
assert pred_logits.grad is not None and pred_boxes.grad is not None


# Query-point supervision must accept point-instance ids 0 and 1 as two GTs.
seed_logits = torch.zeros(1, 1, 4, requires_grad=True)
point_end_points = {
    "box_label_mask": torch.tensor([[1.0, 1.0, 0.0]]),
    "seed_inds": torch.tensor([[0, 1, 2, 3]]),
    "seed_xyz": torch.tensor([[[0.0, 0.0, 0.0],
                                [8.0, 0.0, 0.0],
                                [20.0, 0.0, 0.0],
                                [30.0, 0.0, 0.0]]]),
    "seeds_obj_cls_logits": seed_logits,
    "center_label": torch.tensor([[[0.0, 0.0, 0.0],
                                    [8.0, 0.0, 0.0],
                                    [1000.0, 1000.0, 1000.0]]]),
    "size_gts": torch.ones(1, 3, 3) * 2,
    "point_instance_label": torch.tensor([[0, 1, -1, -1]]),
}
point_loss = compute_points_obj_cls_loss_hard_topk(point_end_points, topk=1)
point_loss.backward()
assert torch.isfinite(point_loss) and seed_logits.grad is not None

print("multi loss OK: two spans -> two boxes -> two queries; instance ids 0 and 1")
