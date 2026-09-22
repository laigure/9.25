"""Measure per-target point coverage before and after 16k point sampling."""

import argparse
import json

import numpy as np
import torch

from src.joint_det_dataset import Joint3DDataset
from ops.teed_pointnet.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="waymo-multi",
                    choices=["waymo-multi", "waymo-others-multi"])
args = parser.parse_args()
np.random.seed(0)
report = {}
splits = ("train", "val") if args.dataset == "waymo-others-multi" else ("train", "val", "test")
for split in splits:
    dataset = Joint3DDataset(
        dataset_dict={args.dataset: 1},
        test_dataset={args.dataset: 1},
        split=split,
        data_path="data/",
    )
    records = min(128, len(dataset))
    targets = sampled_covered = raw_covered = all_sampled = all_raw = 0
    sampled_counts = []
    raw_counts = []
    for index in range(records):
        item = dataset[index]
        anno = dataset.annos[index]
        count = int(item["box_label_mask"].sum())
        labels = set(item["point_instance_label"].astype(int).tolist()) - {-1}
        sampled_per_target = [int((item["point_instance_label"] == i).sum()) for i in range(count)]

        pcd_path = anno["pcd_path"]
        raw = (np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
               if pcd_path.endswith(".bin") else np.load(pcd_path))
        boxes = np.stack(anno["boxes_info"]["bbox3d"], axis=0).astype(np.float32)
        raw_membership = points_in_boxes_cpu(
            torch.from_numpy(raw[:, :3]), torch.from_numpy(boxes)
        ).numpy()
        raw_per_target = [int((raw_membership[i] > 0).sum()) for i in range(count)]

        targets += count
        sampled_covered += sum(value > 0 for value in sampled_per_target)
        raw_covered += sum(value > 0 for value in raw_per_target)
        all_sampled += int(all(value > 0 for value in sampled_per_target))
        all_raw += int(all(value > 0 for value in raw_per_target))
        sampled_counts.extend(sampled_per_target)
        raw_counts.extend(raw_per_target)
    report[split] = {
        "records": records,
        "targets": targets,
        "targets_with_raw_points": raw_covered,
        "raw_target_coverage": raw_covered / targets,
        "targets_with_sampled_points": sampled_covered,
        "sampled_target_coverage": sampled_covered / targets,
        "visible_targets_lost_by_sampling": sum(
            raw_value > 0 and sampled_value == 0
            for raw_value, sampled_value in zip(raw_counts, sampled_counts)
        ),
        "records_all_targets_raw_covered": all_raw,
        "records_all_targets_sampled_covered": all_sampled,
        "raw_points_per_target_median": float(np.median(raw_counts)),
        "sampled_points_per_target_median": float(np.median(sampled_counts)),
    }

print(json.dumps(report, indent=2))
