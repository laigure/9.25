"""Check multi-target caption/box alignment and a real point-cloud sample."""

import argparse
import json

import numpy as np

from src.joint_det_dataset import Joint3DDataset
from utils.transform_waymo import transform_to_front_view


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="waymo-multi",
                    choices=["waymo-multi", "waymo-others-multi"])
args = parser.parse_args()
splits = ("train", "val") if args.dataset == "waymo-others-multi" else ("train", "val", "test")

for split in splits:
    dataset = Joint3DDataset(
        dataset_dict={args.dataset: 1},
        test_dataset={args.dataset: 1},
        split=split,
        data_path="data/",
    )
    assert len(dataset) > 0
    for annotation in dataset.annos:
        count = len(annotation["boxes_info"]["bbox3d"])
        assert count == 2 if args.dataset == "waymo-others-multi" else count in (2, 3)
        assert annotation["pred_pos_map"].shape == (count, 256)
        assert (annotation["pred_pos_map"].sum(axis=1) > 0).all()
        overlap = (annotation["pred_pos_map"] > 0).astype(np.int32)
        overlap = overlap @ overlap.T
        assert np.count_nonzero(overlap - np.diag(np.diag(overlap))) == 0
        assert len(json.loads(annotation["object_ids_json"])) == count
        centers = np.stack(annotation["boxes_info"]["bbox3d"])[:, :3]
        assert np.unique(centers, axis=0).shape[0] == count
    item = dataset[0]
    item_count = int(item["box_label_mask"].sum())
    assert item_count == 2 if args.dataset == "waymo-others-multi" else item_count in (2, 3)
    assert item["coordinate_frame"] == "source_lidar"
    instance_ids = set(item["point_instance_label"].astype(int).tolist()) - {-1}
    assert instance_ids <= set(range(item_count))
    if split != "train":
        repeated = dataset[0]
        np.testing.assert_array_equal(item["point_clouds"], repeated["point_clouds"])
        np.testing.assert_array_equal(
            item["point_instance_label"], repeated["point_instance_label"]
        )
    probe_ids = set()
    for probe_index in range(min(64, len(dataset))):
        probe = dataset[probe_index]
        probe_ids.update(
            set(probe["point_instance_label"].astype(int).tolist()) - {-1}
        )
        if any(instance_id > 0 for instance_id in probe_ids):
            break
    assert any(instance_id > 0 for instance_id in probe_ids), (
        "The multi-object point labels collapsed to target 0 across the probe; "
        f"observed ids: {probe_ids}"
    )

    # The reconstructed set contains all five LiDAR views. Verify that a side
    # view rotates every GT box to the same canonical front-view frame used by
    # the released single-object loader.
    side_index = next(
        i for i, annotation in enumerate(dataset.annos)
        if not annotation["scan_id"].endswith("_0")
    )
    annotation = dataset.annos[side_index]
    raw_boxes = np.stack(annotation["boxes_info"]["bbox3d"]).astype(np.float32)
    view_id = int(annotation["scan_id"].split("_")[-1])
    _, expected_boxes = transform_to_front_view(
        np.zeros((1, 3), dtype=np.float32),
        raw_boxes,
        ["F", "FL", "FR", "SL", "SR"][view_id],
    )
    side_item = dataset[side_index]
    count = int(side_item["box_label_mask"].sum())
    np.testing.assert_allclose(
        side_item["gt_bboxes"][:count], expected_boxes, rtol=1e-6, atol=1e-6
    )
    print(split, len(dataset), item["point_clouds"].shape,
          item_count, "probe_instance_ids", sorted(probe_ids),
          "side_view_rotation_ok", "deterministic_eval_ok", flush=True)
