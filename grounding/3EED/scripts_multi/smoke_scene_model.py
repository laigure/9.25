"""Run one variable-target 512-token forward/backward step without saving."""

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from src.joint_det_dataset import Joint3DDataset
from train_dist_mod import TrainTester


def main():
    dataset = Joint3DDataset(
        dataset_dict={"waymo-scene-multi": 1},
        test_dataset={"waymo-scene-multi": 1}, split="val",
        data_path="data/", use_color=True, text_token_budget=512,
    )
    index = next(i for i, anno in enumerate(dataset.annos)
                 if len(anno["boxes_info"]["bbox3d"]) >= 4)
    batch = next(iter(DataLoader(Subset(dataset, [index]), batch_size=1,
                                 num_workers=0)))
    args = SimpleNamespace(
        use_color=True, use_height=False, use_multiview=False,
        use_soft_token_loss=True, text_token_budget=512,
        num_target=256, num_decoder_layers=6,
        self_position_embedding="loc_learned", use_contrastive_align=True,
        butd=False, butd_gt=False, butd_cls=False, pp_checkpoint=None,
        self_attend=True, query_points_obj_topk=4,
    )
    model = TrainTester.get_model(args).cuda().train()
    batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    end_points = model({"point_clouds": batch["point_clouds"].float(),
                        "text": batch["utterances"]},
                       return_query_features=True)
    for key, value in batch.items():
        end_points[key] = value
    matcher = HungarianMatcher(1, 0, 2, True)
    criterion = SetCriterion(matcher=matcher,
                             losses=["boxes", "labels", "contrastive_align"],
                             eos_coef=0.1, temperature=0.07).cuda()
    loss, end_points = compute_hungarian_loss(
        end_points, args.num_decoder_layers, criterion,
        query_points_obj_topk=args.query_points_obj_topk)
    loss.backward()
    target_count = int(batch["box_label_mask"].sum().item())
    assert target_count >= 4
    assert end_points["positive_map"].shape[-1] == 512
    assert end_points["last_sem_cls_scores"].shape[-1] == 512
    assert end_points["last_query_features"].shape[-1] == 288
    print({"loss": float(loss.detach().cpu()), "target_count": target_count,
           "positive_map": tuple(end_points["positive_map"].shape),
           "sem_scores": tuple(end_points["last_sem_cls_scores"].shape),
           "query_features": tuple(end_points["last_query_features"].shape),
           "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)})


if __name__ == "__main__":
    main()
