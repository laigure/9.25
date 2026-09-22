"""Small CPU check that multi-target token export keeps distinct query indices."""

import tempfile
import numpy as np
import torch

from utils.qa_token_export import export_selected_tokens


torch.manual_seed(0)
batch, queries, tokens = 1, 4, 8
feature = torch.randn(batch, queries, 288)
projection = torch.zeros(batch, queries, 64)
language = torch.zeros(batch, tokens, 64)
projection[0, 0, 0] = 1
projection[0, 1, 1] = 1
language[0, 1, 0] = 1
language[0, 3, 1] = 1
positive = torch.zeros(batch, 132, 256)
positive[0, 0, 1] = 1
positive[0, 1, 3] = 1
soft_logits = torch.zeros(batch, queries, tokens)
soft_logits[0, 0, 1] = 10
soft_logits[0, 1, 3] = 10
mask = torch.zeros(batch, 132)
mask[0, :2] = 1
end_points = {
    "last_query_features": feature,
    "last_proj_queries": projection,
    "proj_tokens": language,
    "positive_map": positive,
    "last_sem_cls_scores": soft_logits,
    "box_label_mask": mask,
    "last_center": torch.randn(batch, queries, 3),
    "last_pred_size": torch.ones(batch, queries, 3),
    "gt_bboxes": torch.zeros(batch, 132, 7),
    "meta_path": ["test/scene/frame/image.jpg"],
    "utterances": ["Object A: car. Object B: van."],
    "object_ids_json": ['["objA", "objB"]'],
}
with tempfile.TemporaryDirectory() as directory:
    path = export_selected_tokens(end_points, directory, 0, 0, "synthetic")
    result = np.load(path)
    assert result["object_token"].shape == (1, 2, 288)
    assert result["query_index"].tolist() == [[0, 1]]
    assert result["contrastive_object_token"].shape == (1, 2, 288)
    assert result["contrastive_query_index"].tolist() == [[0, 1]]
    assert result["target_count"].tolist() == [2]
    assert str(result["selection_policy"]) == "soft_token_unique_assignment_normalized_text_spans"
    assert str(result["contrastive_selection_policy"]) == "contrastive_unique_assignment_normalized_text_spans"
    assert result["object_ids_json"].tolist() == ['["objA", "objB"]']
    print("multi token export OK", result["query_index"].tolist())
