"""Export ordered language-matched final decoder queries from 3EED evaluation."""

import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def export_selected_tokens(end_points, output_dir, batch_idx, rank, checkpoint_path):
    """Select one distinct candidate per referent using soft-token scores.

    This is predicted top-1 selection using the dataset's annotated target
    text span, never oracle GT-box matching. A deployed system needs a text
    span predictor or a different query-ranking policy. The query feature
    and predicted box share the same query index.
    """
    query_features = end_points["last_query_features"]
    query_projection = end_points["last_proj_queries"]
    text_projection = end_points["proj_tokens"]
    positive_map = end_points["positive_map"]
    box_counts = end_points["box_label_mask"].sum(-1)
    if not torch.all(box_counts >= 1):
        raise ValueError("Every caption needs at least one referred object")
    if query_features.shape[:2] != query_projection.shape[:2]:
        raise ValueError("query feature and projection counts differ")
    if query_features.shape[-1] != 288:
        raise ValueError(f"expected 288-D query, got {query_features.shape[-1]}")

    text_length = text_projection.shape[1]
    max_targets = int(box_counts.max().item())
    target_span = (positive_map[:, :max_targets, :text_length] > 0).to(text_projection.dtype)
    valid = torch.arange(max_targets, device=box_counts.device)[None] < box_counts[:, None]
    if not torch.all(target_span.sum(-1)[valid] > 0):
        raise ValueError("missing positive language span for a target")
    target_span = target_span / target_span.sum(-1, keepdim=True).clamp_min(1)

    # The span is determined from the input caption, never from a GT box. Save
    # both language-query heads so validation can choose the stronger selector
    # without another model forward pass.
    soft_scores = end_points["last_sem_cls_scores"].softmax(-1)
    score_length = min(text_length, soft_scores.shape[-1])
    soft_target_scores = torch.einsum(
        "bqt,bmt->bmq",
        soft_scores[:, :, :score_length],
        target_span[:, :, :score_length],
    )
    contrastive_probs = torch.matmul(
        query_projection, text_projection.transpose(-1, -2)
    ).div(0.07).softmax(-1)
    contrastive_target_scores = torch.einsum(
        "bqt,bmt->bmq", contrastive_probs, target_span
    )

    def unique_assignment(scores):
        result = torch.full(
            (scores.shape[0], max_targets), -1,
            device=scores.device, dtype=torch.long,
        )
        for bid, count in enumerate(box_counts.int().tolist()):
            row_ids, query_ids = linear_sum_assignment(-scores[bid, :count].cpu().numpy())
            result[bid, torch.as_tensor(row_ids, device=scores.device)] = torch.as_tensor(
                query_ids, device=scores.device
            )
        return result

    indices = unique_assignment(soft_target_scores)
    contrastive_indices = unique_assignment(contrastive_target_scores)
    rows = torch.arange(indices.shape[0], device=indices.device)[:, None]
    targets = torch.arange(max_targets, device=indices.device)[None, :]
    safe_indices = indices.clamp_min(0)
    safe_contrastive_indices = contrastive_indices.clamp_min(0)
    selected_scores = soft_target_scores[rows, targets, safe_indices].masked_fill(~valid, float("nan"))
    selected_tokens = query_features[rows, safe_indices].masked_fill(~valid[..., None], float("nan"))
    selected_center = end_points["last_center"][rows, safe_indices].masked_fill(~valid[..., None], float("nan"))
    selected_size = end_points["last_pred_size"][rows, safe_indices].masked_fill(~valid[..., None], float("nan"))
    contrastive_selected_scores = contrastive_target_scores[
        rows, targets, safe_contrastive_indices
    ].masked_fill(~valid, float("nan"))
    contrastive_selected_tokens = query_features[
        rows, safe_contrastive_indices
    ].masked_fill(~valid[..., None], float("nan"))
    contrastive_selected_center = end_points["last_center"][
        rows, safe_contrastive_indices
    ].masked_fill(~valid[..., None], float("nan"))
    contrastive_selected_size = end_points["last_pred_size"][
        rows, safe_contrastive_indices
    ].masked_fill(~valid[..., None], float("nan"))
    is_multi = max_targets > 1
    def output_shape(value):
        return value if is_multi else value[:, 0]

    rank_dir = os.path.join(output_dir, f"rank{rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)
    path = os.path.join(rank_dir, f"batch{batch_idx:06d}.npz")
    np.savez_compressed(
        path,
        feature_name="last_query_features",
        selection_policy=("soft_token_unique_assignment_normalized_text_spans" if is_multi
                          else "soft_token_top1_text_span"),
        checkpoint_path=str(checkpoint_path),
        target_count=box_counts.cpu().numpy().astype(np.int32),
        query_index=output_shape(indices).cpu().numpy().astype(np.int32),
        match_score=output_shape(selected_scores).cpu().numpy().astype(np.float32),
        object_token=output_shape(selected_tokens).cpu().numpy().astype(np.float32),
        pred_center=output_shape(selected_center).cpu().numpy().astype(np.float32),
        pred_size=output_shape(selected_size).cpu().numpy().astype(np.float32),
        contrastive_selection_policy=("contrastive_unique_assignment_normalized_text_spans" if is_multi
                                      else "contrastive_top1_text_span"),
        contrastive_query_index=output_shape(contrastive_indices).cpu().numpy().astype(np.int32),
        contrastive_match_score=output_shape(contrastive_selected_scores).cpu().numpy().astype(np.float32),
        contrastive_object_token=output_shape(contrastive_selected_tokens).cpu().numpy().astype(np.float32),
        contrastive_pred_center=output_shape(contrastive_selected_center).cpu().numpy().astype(np.float32),
        contrastive_pred_size=output_shape(contrastive_selected_size).cpu().numpy().astype(np.float32),
        gt_box=output_shape(end_points["gt_bboxes"][:, :max_targets]).cpu().numpy().astype(np.float32),
        meta_path=np.asarray(end_points["meta_path"], dtype=str),
        utterance=np.asarray(end_points["utterances"], dtype=str),
        object_ids_json=np.asarray(end_points.get("object_ids_json", ["[]"] * len(box_counts)), dtype=str),
    )
    return path
