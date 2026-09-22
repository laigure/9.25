#!/usr/bin/env python3
"""Validate the published datasets, token exports, models, and result summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    expected = {
        "artifacts/remote/qa_correct_others_v4_contrastive/joint/qa.jsonl": 4225,
        "artifacts/remote/qa_correct_others_v4_contrastive/swap_eval/qa.jsonl": 615,
        "artifacts/remote/qa_all_others_v4_contrastive/joint/qa.jsonl": 1638,
        "artifacts/remote/qa_all_others_v4_contrastive/swap_eval/qa.jsonl": 1638,
    }
    for relative, wanted in expected.items():
        path = ROOT / relative
        assert path.is_file(), f"missing {relative}"
        actual = count_jsonl(path)
        assert actual == wanted, f"{relative}: expected {wanted}, got {actual}"

    token_root = ROOT / "artifacts/remote/others_linked_tokens"
    split_expected = {"train": 1137, "val": 1300}
    for split, wanted in split_expected.items():
        samples = 0
        files = sorted((token_root / split / "rank00").glob("batch*.npz"))
        assert files, f"no token files for {split}"
        for path in files:
            with np.load(path, allow_pickle=True) as data:
                token = data["contrastive_object_token"]
                count = data["target_count"]
                assert token.ndim == 3 and token.shape[1:] == (2, 288), path
                assert np.all(count == 2), path
                assert data["contrastive_query_index"].shape == token.shape[:2], path
                samples += token.shape[0]
        assert samples == wanted, f"{split}: expected {wanted}, got {samples}"

    for relative in [
        "artifacts/models/relation_projector/best.pt",
        "artifacts/models/relation_lora/best.pt",
        "artifacts/results/others_v4_end_to_end/analysis.json",
        "artifacts/results/grounding_metrics.json",
    ]:
        assert (ROOT / relative).is_file(), f"missing {relative}"

    analysis = json.loads(
        (ROOT / "artifacts/results/others_v4_end_to_end/analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["qa_rows"] == 1638
    assert abs(analysis["end_to_end"]["accuracy"] - 0.5708180708180708) < 1e-12

    scene_summary = json.loads((
        ROOT / "artifacts/local_qa/scene_multi_data/summary.json"
    ).read_text(encoding="utf-8"))
    assert scene_summary["max_target_count"] == 5
    assert scene_summary["splits"]["train"]["records"] == 2701
    assert scene_summary["splits"]["val"]["records"] == 2708
    assert scene_summary["counters"]["targets_total"] == 7481
    scene_qa = ROOT / "artifacts/local_qa/scene_reasoning_qa_v1/qa.jsonl"
    assert count_jsonl(scene_qa) == 5518
    scene_qa_summary = json.loads((scene_qa.parent / "summary.json").read_text(
        encoding="utf-8"))
    assert scene_qa_summary["all_answers_one_sentence"] is True
    oracle = json.loads((scene_qa.parent / "oracle_eval.json").read_text(
        encoding="utf-8"))
    assert oracle["strict_language_all_pass"] == 1.0
    print("release validation passed")


if __name__ == "__main__":
    main()
