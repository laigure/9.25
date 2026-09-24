"""Small synthetic checks for the native-caption natural QA contract."""

import json
import tempfile
import unittest
from pathlib import Path

from audit_native_others import category_mentions, secondary_entity_mentions
from build_native_natural_qa import audit, build_record, write_jsonl
from strict_natural_qa_eval import evaluate


def synthetic_record(split="val"):
    scene = f"waymo/seq_{split}/0001_0#g0"
    descriptions = ["silver sedan", "yellow excavator", "cement mixer truck"]
    boxes = [[10.0, 4.0, 0.0, 4.0, 2.0, 2.0, 0.0],
             [20.0, -3.0, 0.0, 5.0, 2.0, 3.0, 0.0],
             [6.0, -8.0, 0.0, 8.0, 3.0, 3.0, 0.0]]
    row = {
        "scene_id": scene, "source_scene_id": scene.split("#")[0],
        "segment_name": f"seq_{split}", "frame_name": "0001_0",
        "target_count": 3,
        "object_ids": [f"{scene}:obj{i}" for i in range(3)],
        "object_descriptions": descriptions,
        "object_views": ["front view"] * 3,
        "object_classes": ["car", "othervehicle", "truck"],
    }
    for index, box in enumerate(boxes, 1):
        row[f"bbox3d_obj_{index}"] = box
    return row


class NativeNaturalQATest(unittest.TestCase):
    def test_secondary_mentions_exclude_main_coreference(self):
        caption = ("The pedestrian is seen. A silver car is behind and to the "
                   "right of the pedestrian, and an orange guardrail is behind "
                   "the pedestrian.")
        main = category_mentions(caption, "pedestrian")[0]
        self.assertEqual(secondary_entity_mentions(caption, "pedestrian", main), [])

        caption = ("The pedestrian is seen, with the side facing the observer "
                   "and a white postal vehicle in front.")
        main = category_mentions(caption, "pedestrian")[0]
        self.assertEqual([item[2] for item in secondary_entity_mentions(
            caption, "car", main)], ["vehicle"])

    def test_oracle_passes_strict_contract_without_role_aliases(self):
        rows, private = [], {}
        build_record(synthetic_record(), "val", rows, private, 5.0)
        report = audit(rows, private)
        self.assertEqual(report["counts"].get("object_alias_leaks", 0), 0)
        predictions = [{"qa_id": row["qa_id"], "prediction": row["answer"]}
                       for row in rows]
        result = evaluate(rows, predictions, list(private.values()))
        self.assertEqual(result["strict_svo_accuracy"], 1.0)
        self.assertEqual(result["canonical_sentence_match_rate"], 1.0)
        self.assertEqual(result["strict_all_pass"], 1.0)
        self.assertEqual(result["inverse_relation_consistency"], 1.0)

    def test_jsonl_round_trip(self):
        rows, private = [], {}
        build_record(synthetic_record(), "val", rows, private, 5.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.jsonl"
            write_jsonl(path, rows)
            loaded = [json.loads(line) for line in path.read_text(
                encoding="utf-8").splitlines()]
        self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
