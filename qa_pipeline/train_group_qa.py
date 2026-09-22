"""Train/evaluate Qwen2.5-7B LoRA scenario QA with rich-export group features.

Same question set, projector design, LoRA settings, trainer and evaluator as
train_scenario_qa.py; the only difference is the visual input. Every referenced
object contributes one soft token whose 288*d vector is read from the rich
feature export (qa_features_v1) according to the T-group spec of
screen_token_groups.py. Features are used raw (no standardization), matching
the v3 token convention. GT geometry is never read by the model input; it is
used only by the offline diagnostics (answer keys, grounding grouping).

Usage:
  python train_group_qa.py --group T1 --mode lora --out-dir <run>
  python train_group_qa.py --group T1 --mode eval --init-projector <run>/best.pt \
      --out-dir <run> --eval-splits val,test
  add --swap          eval probe: slot A gets role B's block and vice versa
  add --shuffle-seed 42   eval probe: features of a same (split, category) donor
  add --only-multi-ref    probe scope: rows referencing >=2 objects only
"""

import argparse
import collections
import json
import os

import numpy as np
import torch

from screen_token_groups import (GROUPS, MAX_ROLES, FeatureStore, build_visual,
                                 load_feature_records)
from train_qa import (ASSISTANT_PREFIX, ASSISTANT_SUFFIX, PROMPT_PREFIX, QATrainer,
                      TokenProjector)
from train_scenario_qa import evaluate, read_jsonl, select_eval

RELATION_SCENARIOS = ("which_is_left", "closer_to_ego", "closer_of_two",
                      "closest_of_three_to_ego", "relative_location")


def build_donor_map(features_dir, seed):
    """object_id -> donor object_id from the same (split, category), no fixed points."""
    rows = read_jsonl(os.path.join(features_dir, "records.jsonl"))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["split"], row["category"])].append(row["object_id"])
    rng = np.random.default_rng(seed)
    donor = {}
    for ids in groups.values():
        if len(ids) < 2:
            continue
        order = np.asarray(sorted(ids))
        shuffled = order.copy()
        rng.shuffle(shuffled)
        for index in np.flatnonzero(shuffled == order):
            other = (index + 1) % len(shuffled)
            shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
        assert not np.any(shuffled == order)
        donor.update(zip(order.tolist(), shuffled.tolist()))
    return donor


class GroupTrainer(QATrainer):
    def __init__(self, args):
        self.specs = GROUPS[args.group]
        assert self.specs, "T0 has no visual input; screen it only with screen_token_groups.py"
        self.block_dim = 288 * len(self.specs)
        # The base class builds a 288-dim projector and loads --init-projector
        # into it before we can resize; for wider groups skip that load and
        # reload the checkpoint after the rebuild, otherwise the trained
        # projector would be replaced by a fresh random one.
        deferred_load = args.init_projector if self.block_dim != 288 else None
        if deferred_load:
            args.init_projector = None
        super().__init__(args)
        args.init_projector = deferred_load
        self.store = FeatureStore(args.features, load_feature_records(args.features))
        if self.block_dim != self.projector.net[0].in_features:
            self.projector = TokenProjector(self.block_dim, 1024, self.hidden_size).to(self.device)
            print("projector rebuilt | in_dim", self.block_dim, "-> 1024 ->",
                  self.hidden_size, flush=True)
        if deferred_load:
            self.load_checkpoint(deferred_load)
        self.swap = bool(getattr(args, "swap", False))
        self.donor = None
        self._visual_cache = {}

    def set_donor(self, donor):
        self.donor = donor
        self._visual_cache.clear()

    def slots(self, record):
        """Objects in role-slot order after the probe transforms."""
        refs = sorted(record["object_refs"], key=lambda ref: ref["role"])
        if self.swap and len(refs) > 1:
            refs = [refs[1], refs[0]] + refs[2:]
        if self.donor is not None:
            refs = [dict(ref, object_id=self.donor.get(ref["object_id"], ref["object_id"]))
                    for ref in refs]
        return refs

    def visual_for(self, record):
        qa_id = record["qa_id"]
        if qa_id not in self._visual_cache:
            flat = build_visual(self.specs, self.slots(record), self.store)
            assert flat.shape[0] == MAX_ROLES * self.block_dim, qa_id
            self._visual_cache[qa_id] = flat
        return self._visual_cache[qa_id]

    def check_inputs(self, records):
        for record in records:
            for ref in record["object_refs"]:
                if ref["object_id"] not in self.store.index:
                    raise KeyError("object missing from rich export: " + ref["object_id"])
            flat = self.visual_for(record)
            if not np.isfinite(flat).all():
                raise ValueError("non-finite visual input: " + record["qa_id"])
        return len(records)

    def pieces(self, record, with_answer):
        # This is the complete model input. No GT or predicted coordinate,
        # bounding box, distance, or relation field is read here.
        pieces = [("text", self.encode(PROMPT_PREFIX), False),
                  ("text", self.encode(record["question"]), False)]
        flat = torch.as_tensor(self.visual_for(record), dtype=torch.float32,
                               device=self.device)
        for slot, obj in enumerate(sorted(record["object_refs"], key=lambda r: r["role"])):
            vector = flat[slot * self.block_dim:(slot + 1) * self.block_dim]
            soft = self.projector(vector).to(self.embed_dtype).reshape(1, 1, -1)
            pieces += [("text", self.encode("\nObject {} token: ".format(obj["role"])), False),
                       ("soft", soft, False)]
        pieces.append(("text", self.encode(ASSISTANT_PREFIX), False))
        if with_answer:
            pieces += [("text", self.encode(record["answer"]), True),
                       ("text", self.encode(ASSISTANT_SUFFIX), True)]
        return pieces

    def min_top1_iou(self, record):
        return min(self.store.top1_iou_last(ref["object_id"]) for ref in record["object_refs"])


def relation_summary(by_scenario):
    total, correct = 0, 0.0
    for scenario in RELATION_SCENARIOS:
        entry = by_scenario.get(scenario)
        if not entry:
            continue
        total += entry["n"]
        correct += entry["accuracy"] * entry["n"]
    return {"n": total, "accuracy": round(correct / total, 5) if total else None}


def evaluate_with_diagnostics(trainer, records, predictions, private):
    result = evaluate(records, predictions, private)
    ious = np.asarray([trainer.min_top1_iou(r) for r in records], dtype=np.float32)
    grounded = ious >= 0.25
    correct = np.asarray([row["correct"] for row in result["rows"]], dtype=bool)
    result["grounding"] = {
        "all_refs_top1_iou_ge_025": round(float(grounded.mean()), 5),
        "acc_when_grounded": round(float(correct[grounded].mean()), 5) if grounded.any() else None,
        "acc_when_missed": round(float(correct[~grounded].mean()), 5) if (~grounded).any() else None,
    }
    result["relation"] = relation_summary(result["by_scenario"])
    for row, iou in zip(result["rows"], ious):
        row["min_top1_iou"] = round(float(iou), 4)
    return result


def mirror_key(key):
    if key is None:
        return None
    return key.translate(str.maketrans({"A": "B", "B": "A"}))


def swap_summary(records, normal, swapped):
    """Metrics on multi-target rows: swap of the A/B object blocks.

    flip_rate: swapped prediction mirrors the normal prediction (order-driven).
    content_rate: swapped prediction mirrors the GT key (feature-content-driven).
    """
    by_scenario = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for record, nrow, srow in zip(records, normal["rows"], swapped["rows"]):
        if len(record["object_refs"]) < 2:
            continue
        stats = by_scenario[record["scenario"]]
        stats[0] += 1
        stats[1] += nrow["correct"]
        stats[2] += srow["correct"]
        stats[3] += srow["pred_key"] is not None and srow["pred_key"] == mirror_key(nrow["pred_key"])
        stats[4] += srow["pred_key"] is not None and srow["pred_key"] == mirror_key(nrow["gt_key"])
    totals = [sum(v[i] for v in by_scenario.values()) for i in range(5)]
    n = totals[0]
    return {
        "n_applicable": n,
        "acc_normal": round(totals[1] / n, 5) if n else None,
        "acc_swapped": round(totals[2] / n, 5) if n else None,
        "flip_rate": round(totals[3] / n, 5) if n else None,
        "content_rate": round(totals[4] / n, 5) if n else None,
        "by_scenario": {scenario: {"n": v[0],
                                   "acc_normal": round(v[1] / v[0], 5),
                                   "acc_swapped": round(v[2] / v[0], 5),
                                   "flip_rate": round(v[3] / v[0], 5),
                                   "content_rate": round(v[4] / v[0], 5)}
                        for scenario, v in sorted(by_scenario.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/3eed_data/qa_v3/scenario_qa/qa.jsonl")
    ap.add_argument("--private-gt", default=None)
    ap.add_argument("--features", default="/root/autodl-tmp/3eed_data/qa_features_v1")
    ap.add_argument("--group", required=True,
                    choices=[name for name in sorted(GROUPS) if name != "T0"])
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["lora", "eval"], default="lora")
    ap.add_argument("--init-projector", default=None)
    ap.add_argument("--swap", action="store_true",
                    help="eval probe: slot A gets role B's block and vice versa")
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="eval probe: features of a same (split, category) donor object")
    ap.add_argument("--only-multi-ref", action="store_true",
                    help="eval probe: restrict to rows referencing >=2 objects")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gen-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--eval-per-scenario", type=int, default=50,
                    help="Deterministic cap per question type; 0 uses full split")
    ap.add_argument("--eval-splits", default="val,test")
    args = ap.parse_args()
    args.geom = "none"
    args.no_tokens = False
    args.pred_geometry = None
    args.oracle_center = False
    args.with_lora = True
    if args.mode == "eval" and not args.init_projector:
        ap.error("--mode eval requires --init-projector")
    args.private_gt = args.private_gt or os.path.join(os.path.dirname(args.data),
                                                      "private_gt.jsonl")
    args.tokens_root = os.path.dirname(os.path.dirname(args.data))
    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = read_jsonl(args.data)
    private = {x["qa_id"]: x for x in read_jsonl(args.private_gt)}
    assert {x["qa_id"] for x in all_rows} == set(private)
    split = {name: [r for r in all_rows if r["qa_split"] == name]
             for name in ("train", "val", "test")}
    seq = {name: {r["sequence"] for r in records} for name, records in split.items()}
    assert not (seq["train"] & seq["val"] or seq["train"] & seq["test"]
                or seq["val"] & seq["test"])

    trainer = GroupTrainer(args)
    if args.shuffle_seed is not None:
        trainer.set_donor(build_donor_map(args.features, args.shuffle_seed))
        changed = sum(1 for oid, donor in trainer.donor.items() if oid != donor)
        print("shuffle control: seed", args.shuffle_seed, "| donors for",
              changed, "objects", flush=True)
    checked = trainer.check_inputs(all_rows)
    print("group", args.group, "| specs", trainer.specs, "| block_dim", trainer.block_dim,
          "| checked", checked, "records", flush=True)

    if args.mode == "lora":
        train = split["train"][:args.max_train] if args.max_train else split["train"]
        val = select_eval(split["val"], args)
        print("TRAIN", len(train), "VAL", len(val), flush=True)
        trainer.train(train, val, args.epochs, log_every=args.log_every, out_dir=args.out_dir)
        trainer.load_checkpoint(os.path.join(args.out_dir, "best.pt"))

    for name in args.eval_splits.split(","):
        records = select_eval(split[name], args)
        if args.only_multi_ref:
            records = [r for r in records if len(r["object_refs"]) >= 2]
        print("GENERATE", name, len(records), flush=True)
        predictions = trainer.generate(records)
        result = evaluate_with_diagnostics(trainer, records, predictions, private)
        result["group"] = args.group
        result["probe"] = ("swap" if args.swap else
                           "shuffle" if args.shuffle_seed is not None else "normal")
        result["n_eval"] = len(records)
        result["only_multi_ref"] = args.only_multi_ref
        if args.swap:
            trainer.swap = True
            trainer._visual_cache.clear()
            swapped = evaluate_with_diagnostics(
                trainer, records, trainer.generate(records), private)
            trainer.swap = False
            trainer._visual_cache.clear()
            result["swap"] = swap_summary(records, result, swapped)
        suffix = ("_swap" if args.swap else
                  "_shuffle{}".format(args.shuffle_seed) if args.shuffle_seed is not None else "")
        with open(os.path.join(args.out_dir, "eval_{}{}.json".format(name, suffix)),
                  "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        summary = {k: v for k, v in result.items()
                   if k in ("n", "accuracy", "exact_sentence", "parse_rate",
                            "grounding", "relation", "swap")}
        print(name, suffix, json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
