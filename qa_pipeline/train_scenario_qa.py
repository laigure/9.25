"""Train/evaluate implicit-token scenario QA with Qwen2.5-7B LoRA.

The model sees only generic role/task text and 1-5 projected 288D Grounding
tokens.  Referring expressions and GT geometry remain outside the prompt.
GT geometry and answer keys are loaded only by the post-generation evaluator.
"""

import argparse
import collections
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_qa import (QATrainer, TokenProjector, PROMPT_PREFIX,
                      ASSISTANT_PREFIX, ASSISTANT_SUFFIX)
from strict_scene_qa_eval import evaluate as evaluate_scene_strict


class RelationalTokenProjector(nn.Module):
    """Contextualize ordered object tokens before mapping them into the LLM."""

    def __init__(self, token_dim, projector_hidden, llm_hidden,
                 max_roles=3, layers=1, heads=8):
        super().__init__()
        self.role_embedding = nn.Embedding(max_roles, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads,
            dim_feedforward=token_dim * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.relation_encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                                       norm=nn.LayerNorm(token_dim))
        self.projector = TokenProjector(token_dim, projector_hidden, llm_hidden)

    def forward_group(self, tokens, role_indices):
        contextual = self.relation_encoder(
            (tokens + self.role_embedding(role_indices)).unsqueeze(0)).squeeze(0)
        return self.projector(contextual)

    def forward(self, token):
        # Kept for compatibility with generic projector diagnostics.
        return self.projector(token)


class PairwiseRelationalTokenProjector(nn.Module):
    """Represent every directed object pair before LLM projection.

    The Grounding vectors are the only model inputs.  GT geometry is never
    read here; it is used separately to create training-only relation labels.
    """

    def __init__(self, token_dim, projector_hidden, llm_hidden,
                 max_roles=16, layers=2, heads=8):
        super().__init__()
        self.token_dim = token_dim
        self.role_embedding = nn.Embedding(max_roles, token_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(token_dim * 4, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
            nn.LayerNorm(token_dim))
        self.pair_merge = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads,
            dim_feedforward=token_dim * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.relation_encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(token_dim))
        self.lateral_head = nn.Linear(token_dim, 3)
        self.longitudinal_head = nn.Linear(token_dim, 3)
        self.projector = TokenProjector(token_dim, projector_hidden, llm_hidden)

    def encode_group(self, tokens, role_indices):
        base = tokens + self.role_embedding(role_indices)
        count = base.shape[0]
        source = base[:, None, :].expand(count, count, -1)
        target = base[None, :, :].expand(count, count, -1)
        pair_features = torch.cat(
            [source, target, source - target, source * target], dim=-1)
        pair_hidden = self.pair_mlp(pair_features)
        if count > 1:
            mask = ~torch.eye(count, dtype=torch.bool, device=tokens.device)
            pair_sum = (pair_hidden * mask.unsqueeze(-1)).sum(dim=1)
            pair_aggregate = pair_sum / float(count - 1)
        else:
            mask = torch.zeros((count, count), dtype=torch.bool,
                               device=tokens.device)
            pair_aggregate = torch.zeros_like(base)
        merged = self.pair_merge(torch.cat([base, pair_aggregate], dim=-1))
        contextual = self.relation_encoder(merged.unsqueeze(0)).squeeze(0)
        return contextual, pair_hidden, mask

    def forward_group(self, tokens, role_indices):
        contextual, _, _ = self.encode_group(tokens, role_indices)
        return self.projector(contextual)

    def auxiliary_logits(self, tokens, role_indices):
        _, pair_hidden, mask = self.encode_group(tokens, role_indices)
        directed_pairs = pair_hidden[mask]
        return (self.lateral_head(directed_pairs),
                self.longitudinal_head(directed_pairs), mask)

    def forward(self, token):
        return self.projector(token)


class ScenarioTrainer(QATrainer):
    def __init__(self, args):
        # The base class constructs the LLM and shared training machinery.
        # Defer checkpoint loading until the relational projector is installed.
        deferred = args.init_projector
        args.init_projector = None
        super().__init__(args)
        args.init_projector = deferred
        projector_class = (PairwiseRelationalTokenProjector
                           if args.relation_arch == "pairwise"
                           else RelationalTokenProjector)
        self.projector = projector_class(
            token_dim=288, projector_hidden=1024,
            llm_hidden=self.hidden_size,
            max_roles=args.max_object_roles,
            layers=args.relation_layers,
            heads=args.relation_heads).to(self.device)
        relation_params = sum(p.numel() for p in self.projector.parameters())
        self.private = {}
        print("relational projector params", relation_params,
              "| architecture", args.relation_arch,
              "| relation layers", args.relation_layers,
              "| heads", args.relation_heads,
              "| max roles", args.max_object_roles,
              "| auxiliary relation weight", args.aux_relation_weight,
              flush=True)
        if deferred:
            self.load_checkpoint(deferred)

    def token_arrays(self, record):
        path = record["token_path"]
        if not os.path.isabs(path):
            path = os.path.join(self.tokens_root, path)
        if path not in self._tokens_cache:
            with np.load(path) as data:
                self._tokens_cache[path] = (data["object_token"], data["object_id"])
        return self._tokens_cache[path]

    def check_token_identity(self, records):
        for record in records:
            tokens, ids = self.token_arrays(record)
            for obj in record["object_refs"]:
                index = obj["token_index"]
                assert str(ids[index]) == obj["object_id"], record["qa_id"]
                assert tokens[index].shape == (288,), record["qa_id"]
        return len(records)

    def pieces(self, record, with_answer):
        # This is the complete model input. No GT or predicted coordinate,
        # bounding box, distance, or relation field is read here.
        if record.get("input_policy") == "implicit_grounding_tokens_only":
            assert all(not any(key in obj for key in (
                "description", "category", "bbox", "center"))
                for obj in record["object_refs"]), record["qa_id"]
        pieces = [("text", self.encode(PROMPT_PREFIX), False),
                  ("text", self.encode(record["question"]), False)]
        tokens, _ = self.token_arrays(record)
        vectors = torch.stack([
            torch.as_tensor(tokens[obj["token_index"]], dtype=torch.float32,
                            device=self.device)
            for obj in record["object_refs"]])
        role_indices = torch.arange(len(record["object_refs"]), device=self.device)
        assert len(record["object_refs"]) <= self.args.max_object_roles
        contextual = self.projector.forward_group(vectors, role_indices)
        for index, obj in enumerate(record["object_refs"]):
            soft = contextual[index].to(self.embed_dtype).reshape(1, 1, -1)
            pieces += [("text", self.encode("\nObject {} token: ".format(obj["role"])), False),
                       ("soft", soft, False)]
        pieces.append(("text", self.encode(ASSISTANT_PREFIX), False))
        if with_answer:
            pieces += [("text", self.encode(record["answer"]), True),
                       ("text", self.encode(ASSISTANT_SUFFIX), True)]
        return pieces

    def additional_loss(self, records):
        """Supervise directed left/right and front/behind pair relations.

        Geometry is accessed only in this training-only label path.  It is not
        appended to a Grounding token, projected into the LLM, or serialized in
        a public QA prompt.
        """
        if (self.args.relation_arch != "pairwise" or
                self.args.aux_relation_weight <= 0):
            return None
        lateral_logits = []
        longitudinal_logits = []
        lateral_labels = []
        longitudinal_labels = []
        deadband = 1.0
        for record in records:
            token_array, _ = self.token_arrays(record)
            tokens = torch.stack([
                torch.as_tensor(token_array[obj["token_index"]],
                                dtype=torch.float32, device=self.device)
                for obj in record["object_refs"]])
            roles = torch.arange(len(record["object_refs"]), device=self.device)
            lat, lon, mask = self.projector.auxiliary_logits(tokens, roles)
            geometry = self.private[record["qa_id"]]["gt_geometry"]
            count = len(geometry)
            lat_target = []
            lon_target = []
            for source in range(count):
                for target in range(count):
                    if source == target:
                        continue
                    dx = float(geometry[source][0]) - float(geometry[target][0])
                    dy = float(geometry[source][1]) - float(geometry[target][1])
                    # 0=right/behind, 1=deadband, 2=left/front.
                    lat_target.append(2 if dy >= deadband else
                                      0 if dy <= -deadband else 1)
                    lon_target.append(2 if dx >= deadband else
                                      0 if dx <= -deadband else 1)
            assert lat.shape[0] == int(mask.sum()) == len(lat_target)
            lateral_logits.append(lat)
            longitudinal_logits.append(lon)
            lateral_labels.extend(lat_target)
            longitudinal_labels.extend(lon_target)
        if not lateral_labels:
            return None
        lat_logits = torch.cat(lateral_logits, dim=0)
        lon_logits = torch.cat(longitudinal_logits, dim=0)
        lat_labels = torch.tensor(lateral_labels, dtype=torch.long,
                                  device=self.device)
        lon_labels = torch.tensor(longitudinal_labels, dtype=torch.long,
                                  device=self.device)
        auxiliary = F.cross_entropy(lat_logits, lat_labels)
        auxiliary = auxiliary + F.cross_entropy(lon_logits, lon_labels)
        return auxiliary * self.args.aux_relation_weight


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def select_eval(records, args):
    if args.eval_per_scenario:
        grouped = collections.defaultdict(list)
        for row in records:
            grouped[row["scenario"]].append(row)
        rng = random.Random(args.seed)
        return [row for scenario in sorted(grouped)
                for row in rng.sample(grouped[scenario],
                                      min(args.eval_per_scenario, len(grouped[scenario])))]
    return records[:args.max_eval] if args.max_eval else records


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parsed_key(scenario, text):
    t = norm(text)
    if scenario in ("closer_of_two", "closer_to_ego", "which_is_left",
                    "closest_of_three_to_ego"):
        choices = "abc" if scenario == "closest_of_three_to_ego" else "ab"
        m = re.search(r"\bobject ([{}])\b".format(choices), t)
        return m.group(1).upper() if m else None
    if scenario == "ego_near_far":
        near = bool(re.search(r"\bnear\b", t))
        far = bool(re.search(r"\bfar\b", t))
        return "near" if near and not far else "far" if far and not near else None
    lateral = "left" if re.search(r"\bleft\b", t) else "right" if re.search(r"\bright\b", t) else None
    longitudinal = "behind" if re.search(r"\bbehind\b", t) else "front" if re.search(r"\bfront\b", t) else None
    if lateral and longitudinal:
        return lateral + "_" + longitudinal
    return lateral or longitudinal


def evaluate(records, predictions, private):
    rows, score = [], collections.Counter()
    by_scenario = collections.defaultdict(lambda: [0, 0])
    for rec, pred in zip(records, predictions):
        key = private[rec["qa_id"]]["answer_key"]
        parsed = parsed_key(rec["scenario"], pred)
        correct = parsed == key
        exact = norm(pred) == norm(rec["answer"])
        score["correct"] += correct
        score["exact"] += exact
        score["parsed"] += parsed is not None
        by_scenario[rec["scenario"]][0] += correct
        by_scenario[rec["scenario"]][1] += 1
        rows.append({"qa_id": rec["qa_id"], "scenario": rec["scenario"],
                     "question": rec["question"], "answer": rec["answer"],
                     "prediction": pred, "gt_key": key, "pred_key": parsed,
                     "correct": bool(correct), "exact": bool(exact)})
    n = len(rows)
    return {"n": n, "accuracy": score["correct"] / max(n, 1),
            "exact_sentence": score["exact"] / max(n, 1),
            "parse_rate": score["parsed"] / max(n, 1),
            "by_scenario": {k: {"n": v[1], "accuracy": v[0] / v[1]}
                            for k, v in by_scenario.items()}, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/3eed_data/qa_v2/scenario_qa/qa.jsonl")
    ap.add_argument("--private-gt", default=None)
    ap.add_argument("--tokens-root", default=None)
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["projector", "lora", "eval"], default="lora")
    ap.add_argument("--init-projector", default=None)
    ap.add_argument("--with-lora", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gen-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--eval-per-scenario", type=int, default=0,
                    help="Deterministic cap per question type; 0 uses full split")
    ap.add_argument("--eval-splits", default="val,test")
    ap.add_argument("--relation-arch", choices=["transformer", "pairwise"],
                    default="transformer")
    ap.add_argument("--relation-layers", type=int, default=1)
    ap.add_argument("--relation-heads", type=int, default=8)
    ap.add_argument("--aux-relation-weight", type=float, default=0.0,
                    help="Training-only GT pair-relation loss weight")
    ap.add_argument("--max-object-roles", type=int, default=16,
                    help="Maximum variable-cardinality scene object set")
    args = ap.parse_args()
    args.geom = "none"
    args.no_tokens = False
    args.pred_geometry = None
    args.oracle_center = False
    if args.mode == "eval" and not args.init_projector:
        ap.error("--mode eval requires --init-projector")
    args.private_gt = args.private_gt or os.path.join(os.path.dirname(args.data), "private_gt.jsonl")
    args.tokens_root = args.tokens_root or os.path.dirname(os.path.dirname(args.data))
    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = read_jsonl(args.data)
    private_rows = read_jsonl(args.private_gt)
    private = {x["qa_id"]: x for x in private_rows}
    assert {x["qa_id"] for x in all_rows} == set(private)
    split = {name: [r for r in all_rows if r["qa_split"] == name]
             for name in ("train", "val", "test")}
    seq = {name: {r["sequence"] for r in records} for name, records in split.items()}
    assert not (seq["train"] & seq["val"] or seq["train"] & seq["test"] or seq["val"] & seq["test"])
    trainer = ScenarioTrainer(args)
    trainer.private = private
    trainer.check_token_identity(all_rows)
    if args.mode in ("projector", "lora"):
        train = split["train"][:args.max_train] if args.max_train else split["train"]
        val = select_eval(split["val"], args)
        print("TRAIN", len(train), "VAL", len(val), flush=True)
        trainer.train(train, val, args.epochs, log_every=args.log_every, out_dir=args.out_dir)
        trainer.load_checkpoint(os.path.join(args.out_dir, "best.pt"))
    for name in args.eval_splits.split(","):
        records = select_eval(split[name], args)
        print("GENERATE", name, len(records), flush=True)
        predictions = trainer.generate(records)
        if records and "canonical_answer" in private[records[0]["qa_id"]]:
            prediction_rows = [{"qa_id": row["qa_id"], "prediction": prediction}
                               for row, prediction in zip(records, predictions)]
            result = evaluate_scene_strict(
                records, prediction_rows,
                [private[row["qa_id"]] for row in records])
        else:
            result = evaluate(records, predictions, private)
        with open(os.path.join(args.out_dir, "eval_{}.json".format(name)), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(name, json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
