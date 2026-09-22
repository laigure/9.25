"""Train/evaluate implicit-token scenario QA with Qwen2.5-7B LoRA.

The model sees only question text and 1-3 projected 288D Grounding tokens.
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

from train_qa import (QATrainer, TokenProjector, PROMPT_PREFIX,
                      ASSISTANT_PREFIX, ASSISTANT_SUFFIX)


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


class ScenarioTrainer(QATrainer):
    def __init__(self, args):
        # The base class constructs the LLM and shared training machinery.
        # Defer checkpoint loading until the relational projector is installed.
        deferred = args.init_projector
        args.init_projector = None
        super().__init__(args)
        args.init_projector = deferred
        self.projector = RelationalTokenProjector(
            token_dim=288, projector_hidden=1024,
            llm_hidden=self.hidden_size,
            max_roles=args.max_object_roles,
            layers=args.relation_layers,
            heads=args.relation_heads).to(self.device)
        relation_params = sum(p.numel() for p in self.projector.parameters())
        print("relational projector params", relation_params,
              "| relation layers", args.relation_layers,
              "| heads", args.relation_heads,
              "| max roles", args.max_object_roles, flush=True)
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
    ap.add_argument("--relation-layers", type=int, default=1)
    ap.add_argument("--relation-heads", type=int, default=8)
    ap.add_argument("--max-object-roles", type=int, default=3)
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
    private = {x["qa_id"]: x for x in read_jsonl(args.private_gt)}
    assert {x["qa_id"] for x in all_rows} == set(private)
    split = {name: [r for r in all_rows if r["qa_split"] == name]
             for name in ("train", "val", "test")}
    seq = {name: {r["sequence"] for r in records} for name, records in split.items()}
    assert not (seq["train"] & seq["val"] or seq["train"] & seq["test"] or seq["val"] & seq["test"])
    trainer = ScenarioTrainer(args)
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
        result = evaluate(records, predictions, private)
        with open(os.path.join(args.out_dir, "eval_{}.json".format(name)), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(name, json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
