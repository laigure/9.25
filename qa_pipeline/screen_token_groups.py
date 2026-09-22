"""Lightweight screening of T0-T9 implicit-token input groups.

This is a SCREENING harness, not the final QA model. It trains one small
classifier per input group on the fixed scenario-QA question set and reports
comparative accuracy. Numbers here are screening signals only and must never
be reported as LLM QA results.

Data (fixed for every group):
- questions/answers: artifacts/qa_v3/scenario_qa/qa.jsonl + private_gt.jsonl
  (same sequence-isolated train/val/test for all groups);
- visual features: the rich export from export_rich_features.py, joined by
  object_id; only model-produced query features and point-cloud scene
  features are used. GT boxes never enter any model input; they are used
  after prediction only for grounding-correct/incorrect diagnostics.

Light model (identical for every group):
- frozen RoBERTa-base mean-pooled question embedding (768-D), no fine-tuning;
- concatenated visual block per group (see GROUPS);
- 2-layer MLP, masked-softmax over the answer keys valid for the scenario.

Groups (top-k = full-phrase text ranking, no GT; layer indices 0..5):
  T0  text only
  T1  layer5 top1
  T2  layer4 top1
  T3  layer2 top1 (middle layer)
  T4  layer5 top1 + layer4 top1 (concatenated)
  T5  layer5 top1 + scene seed-mean
  T6  layer5 top1 + 4 scene features (fp2 mean/max, seed mean/max)
  T7  layer5 top3
  T8  layer5 top5
  T9  layer5 top3 + 4 scene features
Per object reference, features are emitted in role order (A, B, C); rows with
fewer targets zero-pad the absent role blocks, so the visual dimension is the
same for every question.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCENE_FEATURES = ("scene_fp2_mean", "scene_fp2_max", "scene_seed_mean", "scene_seed_max")
MAX_ROLES = 3  # qa_v3 rows reference 1..3 objects, roles always A, B, C
GROUPS = {
    "T0": [],
    "T1": [("tok", 5, 0)],
    "T2": [("tok", 4, 0)],
    "T3": [("tok", 2, 0)],
    "T4": [("tok", 5, 0), ("tok", 4, 0)],
    "T5": [("tok", 5, 0), ("scene", "seed_mean")],
    "T6": [("tok", 5, 0)] + [("scene", name[6:]) for name in SCENE_FEATURES],
    "T7": [("tok", 5, 0), ("tok", 5, 1), ("tok", 5, 2)],
    "T8": [("tok", 5, 0), ("tok", 5, 1), ("tok", 5, 2), ("tok", 5, 3), ("tok", 5, 4)],
    "T9": [("tok", 5, 0), ("tok", 5, 1), ("tok", 5, 2)]
          + [("scene", name[6:]) for name in SCENE_FEATURES],
    # Extra attribution diagnostics, not part of the directive's T0-T9 table:
    # which channel carries the scene-feature gain (pre-fusion point cloud,
    # text-conditioned seed features, or the token itself).
    "X1": [("tok", 5, 0), ("scene", "fp2_mean")],
    "X2": [("scene", "seed_mean")],
    "X3": [("scene", name[6:]) for name in SCENE_FEATURES],
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_feature_records(export_dir):
    index = {}
    for record in read_jsonl(os.path.join(export_dir, "records.jsonl")):
        index[record["object_id"]] = (record["feature_file"], record["feature_row"])
    return index


class FeatureStore:
    def __init__(self, export_dir, index):
        self.root = export_dir
        self.index = index
        self.cache = {}
        self.cache_order = []

    def row(self, object_id):
        rel, row = self.index[object_id]
        if rel not in self.cache:
            if len(self.cache_order) >= 32:
                self.cache.pop(self.cache_order.pop(0), None)
            self.cache[rel] = np.load(os.path.join(self.root, rel))
            self.cache_order.append(rel)
        return self.cache[rel], row

    def token(self, object_id, layer, rank):
        data, row = self.row(object_id)
        index = int(data["layer_top5_idx"][row, layer, rank])
        return data["layer_features_top5"][row, layer, rank], index

    def scene(self, object_id, name):
        data, row = self.row(object_id)
        return data["scene_" + name][row]

    def top1_iou_last(self, object_id, rank=0):
        data, row = self.row(object_id)
        index = int(data["layer_top5_idx"][row, 5, rank])
        return float(data["layer_iou"][row, 5, index])


def build_visual(specs, refs, store):
    if not specs:
        return np.zeros(0, dtype=np.float32)
    if len(refs) > MAX_ROLES:
        raise ValueError("more referenced objects than role slots")
    role_blocks = []
    for ref in refs:
        parts = []
        for spec in specs:
            if spec[0] == "tok":
                parts.append(store.token(ref["object_id"], spec[1], spec[2])[0])
            elif spec[0] == "scene":
                parts.append(store.scene(ref["object_id"], spec[1]))
            else:
                raise ValueError(spec)
        role_blocks.append(np.concatenate(parts).astype(np.float32))
    block_dim = role_blocks[0].shape[0]
    for block in role_blocks:
        if block.shape[0] != block_dim:
            raise ValueError("inconsistent per-role feature block size")
    while len(role_blocks) < MAX_ROLES:
        role_blocks.append(np.zeros(block_dim, dtype=np.float32))
    return np.concatenate(role_blocks)


class ScreenModel(nn.Module):
    def __init__(self, text_dim, visual_dim, num_keys):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_keys),
        )

    def forward(self, text, visual):
        return self.net(torch.cat([text, visual], dim=-1))


def encode_texts(questions, encoder_dir, batch_size=64, max_length=160):
    from transformers import RobertaModel, RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained(encoder_dir)
    model = RobertaModel.from_pretrained(encoder_dir).cuda().eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(questions), batch_size):
            batch = tokenizer(questions[start:start + batch_size], padding=True,
                              truncation=True, max_length=max_length,
                              return_tensors="pt").to("cuda")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(embeddings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", default="/root/autodl-tmp/3eed_data/qa_v3/scenario_qa")
    parser.add_argument("--features", default="/root/autodl-tmp/3eed_data/qa_features_v1")
    parser.add_argument("--encoder", default="/root/3eedqa/3EED/data/roberta_base")
    parser.add_argument("--out-dir", default="/root/autodl-tmp/3eed_data/screening/runs")
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--error-limit", type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(os.path.join(args.qa_dir, "qa.jsonl"))
    private = {r["qa_id"]: r for r in read_jsonl(os.path.join(args.qa_dir, "private_gt.jsonl"))}
    assert len(rows) == len(private)
    specs = GROUPS[args.group]
    store = None
    if specs:
        store = FeatureStore(args.features, load_feature_records(args.features))

    for row in rows:
        assert row["qa_id"] in private
        for ref in row["object_refs"]:
            if store is not None and ref["object_id"] not in store.index:
                raise KeyError("object missing from rich export: " + ref["object_id"])

    keys = sorted({private[r["qa_id"]]["answer_key"] for r in rows})
    key_to_id = {key: i for i, key in enumerate(keys)}
    scenario_keys = {}
    for row in rows:
        scenario = row["scenario"]
        key = private[row["qa_id"]]["answer_key"]
        scenario_keys.setdefault(scenario, set()).add(key)
    scenario_keys = {s: sorted(v) for s, v in scenario_keys.items()}

    if specs:
        text = encode_texts([r["question"] for r in rows], args.encoder)
        visuals, ious, n_roles = [], [], []
        for row in rows:
            refs = sorted(row["object_refs"], key=lambda r: r["role"])
            visuals.append(build_visual(specs, refs, store))
            ious.append(min(store.top1_iou_last(ref["object_id"]) for ref in refs))
            n_roles.append(len(refs))
        visual = np.stack(visuals)
        ious = np.asarray(ious, dtype=np.float32)
        n_roles = np.asarray(n_roles)
    else:
        text = encode_texts([r["question"] for r in rows], args.encoder)
        visual = np.zeros((len(rows), 0), dtype=np.float32)
        ious = np.full(len(rows), -1.0, dtype=np.float32)

    split_ids = {"train": [], "val": [], "test": []}
    for i, row in enumerate(rows):
        split_ids[row["qa_split"]].append(i)

    # scale visual features with train-split statistics only (no-op for T0);
    # stats are fit per role block on rows where the role exists, and the
    # blocks of absent roles are then kept at exactly zero
    if visual.shape[1]:
        per_role = visual.shape[1] // MAX_ROLES
        is_train = np.zeros(len(rows), dtype=bool)
        is_train[split_ids["train"]] = True
        for role in range(MAX_ROLES):
            block = visual[:, role * per_role:(role + 1) * per_role]
            present = n_roles > role
            fit = present & is_train
            if fit.any():
                mean = block[fit].mean(0, keepdims=True)
                std = block[fit].std(0, keepdims=True) + 1e-6
                block -= mean
                block /= std
            block[~present] = 0.0

    labels = np.asarray([key_to_id[private[r["qa_id"]]["answer_key"]] for r in rows])
    scenarios = np.asarray([r["scenario"] for r in rows])

    def split_tensors(split):
        idx = torch.as_tensor(split_ids[split], dtype=torch.long)
        return (text[idx].cuda(), torch.as_tensor(visual, dtype=torch.float32)[idx].cuda(),
                torch.as_tensor(labels, dtype=torch.long)[idx].cuda(), idx)

    valid_keys = []
    for scenario in scenarios:
        mask = torch.zeros(len(keys), dtype=torch.bool)
        mask[[key_to_id[k] for k in scenario_keys[scenario]]] = True
        valid_keys.append(mask)
    key_mask = torch.stack(valid_keys)

    train_text, train_visual, train_labels, train_idx = split_tensors("train")
    generator = torch.Generator().manual_seed(args.seed)
    model = ScreenModel(text.shape[1], visual.shape[1], len(keys)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(split):
        model.eval()
        with torch.no_grad():
            inputs = split_tensors(split)
            logits = model(inputs[0], inputs[1])
            masked = logits.masked_fill(~key_mask[inputs[3]].cuda(), float("-inf"))
            pred = masked.argmax(-1)
            correct = (pred == inputs[2]).cpu().numpy()
        return inputs[3].cpu().numpy(), pred.cpu().numpy(), correct

    best_val = -1.0
    best_state = None
    history = []
    order = torch.randperm(train_text.shape[0], generator=generator)
    for epoch in range(args.epochs):
        model.train()
        permutation = order[torch.randperm(len(order), generator=generator)]
        total_loss, seen = 0.0, 0
        for start in range(0, len(permutation), args.batch_size):
            batch = permutation[start:start + args.batch_size]
            logits = model(train_text[batch], train_visual[batch])
            masked = logits.masked_fill(~key_mask[train_idx[batch]].cuda(), float("-inf"))
            loss = loss_fn(masked, train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(batch)
            seen += len(batch)
        _, _, val_correct = evaluate("val")
        history.append({"epoch": epoch + 1, "train_loss": round(total_loss / seen, 5),
                        "val_acc": round(float(val_correct.mean()), 5)})
        if val_correct.mean() > best_val:
            best_val = float(val_correct.mean())
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    run_dir = Path(args.out_dir) / args.group
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {"group": args.group, "specs": specs, "note":
              "screening harness only; not an LLM QA result",
              "visual_layout": "per-target blocks in role order A,B,C; absent roles "
                               "zero-padded; per-role train-only standardization",
              "n_samples": {s: len(split_ids[s]) for s in ("train", "val", "test")},
              "text_dim": int(text.shape[1]), "visual_dim": int(visual.shape[1]),
              "num_keys": len(keys), "epochs": args.epochs, "lr": args.lr,
              "batch_size": args.batch_size, "seed": args.seed,
              "best_val_acc": round(best_val, 5), "history": history}
    for split in ("val", "test"):
        idx, pred, correct = evaluate(split)
        per_scenario = {}
        for scenario in sorted(set(scenarios[idx])):
            sel = scenarios[idx] == scenario
            per_scenario[scenario] = {
                "n": int(sel.sum()), "acc": round(float(correct[sel].mean()), 5)}
        entry = {"acc": round(float(correct.mean()), 5), "by_scenario": per_scenario}
        if specs:
            gt = ious[idx]
            entry["grounding"] = {
                "all_refs_top1_iou_ge_025": round(float((gt >= 0.25).mean()), 5),
                "acc_when_grounded": round(float(correct[gt >= 0.25].mean()), 5) if (gt >= 0.25).any() else None,
                "acc_when_missed": round(float(correct[gt < 0.25].mean()), 5) if (gt < 0.25).any() else None,
            }
        report[split] = entry
        errors = []
        wrong = np.flatnonzero(~correct)
        for i in wrong[:args.error_limit]:
            row = rows[idx[i]]
            item = {"qa_id": row["qa_id"], "scenario": row["scenario"],
                    "question": row["question"], "gold": keys[int(labels[idx[i]])],
                    "pred": keys[int(pred[i])],
                    "object_ids": [ref["object_id"] for ref in row["object_refs"]]}
            if specs:
                item["min_top1_iou"] = round(float(ious[idx[i]]), 4)
            errors.append(item)
        (run_dir / f"errors_{split}.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors) + "\n",
            encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": report}, run_dir / "screen_model.pt")
    print(json.dumps({k: report[k] for k in ("group", "visual_dim", "best_val_acc", "val", "test")},
                     ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
