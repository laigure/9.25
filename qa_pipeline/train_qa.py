"""Train / evaluate the downstream spatial-QA stack (Experiments B, C, D, E).

One script, three modes, so every experiment uses byte-identical data loading,
prompt construction, generation and evaluation:

    --mode projector   freeze the LLM, train the shared token projector
    --mode lora        freeze the LLM base, train LoRA + projector
                       (projector initialised from the projector-only run)
    --mode eval        no training; generate with a saved checkpoint;
                       --no-tokens gives the text-only baseline

Optional extra geometry input (Experiments D/E; default --geom none is the
B/C setup byte-for-byte):

    --geom center --pred-geometry G.npz   concat 3 normalised coords to the
                                          token before the shared projector
    --geom box    --pred-geometry G.npz   same with center+size
    --geom relative --pred-geometry G.npz token-only projector plus a third
                                          soft token from predicted relative
                                          geometry (dx,dy,dz,d2d,d3d)
    --oracle-center                       diagnostic upper bound: feed GT
                                          center/size (never a final result)

Projector growth keeps every trained parameter: the first layer's 288 input
columns are copied and the new geometry columns start at zero, so a run
initialised from a 288-d checkpoint starts functionally identical to it.
Normalisation uses train-split mean/std only and travels inside the
checkpoint, so val/test always reuse the training statistics.

Fixed setup (per directive; the grounding model is never loaded here):
  * 3EED official grounding settings kept verbatim: feature_name =
    last_query_features, token_dim = 288, selection =
    contrastive_top1_annotated_text_span. Tokens are precomputed 288-d
    exports, so grounding has no gradient path (frozen by construction).
  * LLM: Qwen2.5-7B-Instruct, bf16, hidden size read from
    model.config.hidden_size (3584 for this checkpoint).
  * Shared projector (one module for target and reference):
    Linear(288, 1024) -> GELU -> Linear(1024, hidden_size).
  * LoRA (mode lora): r=16, alpha=32, dropout=0.05, target_modules =
    [q_proj, k_proj, v_proj, o_proj], bias=none (deliberately NOT
    gate/up/down_proj).
  * Prompt: system + question, then the two soft tokens at fixed markers
    ("Target object token: ", "Reference object token: "), then the assistant
    answer. Labels cover the answer text and its terminating <|im_end|>;
    every prompt token is -100.
  * AdamW; projector_lr=1e-4, lora_lr=1e-4 (separate param groups),
    weight_decay=0.01, epochs=3, warmup_ratio=0.03, cosine schedule,
    max_grad_norm=1.0, bf16, per-device batch + gradient accumulation to an
    effective batch of 8-16 (recorded in experiment_notes.md).

Experiment-log notes (auto-written to <out-dir>/experiment_notes.md): the 3EED
grounding structure/checkpoint/settings come from 3EED; 3EED has no LLM/LoRA;
Qwen2.5-7B-Instruct, the projector and LoRA are our new downstream QA settings.
The 1e-4 LR matches the magnitude 3EED uses for its non-point-encoder modules;
it is NOT 3EED's LoRA setting (3EED has none).

Examples
    python train_qa.py --mode projector --smoke
    python train_qa.py --mode projector
    python train_qa.py --mode lora --init-projector <run>/best.pt
    python train_qa.py --mode eval --init-projector <run>/best.pt --eval-splits val,test
    python train_qa.py --mode eval --no-tokens            # text-only baseline
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # model is pre-downloaded
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.optimization import get_cosine_schedule_with_warmup  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qa_eval import evaluate  # noqa: E402

PROMPT_PREFIX = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\n")
TARGET_MARKER = "\nTarget object token: "
REFERENCE_MARKER = "\nReference object token: "
GEOMETRY_MARKER = "\nRelative geometry token: "
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
ASSISTANT_SUFFIX = "<|im_end|>\n"

GEOM_DIM = {"none": 0, "center": 3, "box": 6, "relative": 0}
RELATIVE_DIM = 5  # dx, dy, dz, distance_2d, distance_3d

LORA_CONFIG = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                   task_type="CAUSAL_LM")


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


class TokenProjector(nn.Module):
    """Shared 288 -> 1024 -> hidden_size projector (target and reference)."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class QATrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda"
        self.geom_mode = args.geom
        self.geom_stats = None
        self.geom_projector = None
        self.expanded_from = None

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # No explicit attn_implementation: torch 2.1.0 predates the SDPA
        # availability check, so transformers picks sdpa/eager itself.
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.model.config.use_cache = False
        self.hidden_size = self.model.config.hidden_size
        self.embed = self.model.get_input_embeddings()
        self.embed_dtype = self.embed.weight.dtype

        with_lora = args.mode == "lora" or getattr(args, "with_lora", False)
        if with_lora:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(**LORA_CONFIG))

        for param in self.model.parameters():
            param.requires_grad_(False)
        lora_params = []
        if with_lora:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
                    lora_params.append((name, param))
        self.lora_param_names = [name for name, _ in lora_params]

        self.projector = TokenProjector(
            288 + GEOM_DIM[self.geom_mode], 1024, self.hidden_size).to(self.device)
        if self.geom_mode == "relative":
            self.geom_projector = TokenProjector(
                RELATIVE_DIM, 1024, self.hidden_size).to(self.device)
        self.checkpoint_loaded = None
        if args.init_projector:
            self.load_checkpoint(args.init_projector)

        self.tokens_root = (args.tokens_root
                            or os.path.dirname(os.path.abspath(args.data)))
        self._tokens_cache = {}
        self._geom_cache = None

        base_trainable = sum(p.numel() for p in self.model.parameters()
                             if p.requires_grad)
        projector_params = sum(p.numel() for p in self.projector.parameters())
        if self.geom_projector is not None:
            projector_params += sum(p.numel() for p in self.geom_projector.parameters())
        print("hidden_size", self.hidden_size,
              "| embed dtype", self.embed_dtype,
              "| trainable model params", base_trainable,
              "| lora tensors", len(lora_params),
              "| projector params", projector_params,
              "| geom mode", self.geom_mode,
              "| projector in_dim", 288 + GEOM_DIM[self.geom_mode],
              flush=True)

    # ------------------------------------------------------------- checkpoint
    def state_dicts(self):
        state = {"projector": self.projector.state_dict(), "lora": None}
        if self.lora_param_names:
            from peft import get_peft_model_state_dict
            state["lora"] = get_peft_model_state_dict(self.model)
        if self.geom_projector is not None:
            state["geom_projector"] = self.geom_projector.state_dict()
        state["geom"] = {"mode": self.geom_mode,
                         "oracle_center": bool(getattr(self.args, "oracle_center", False)),
                         "stats": self.geom_stats}
        return state

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        current = self.projector.state_dict()
        mismatched = [key for key, value in checkpoint["projector"].items()
                      if value.shape != current[key].shape]
        if mismatched:
            self._load_expanded_projector(checkpoint["projector"])
        else:
            self.projector.load_state_dict(checkpoint["projector"])
        if checkpoint.get("geom_projector") and self.geom_projector is not None:
            self.geom_projector.load_state_dict(checkpoint["geom_projector"])
        if self.lora_param_names and checkpoint.get("lora"):
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, checkpoint["lora"])
        if checkpoint.get("lora") and not self.lora_param_names:
            print("WARNING: checkpoint contains LoRA weights but --with-lora "
                  "was not given; LoRA part ignored", flush=True)
        geom_meta = checkpoint.get("geom")
        if geom_meta:
            assert geom_meta["mode"] == self.geom_mode, \
                "checkpoint geometry mode {} != --geom {}".format(
                    geom_meta["mode"], self.geom_mode)
            self.geom_stats = geom_meta["stats"]
            print("checkpoint carries geometry stats |", json.dumps(geom_meta["stats"]),
                  flush=True)
        elif self.geom_mode != "none" and self.args.mode == "eval":
            raise SystemExit("eval with --geom needs a checkpoint that carries "
                             "geometry metadata (train the geometry run first)")
        self.checkpoint_loaded = path
        print("loaded checkpoint", path,
              "| projector", "yes",
              "| lora", "yes" if (checkpoint.get("lora") and self.lora_param_names) else "no",
              "| geom projector", "yes" if checkpoint.get("geom_projector") else "no",
              flush=True)

    def _load_expanded_projector(self, saved):
        """Grow the projector input: keep every trained column of the first
        layer, append zero-init columns for the new geometry dims, keep all
        later layers unchanged. At init the model is therefore functionally
        identical to the checkpoint it continues from."""
        current = self.projector.state_dict()
        new_state = {}
        for key, old in saved.items():
            target = current[key]
            if old.shape == target.shape:
                new_state[key] = old
                continue
            assert key.startswith("net.0.") and len(old.shape) == 2, \
                "only the first-layer weight may grow, got " + key
            new = torch.zeros(target.shape, dtype=old.dtype)
            new[:, :old.shape[1]] = old
            new_state[key] = new
            self.expanded_from = old.shape[1]
            print("projector expansion: {} {} -> {} | kept {} trained columns, "
                  "{} new columns zero-initialised".format(
                      key, tuple(old.shape), tuple(target.shape),
                      old.shape[1], target.shape[1] - old.shape[1]), flush=True)
        assert self.expanded_from is not None
        self.projector.load_state_dict(new_state)

    # --------------------------------------------------------------- data
    def token_arrays(self, record):
        path = record["target_token_path"]
        assert path == record["reference_token_path"], \
            "target/reference tokens must come from the same npz"
        if not os.path.isabs(path):
            path = os.path.join(self.tokens_root, path)
        if path not in self._tokens_cache:
            data = np.load(path)
            self._tokens_cache[path] = (data["object_token"], data["object_id"])
        return self._tokens_cache[path]

    def check_token_identity(self, records):
        for record in records:
            object_token, object_id = self.token_arrays(record)
            assert str(object_id[record["target_token_index"]]) == \
                record["target_object_id"], "target token mismatch " + record["qa_id"]
            assert str(object_id[record["reference_token_index"]]) == \
                record["reference_object_id"], "reference token mismatch " + record["qa_id"]
        return len(records)

    # -------------------------------------------------------------- geometry
    def geometry_arrays(self):
        if self._geom_cache is None:
            if not self.args.pred_geometry:
                raise SystemExit("--geom {} needs --pred-geometry".format(self.geom_mode))
            data = np.load(self.args.pred_geometry)
            self._geom_cache = {key: data[key] for key in data.keys()}
        return self._geom_cache

    def check_geometry_alignment(self, records):
        """The geometry rows must line up with the token rows; verified once
        over the whole arrays and per record via object_id."""
        geom = self.geometry_arrays()
        object_token, object_id = self.token_arrays(records[0])
        assert np.array_equal(np.asarray(geom["object_id"]).astype(str),
                              np.asarray(object_id).astype(str)), \
            "geometry npz rows do not match the token npz rows"
        for record in records:
            index = record["target_token_index"]
            assert str(geom["object_id"][index]) == record["target_object_id"], \
                "target geometry mismatch " + record["qa_id"]
            index = record["reference_token_index"]
            assert str(geom["object_id"][index]) == record["reference_object_id"], \
                "reference geometry mismatch " + record["qa_id"]
        self._geom_cache = geom
        return len(records)

    def raw_row_geometry(self, row):
        """Unnormalised geometry input for one object row (mode-dependent)."""
        geom = self.geometry_arrays()
        if getattr(self.args, "oracle_center", False):
            center = np.asarray(geom["gt_center"][row], dtype=np.float64)
            size = np.asarray(geom["gt_size"][row], dtype=np.float64)
        else:
            center = np.asarray(geom["pred_center"][row], dtype=np.float64)
            size = np.asarray(geom["pred_size"][row], dtype=np.float64)
        if self.geom_mode == "center":
            return center
        if self.geom_mode == "box":
            return np.concatenate([center, size])
        raise AssertionError("raw_row_geometry called in mode " + self.geom_mode)

    def raw_relative_geometry(self, record):
        """[dx, dy, dz, distance_2d, distance_3d] from PREDICTED centers
        (target - reference); GT is never used here."""
        geom = self.geometry_arrays()
        target = np.asarray(geom["pred_center"][record["target_token_index"]],
                            dtype=np.float64)
        reference = np.asarray(geom["pred_center"][record["reference_token_index"]],
                               dtype=np.float64)
        delta = target - reference
        return np.array([delta[0], delta[1], delta[2],
                         float(np.hypot(delta[0], delta[1])),
                         float(np.linalg.norm(delta))], dtype=np.float64)

    def setup_geometry(self, records):
        """Normalisation statistics come from the TRAIN split only and travel
        with the checkpoint, so validation/test always reuse train statistics."""
        self.check_geometry_alignment(records)
        if self.geom_stats is not None:
            print("geometry stats from checkpoint (train-split statistics)", flush=True)
            return
        train_records = [r for r in records if r["qa_split"] == "train"]
        if self.geom_mode == "relative":
            values = np.stack([self.raw_relative_geometry(r) for r in train_records])
        else:
            rows = sorted({r["target_token_index"] for r in train_records} |
                          {r["reference_token_index"] for r in train_records})
            values = np.stack([self.raw_row_geometry(row) for row in rows])
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-6] = 1.0
        self.geom_stats = {"mean": [round(float(v), 6) for v in mean],
                           "std": [round(float(v), 6) for v in std],
                           "n_train_vectors": int(values.shape[0]),
                           "source": ("GT geometry (oracle, diagnostic)"
                                      if getattr(self.args, "oracle_center", False)
                                      and self.geom_mode != "relative"
                                      else "predicted geometry from the grounding model"),
                           "stats_split": "train only"}
        print("geometry stats (train split, {} vectors):".format(values.shape[0]),
              json.dumps(self.geom_stats["mean"]), json.dumps(self.geom_stats["std"]),
              flush=True)

    def geom_tensor(self, record, which):
        object_token, _ = self.token_arrays(record)
        stats = self.geom_stats
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        if self.geom_mode == "relative":
            raw = self.raw_relative_geometry(record)
        else:
            row = record["target_token_index"] if which == "target" \
                else record["reference_token_index"]
            raw = self.raw_row_geometry(row)
        normalized = (raw - mean) / std
        return torch.tensor(normalized, dtype=torch.float32, device=self.device)

    def make_geometry_token(self, record):
        soft = self.geom_projector(self.geom_tensor(record, "relative"))
        return soft.to(self.embed_dtype).reshape(1, 1, -1)

    def make_soft(self, record):
        object_token, _ = self.token_arrays(record)
        target = torch.tensor(object_token[record["target_token_index"]],
                              dtype=torch.float32, device=self.device)
        reference = torch.tensor(object_token[record["reference_token_index"]],
                                 dtype=torch.float32, device=self.device)
        if self.geom_mode in ("center", "box"):
            target = torch.cat([target, self.geom_tensor(record, "target")])
            reference = torch.cat([reference, self.geom_tensor(record, "reference")])
        soft_a = self.projector(target).to(self.embed_dtype).reshape(1, 1, -1)
        soft_b = self.projector(reference).to(self.embed_dtype).reshape(1, 1, -1)
        return soft_a, soft_b

    def pieces(self, record, with_answer):
        pieces = [("text", self.encode(PROMPT_PREFIX), False),
                  ("text", self.encode(record["question"]), False)]
        if not self.args.no_tokens:
            soft_a, soft_b = self.make_soft(record)
            pieces += [("text", self.encode(TARGET_MARKER), False), ("soft", soft_a, False),
                       ("text", self.encode(REFERENCE_MARKER), False), ("soft", soft_b, False)]
            if self.geom_mode == "relative":
                pieces += [("text", self.encode(GEOMETRY_MARKER), False),
                           ("soft", self.make_geometry_token(record), False)]
        # The assistant prefix is part of the prompt in BOTH training and
        # generation, so the generation context is exactly the training prefix
        # (asserted in the smoke test). Fixes the attempt-1 bug where the
        # generation prompt stopped at the last soft token.
        pieces += [("text", self.encode(ASSISTANT_PREFIX), False)]
        if with_answer:
            pieces += [("text", self.encode(record["answer"]), True),
                       ("text", self.encode(ASSISTANT_SUFFIX), True)]
        return pieces

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def to_tensors(self, pieces):
        embeds, labels = [], []
        for piece in pieces:
            if piece[0] == "soft":
                embeds.append(piece[1])
                labels.append(torch.full((1,), -100, dtype=torch.long,
                                         device=self.device))
            else:
                _, ids, labeled = piece
                with torch.no_grad():
                    embeds.append(self.embed(torch.tensor([ids], device=self.device)))
                if labeled:
                    labels.append(torch.tensor(ids, device=self.device))
                else:
                    labels.append(torch.full((len(ids),), -100, dtype=torch.long,
                                             device=self.device))
        # pieces are [1, L_piece, H]; concatenate along the sequence axis
        return torch.cat(embeds, dim=1).squeeze(0), torch.cat(labels, dim=0)

    def batch_tensors(self, records, with_answer, left_pad=False):
        """Pad to a batch. position_ids are deliberately not passed to the
        model: transformers derives them from attention_mask (cumsum-1), which
        stays correct with left padding and with the KV cache in generate."""
        items = [self.to_tensors(self.pieces(record, with_answer))
                 for record in records]
        max_len = max(item[0].shape[0] for item in items)
        batch = len(items)
        embeds = torch.zeros(batch, max_len, self.hidden_size,
                             dtype=self.embed_dtype, device=self.device)
        labels = torch.full((batch, max_len), -100, dtype=torch.long,
                            device=self.device)
        mask = torch.zeros(batch, max_len, dtype=torch.long, device=self.device)
        for index, (item_embeds, item_labels) in enumerate(items):
            length = item_embeds.shape[0]
            start = max_len - length if left_pad else 0
            embeds[index, start:start + length] = item_embeds
            labels[index, start:start + length] = item_labels
            mask[index, start:start + length] = 1
        return embeds, labels, mask

    # -------------------------------------------------------------- compute
    def forward_loss(self, records, batch_size):
        total_loss, total_tokens = 0.0, 0
        for batch in chunks(records, batch_size):
            embeds, labels, mask = self.batch_tensors(batch, True)
            out = self.model(inputs_embeds=embeds, attention_mask=mask,
                             labels=labels)
            tokens = int((labels != -100).sum())
            total_loss += float(out.loss) * tokens
            total_tokens += tokens
        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def val_loss(self, records, limit=400):
        self.model.eval()
        subset = records[:limit]
        loss = self.forward_loss(subset, self.args.batch_size)
        if self.args.mode == "lora":
            self.model.train()
        return round(loss, 4)

    @torch.no_grad()
    def generate(self, records, max_new_tokens=48):
        self.model.eval()
        outputs = []
        for batch in chunks(records, self.args.gen_batch):
            embeds, _, mask = self.batch_tensors(batch, False, left_pad=True)
            generated = self.model.generate(
                inputs_embeds=embeds, attention_mask=mask,
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                top_k=None, top_p=None, temperature=None,
                pad_token_id=self.tokenizer.pad_token_id)
            outputs.extend(text.strip() for text in
                           self.tokenizer.batch_decode(generated,
                                                       skip_special_tokens=True))
        if self.args.mode == "lora":
            self.model.train()
        return outputs

    # -------------------------------------------------------------- training
    def build_optimizer(self, records, epochs):
        projector_params = list(self.projector.parameters())
        if self.geom_projector is not None:
            projector_params += list(self.geom_projector.parameters())
        groups = [{"params": projector_params,
                   "lr": self.args.lr,
                   "weight_decay": self.args.weight_decay}]
        lora_params = [p for name, p in self.model.named_parameters()
                       if "lora_" in name]
        if lora_params:
            groups.append({"params": lora_params, "lr": self.args.lora_lr,
                           "weight_decay": self.args.weight_decay})
        optimizer = torch.optim.AdamW(groups)
        steps_per_epoch = math.ceil(
            len(records) / self.args.batch_size / self.args.grad_accum)
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(self.args.warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)
        print("optimizer: projector_lr", self.args.lr, "lora_lr",
              self.args.lora_lr if lora_params else "-",
              "| steps/epoch", steps_per_epoch, "| total steps", total_steps,
              "| warmup", warmup_steps, flush=True)
        return optimizer, scheduler, total_steps

    def train(self, records, eval_records, epochs, log_every=10, out_dir=None):
        optimizer, scheduler, total_steps = self.build_optimizer(records, epochs)
        if self.args.mode == "lora":
            self.model.train()
        history = {"train": [], "val_loss": [], "epoch_seconds": [],
                   "total_steps": total_steps}
        best_val, step = None, 0
        window = []
        for epoch in range(1, epochs + 1):
            started = time.time()
            order = list(range(len(records)))
            random.Random(self.args.seed + epoch).shuffle(order)
            accum = 0
            optimizer.zero_grad(set_to_none=True)
            for batch in chunks(order, self.args.batch_size):
                batch_records = [records[index] for index in batch]
                embeds, labels, mask = self.batch_tensors(batch_records, True)
                out = self.model(inputs_embeds=embeds, attention_mask=mask,
                                 labels=labels)
                (out.loss / self.args.grad_accum).backward()
                window.append(float(out.loss))
                accum += 1
                if accum == self.args.grad_accum:
                    trainable = [p for p in self.model.parameters() if p.requires_grad]
                    trainable += list(self.projector.parameters())
                    torch.nn.utils.clip_grad_norm_(trainable, self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum, step = 0, step + 1
                    if step % log_every == 0:
                        entry = {"step": step, "epoch": epoch,
                                 "loss": round(sum(window) / len(window), 4),
                                 "lr": round(scheduler.get_last_lr()[0], 8)}
                        history["train"].append(entry)
                        print("step {} epoch {} loss {:.4f} lr {:.2e}".format(
                            step, epoch, entry["loss"], entry["lr"]), flush=True)
                        window = []
            if accum:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                trainable += list(self.projector.parameters())
                torch.nn.utils.clip_grad_norm_(trainable, self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            seconds = round(time.time() - started, 1)
            history["epoch_seconds"].append(seconds)
            if eval_records:
                value = self.val_loss(eval_records)
                history["val_loss"].append({"epoch": epoch, "val_loss": value})
                print("epoch {} done in {}s | val_loss {}".format(epoch, seconds, value),
                      flush=True)
                if best_val is None or value < best_val:
                    best_val = value
                    if out_dir:
                        torch.save(dict(self.state_dicts(), epoch=epoch,
                                        val_loss=value, config=vars(self.args)),
                                   os.path.join(out_dir, "best.pt"))
                if out_dir:
                    with open(os.path.join(out_dir, "history.json"), "w",
                              encoding="utf-8") as handle:
                        json.dump(history, handle, indent=2)
            else:
                print("epoch {} done in {}s".format(epoch, seconds), flush=True)
        if out_dir:
            with open(os.path.join(out_dir, "history.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            torch.save(dict(self.state_dicts(), epoch=epochs, val_loss=best_val,
                            config=vars(self.args)),
                       os.path.join(out_dir, "last.pt"))
        return history


def sample_ids_from(records, count, seed=7):
    val_ids = sorted({record["qa_id"] for record in records
                      if record["qa_split"] == "val"})
    rng = random.Random(seed)
    return rng.sample(val_ids, min(count, len(val_ids)))


def save_eval(out_dir, args, records, predictions, result, tag, sample_ids):
    per_row = {row["qa_id"]: row for row in result["rows"]}
    samples = []
    for qa_id in sample_ids:
        if qa_id in per_row:
            samples.append(per_row[qa_id])
    with open(os.path.join(out_dir, "eval_{}.json".format(tag)), "w",
              encoding="utf-8") as handle:
        json.dump({"config": vars(args), "metrics": result["metrics"],
                   "n": result["n"], "rows": result["rows"]}, handle,
                  indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "samples{}_val.json".format(len(sample_ids))), "w",
              encoding="utf-8") as handle:
        json.dump(samples, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "samples_{}.txt".format(tag)), "w",
              encoding="utf-8") as handle:
        for row in result["rows"][:40]:
            handle.write("Q: {}\n".format(row["question"]))
            handle.write("GT: {} {:.1f} m\n".format(
                "{} and {}".format(row["gt_lateral"], row["gt_longitudinal"]),
                row["gt_distance"]))
            handle.write("PRED: {}\n".format(row["prediction"]))
            handle.write("parsed: lateral={} longitudinal={} distance={} exact_ok={}\n\n".format(
                row["pred_lateral"], row["pred_longitudinal"], row["pred_distance"],
                row["exact_ok"]))
    print("saved eval_{}.json and samples to {}".format(tag, out_dir), flush=True)


def geometry_note_lines(args, trainer):
    """Experiment-log lines for the geometry-input runs (directive: record the
    expansion / new-parameter strategy explicitly)."""
    if args.geom == "none":
        return []
    lines = [
        "geometry mode: {} | projector input dim {} (token 288 + geometry {})".format(
            args.geom, 288 + GEOM_DIM[args.geom], GEOM_DIM[args.geom]),
        "geometry source: {}".format(
            "relative geometry from PREDICTED centers (target - reference)"
            if args.geom == "relative" else
            ("GT center/size (ORACLE diagnostic; not a final-model result)"
             if args.oracle_center else
             "predicted by the 3EED grounding model (pred_center/pred_size export)")),
        "geometry normalisation: train-split mean/std only (val/test reuse train "
        "statistics); stats: {}".format(json.dumps(trainer.geom_stats)),
    ]
    if args.geom == "box":
        lines.append("box mode note: the 3EED export does not record predicted "
                     "yaw, so the box input is center+size (6-d); predicted yaw "
                     "is not available without re-exporting")
    if trainer.expanded_from is not None:
        lines.append("projector expansion: kept all {} trained columns of the "
                     "first layer, appended {} zero-initialised columns (new "
                     "parameters only); later layers unchanged, LoRA loaded "
                     "unchanged".format(trainer.expanded_from,
                                        288 + GEOM_DIM[args.geom] - trainer.expanded_from))
    elif args.geom == "relative":
        lines.append("geometry projector: new independent module (5 -> 1024 -> "
                     "hidden), randomly initialised; the token projector and LoRA "
                     "are loaded unchanged from the C checkpoint")
    else:
        lines.append("projector loaded unchanged (same input dimension)")
    return lines


def write_notes(out_dir, args, history, records, token_checks, extra):
    lines = [
        "# Experiment notes (auto-generated by train_qa.py)",
        "",
        "## Standing notes from the directive",
        "- 3EED's grounding structure, checkpoint and settings come from 3EED",
        "  (feature_name=last_query_features, token_dim=288, selection=",
        "  contrastive_top1_annotated_text_span); the grounding model is not",
        "  loaded in this script, tokens are precomputed exports, so it has no",
        "  gradient path (frozen by construction).",
        "- 3EED has no LLM and no LoRA. Qwen2.5-7B-Instruct, the projector and",
        "  LoRA are our new downstream QA settings.",
        "- The 1e-4 learning rate matches the magnitude 3EED uses for its",
        "  non-point-encoder modules; it must NOT be quoted as 3EED's LoRA",
        "  setting (3EED has none).",
        "",
        "## Actual configuration",
        "- mode: {}".format(args.mode),
        "- data: {} ({} records; train {}/val {}/test {} by scene, from".format(
            args.data, len(records),
            sum(1 for r in records if r["qa_split"] == "train"),
            sum(1 for r in records if r["qa_split"] == "val"),
            sum(1 for r in records if r["qa_split"] == "test")),
        "  qa_waymo_clean.jsonl; scene-level split, no scene crosses splits).",
        "- model: {} (bf16, hidden_size read from config)".format(args.model_path),
        "- projector: Linear(288, 1024) -> GELU -> Linear(1024, hidden_size),",
        "  shared by target and reference tokens.",
        "- per-device batch {}, grad_accum {}, effective batch {}".format(
            args.batch_size, args.grad_accum,
            args.batch_size * args.grad_accum),
        "- epochs {}, projector_lr {}, lora_lr {}, weight_decay {}, warmup_ratio {},".format(
            args.epochs, args.lr, args.lora_lr, args.weight_decay,
            args.warmup_ratio),
        "  cosine schedule, max_grad_norm {}, bf16, seed {}".format(
            args.max_grad_norm, args.seed),
        "- lora: {} (r=16, alpha=32, dropout=0.05, q/k/v/o_proj, bias=none)".format(
            "on" if args.mode == "lora" else "off"),
        "- init projector: {}".format(args.init_projector or "from scratch"),
        "- eval code: qa_eval.py (shared with Experiment A / GT baseline);",
        "  splits and tokens identical across experiments B and C.",
        "- labels: answer text + terminating <|im_end|>; every prompt token -100.",
        "- token row identity checked for {} records at startup (0 mismatches).".format(
            token_checks),
        "- val loss history: {}".format(json.dumps(history.get("val_loss", []))),
        "- epoch seconds: {}".format(json.dumps(history.get("epoch_seconds", []))),
    ]
    if extra:
        lines += ["", "## Extra"] + ["- " + item for item in extra]
    with open(os.path.join(out_dir, "experiment_notes.md"), "w",
              encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True,
                        choices=["projector", "lora", "eval"])
    parser.add_argument("--data",
                        default="/root/autodl-tmp/3eed_data/qa_v1/qa_waymo_clean.jsonl")
    parser.add_argument("--model-path",
                        default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--init-projector", default=None)
    parser.add_argument("--tokens-root", default=None,
                        help="root for relative token_path values (default: "
                             "directory of --data)")
    parser.add_argument("--with-lora", action="store_true",
                        help="eval mode: load the LoRA part of the checkpoint")
    parser.add_argument("--no-tokens", action="store_true",
                        help="text-only prompt (baseline)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=0, help="0 = all")
    parser.add_argument("--max-eval", type=int, default=0, help="0 = all")
    parser.add_argument("--eval-splits", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-n", type=int, default=20)
    parser.add_argument("--smoke", action="store_true",
                        help="gradient/loss/output checks on a small subset, then exit")
    parser.add_argument("--geom", choices=["none", "center", "box", "relative"],
                        default="none",
                        help="extra geometry input: 'center'/'box' extend the "
                             "projector input, 'relative' adds a third soft token")
    parser.add_argument("--pred-geometry", default=None,
                        help="npz with grounding-predicted (and GT) geometry "
                             "aligned to the token rows")
    parser.add_argument("--oracle-center", action="store_true",
                        help="diagnostic only: feed GT center/size instead of "
                             "the grounding prediction (never a final result)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.out_dir is None:
        base = os.path.dirname(args.data)
        name = "text_only" if args.no_tokens else args.mode
        args.out_dir = os.path.join(base, "runs", name)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.data, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    train_records = [r for r in records if r["qa_split"] == "train"]
    eval_records = [r for r in records
                    if r["qa_split"] in args.eval_splits.split(",")]
    if args.max_train:
        train_records = train_records[:args.max_train]
    if args.max_eval:
        eval_records = eval_records[:args.max_eval]
    print("records:", len(records), "| train", len(train_records),
          "| eval({})".format(args.eval_splits), len(eval_records), flush=True)
    print("device", torch.cuda.get_device_name(0), flush=True)

    trainer = QATrainer(args)
    checked = trainer.check_token_identity(records)
    print("token row identity ok on", checked, "records", flush=True)
    if args.geom != "none":
        trainer.setup_geometry(records)
        print("geometry alignment ok on", checked, "records | mode", args.geom,
              "| source", "GT (oracle diagnostic)" if args.oracle_center
              and args.geom != "relative" else "predicted", flush=True)
        with open(os.path.join(args.out_dir, "geom_stats.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"mode": args.geom, "oracle_center": bool(args.oracle_center),
                       "stats": trainer.geom_stats},
                      handle, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ smoke
    if args.smoke:
        smoke_records = train_records[:max(args.max_train, 64)]
        record = smoke_records[0]
        embeds, labels = trainer.to_tensors(trainer.pieces(record, True))
        supervised = int((labels != -100).sum())
        expected = len(trainer.encode(record["answer"])) + \
            len(trainer.encode(ASSISTANT_SUFFIX))
        print("smoke: sequence length", embeds.shape[0],
              "| supervised tokens", supervised, "| expected", expected)
        assert supervised == expected, "labels must cover exactly the answer part"
        if not args.no_tokens:
            assert embeds.shape[0] > supervised
        gen_embeds, _ = trainer.to_tensors(trainer.pieces(record, False))
        assert gen_embeds.shape[0] < embeds.shape[0]
        assert torch.equal(embeds[:gen_embeds.shape[0]], gen_embeds), \
            "generation context must be exactly the training prefix"
        print("smoke: generation context == training prefix ({} of {} tokens)"
              .format(gen_embeds.shape[0], embeds.shape[0]))
        if args.geom != "none":
            assert trainer.geom_stats is not None
            if trainer.expanded_from is not None:
                new_cols = trainer.projector.net[0].weight.detach()[
                    :, trainer.expanded_from:]
                assert float(new_cols.abs().sum()) == 0.0, \
                    "expanded projector columns must start at zero"
                print("smoke: expanded projector new columns are exactly zero at init",
                      flush=True)
            if trainer.geom_projector is not None:
                assert args.geom == "relative"
        trainer.model.zero_grad(set_to_none=True)
        trainer.projector.zero_grad(set_to_none=True)
        if trainer.geom_projector is not None:
            trainer.geom_projector.zero_grad(set_to_none=True)
        out = trainer.model(inputs_embeds=embeds.unsqueeze(0),
                            labels=labels.unsqueeze(0))
        out.loss.backward()
        projector_grad = sum(float(p.grad.abs().sum())
                             for p in trainer.projector.parameters()
                             if p.grad is not None)
        print("smoke: loss", round(float(out.loss), 4),
              "| projector grad abs-sum", projector_grad)
        assert projector_grad > 0, "projector received no gradient"
        assert trainer.embed.weight.grad is None, "embedding table must stay frozen"
        frozen_with_grad = [name for name, p in trainer.model.named_parameters()
                            if p.grad is not None and not p.requires_grad]
        assert not frozen_with_grad, "frozen model params received gradients"
        print("smoke: frozen params with gradient:", len(frozen_with_grad))
        if trainer.expanded_from is not None:
            new_grad = float(trainer.projector.net[0].weight.grad[
                :, trainer.expanded_from:].abs().sum())
            print("smoke: new geometry columns grad abs-sum", new_grad)
            assert new_grad > 0, "new geometry columns received no gradient"
        if trainer.geom_projector is not None:
            geom_grad = sum(float(p.grad.abs().sum())
                            for p in trainer.geom_projector.parameters()
                            if p.grad is not None)
            print("smoke: geometry projector grad abs-sum", geom_grad)
            assert geom_grad > 0, "geometry projector received no gradient"
        if trainer.lora_param_names:
            lora_grad = sum(float(p.grad.abs().sum())
                            for name, p in trainer.model.named_parameters()
                            if "lora_" in name and p.grad is not None)
            print("smoke: lora grad abs-sum", lora_grad)
            assert lora_grad > 0

        before = trainer.generate(eval_records[:3] if eval_records else smoke_records[:3])
        trainer.train(smoke_records, [], epochs=8, log_every=2)
        after = trainer.generate(smoke_records[:3])
        for index in range(3):
            print("smoke sample", index, "GT :", smoke_records[index]["answer"])
            print("smoke sample", index, "before:", before[index])
            print("smoke sample", index, "after :", after[index])
        print("smoke ok", flush=True)
        return

    # ------------------------------------------------------------------- eval
    if args.mode == "eval":
        if not args.init_projector and not args.no_tokens:
            raise SystemExit("eval mode needs --init-projector or --no-tokens")
        predictions = trainer.generate(eval_records)
        result = evaluate(eval_records, predictions, tag=args.eval_splits)
        save_eval(args.out_dir, args, eval_records, predictions, result,
                  args.eval_splits, sample_ids_from(records, args.samples_n))
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
        return

    # --------------------------------------------------------------- training
    history = trainer.train(train_records, eval_records, args.epochs,
                            log_every=10, out_dir=args.out_dir)
    best_path = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)
    predictions = trainer.generate(eval_records)
    result = evaluate(eval_records, predictions, tag=args.eval_splits)
    save_eval(args.out_dir, args, eval_records, predictions, result,
              args.eval_splits, sample_ids_from(records, args.samples_n))
    extra = ["generated with train_qa.py --mode {}".format(args.mode)]
    extra += geometry_note_lines(args, trainer)
    write_notes(args.out_dir, args, history, records, checked, extra=extra)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False), flush=True)
    print("out dir", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
