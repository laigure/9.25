# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Shared utilities for all main scripts."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
import ipdb


class DistributedTargetCountSampler(Sampler):
    """Draw an exact per-epoch quota for each scene target count.

    Minority groups are cycled through shuffled permutations before any item
    is repeated, so all scarce N=3/4 scenes receive comparable exposure.
    The combined roster is then shuffled and partitioned across DDP ranks.
    """

    def __init__(self, dataset, samples_per_count, seed=0,
                 num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.samples_per_count = {
            count: int(value)
            for count, value in enumerate(samples_per_count, 1)
            if int(value) > 0
        }
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.groups = {}
        for index, anno in enumerate(dataset.annos):
            count = int(anno.get("target_count", len(
                anno.get("boxes_info", {}).get("bbox3d", []))))
            self.groups.setdefault(count, []).append(index)
        missing = [count for count in self.samples_per_count
                   if not self.groups.get(count)]
        if missing:
            raise ValueError(f"balanced sampler requested empty target groups: {missing}")
        requested = sum(self.samples_per_count.values())
        self.num_samples = int(np.ceil(requested / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        if self.rank == 0:
            print("Balanced target-count sampler | source {} | epoch quotas {} | "
                  "total {} | per rank {}".format(
                      {k: len(v) for k, v in sorted(self.groups.items())},
                      self.samples_per_count, requested, self.num_samples),
                  flush=True)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        roster = []
        for count, wanted in sorted(self.samples_per_count.items()):
            source = torch.as_tensor(self.groups[count], dtype=torch.long)
            selected = []
            while len(selected) < wanted:
                order = torch.randperm(len(source), generator=generator)
                take = min(wanted - len(selected), len(source))
                selected.extend(source[order[:take]].tolist())
            roster.extend(selected)
        order = torch.randperm(len(roster), generator=generator).tolist()
        roster = [roster[index] for index in order]
        if len(roster) < self.total_size:
            roster += roster[:self.total_size - len(roster)]
        assert len(roster) == self.total_size
        return iter(roster[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--num_target", type=int, default=256, help="Proposal number")
    parser.add_argument("--text_token_budget", type=int, default=256,
                        choices=[256, 512],
                        help="RoBERTa/positive-map token budget; scene-level multi-target uses 512")
    parser.add_argument("--sampling", default="kps", type=str, help="Query points sampling method (kps, fps)")

    # Transformer
    parser.add_argument("--num_encoder_layers", default=3, type=int)
    parser.add_argument("--num_decoder_layers", default=6, type=int)
    parser.add_argument("--self_position_embedding", default="loc_learned", type=str, help="(none, xyz_learned, loc_learned)")
    parser.add_argument("--self_attend", action="store_true")

    # Loss
    parser.add_argument("--query_points_obj_topk", default=4, type=int)
    parser.add_argument("--use_contrastive_align", action="store_true")
    parser.add_argument("--use_soft_token_loss", action="store_true")
    parser.add_argument("--detect_intermediate", action="store_true")
    parser.add_argument("--joint_det", action="store_true")

    # Data
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size during training")
    parser.add_argument("--dataset", type=str, default=["quad"], nargs="+", help="list of datasets to train on")
    parser.add_argument("--test_dataset", type=str, default=["sr3d"], nargs="+", )
    parser.add_argument("--data_root", default="./", help="Root directory for datasets")
    parser.add_argument("--split_dir", default="data/splits", help="Directory containing split files (train.txt, val.txt)")
    parser.add_argument("--use_height", action="store_true", help="Use height signal in input.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--butd", action="store_true")
    parser.add_argument("--butd_gt", action="store_true")
    parser.add_argument("--butd_cls", action="store_true")
    parser.add_argument("--augment_det", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--balance_target_counts", action="store_true",
                        help="Use exact per-epoch quotas grouped by scene target count")
    parser.add_argument("--target_count_samples", type=int, nargs=5,
                        default=[0, 0, 0, 0, 0], metavar=("N1", "N2", "N3", "N4", "N5"),
                        help="Per-epoch samples for scenes with 1..5 targets")

    # Training
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--eval_split", choices=["train", "val", "test"], default="val",
                        help="Validation split during training, or chosen split with --eval; "
                             "train is for exporting training-split tokens, never for reporting")
    parser.add_argument("--init_checkpoint_path", type=str, default="",
                        help="Initialize model weights only; start a fresh optimizer and epoch schedule")
    parser.add_argument("--allow_partial_init", action="store_true",
                        help="Copy compatible/overlapping checkpoint tensors when an output head was expanded")
    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument("--optimizer", type=str, default="adamW")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument("--lr-scheduler", type=str, default="step", choices=["step", "cosine"])
    parser.add_argument("--lr_decay_epochs", type=int, default=[280, 340], nargs="+", help="when to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="for step scheduler. decay rate for lr")
    parser.add_argument("--clip_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument("--bn_momentum", type=float, default=0.1)
    parser.add_argument("--syncbn", action="store_true")
    parser.add_argument("--warmup-epoch", type=int, default=-1)
    parser.add_argument("--warmup-multiplier", type=int, default=100)
    parser.add_argument("--flag", default=None, help="a flag to identify the experiment")

    # io
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Model checkpoint path",
    )
    parser.add_argument("--export_qa_tokens", default=None,
                        help="Directory for selected 288-D QA tokens during --eval")
    parser.add_argument("--log_dir", default="log", help="Dump dir to save model checkpoint")
    parser.add_argument("--print_freq", type=int, default=10)  # batch-wise
    parser.add_argument("--save_freq", type=int, default=10)  # epoch-wise
    parser.add_argument("--val_freq", type=int, default=5)  # epoch-wise

    # others
    parser.add_argument("--local_rank", type=int, help="local rank for DistributedDataParallel")
    parser.add_argument("--ap_iou_thresholds", type=float, default=[0.25, 0.5], nargs="+", help="A list of AP IoU thresholds")
    parser.add_argument("--rng_seed", type=int, default=0, help="manual seed")
    parser.add_argument(
        "--debug",
        action="store_true",
        # default=True,
        # default=False,
        help="try to overfit few samples",
    )
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--pp_checkpoint", default=None)
    parser.add_argument("--reduce_lr", action="store_true")

    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train
    
    # Set log_dir based on eval mode
    if args.eval:
        # For evaluation: use checkpoint_path's parent directory + /evaluation
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        
        test_datasets = "_".join(args.test_dataset)
        exp_name = f"Val_{test_datasets}"
        args.log_dir = os.path.join(checkpoint_dir, "eval", exp_name, time.strftime('%m%d_%H%M'))
    else:
        # For training: use Train_<datasets>_Val_<test_datasets> format
        train_datasets = "_".join(args.dataset)
        test_datasets = "_".join(args.test_dataset)
        exp_name = f"Train_{train_datasets}_Val_{test_datasets}"
        
        if args.flag is not None:
            exp_name = f"{exp_name}/{args.flag}"
        else:
            exp_name = f"{exp_name}/{time.strftime('%m%d_%H%M')}"
        
        args.log_dir = os.path.join(args.log_dir, exp_name)
    
    # Add debug suffix if needed
    if args.debug:
        args.num_workers = 0
        args.log_dir =  os.path.join(args.log_dir, "debug")

    os.makedirs(args.log_dir, exist_ok=True)
    print(f"\033[93mLog directory: {args.log_dir}\033[0m")
    # exit()
    return args


def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    try:
        args.start_epoch = int(checkpoint["epoch"]) + 1
    except Exception:
        args.start_epoch = 0
    model.load_state_dict(checkpoint["model"], strict=True)
    if not args.eval and not args.reduce_lr:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    print("=> loaded successfully '{}' (epoch {})".format(args.checkpoint_path, checkpoint["epoch"]))

    del checkpoint
    torch.cuda.empty_cache()


def backup_python_files(self, backup_dirs, target_dir):
    """
    Backup specified python files/folders to target directory.

    Args:
        backup_dirs (list): List of directories or files to backup.
        target_dir (str): Where to store the backup.
    """
    for item in backup_dirs:
        if os.path.isdir(item):
            shutil.copytree(item, os.path.join(target_dir, item), dirs_exist_ok=True)
        elif item.endswith(".py"):
            shutil.copy2(item, target_dir)


def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {"config": args, "save_path": "", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch}
        spath = os.path.join(args.log_dir, f"ckpt_epoch_{epoch}.pth")
        state["save_path"] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split("/")[-1]
        
        self.debug = args.debug

        # Create logger
        self.logger = setup_logger(output=args.log_dir, distributed_rank=dist.get_rank(), name=name)

        self.log_dir = args.log_dir
        
        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, "w") as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

        # Backup used python files
        # Main process saves config and backs up code
        if dist.get_rank() == 0:

            # Save config
            path = os.path.join(args.log_dir, "config.json")
            with open(path, "w") as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

            # Backup code
            backup_files = ["main_utils.py", "prepare_data.py", "train_dist_mod.py"]
            backup_dirs = ["models", "src", "utils", "scripts"]
            backup_path = os.path.join(args.log_dir, "code_backup")
            os.makedirs(backup_path, exist_ok=True)
            self.backup_code(backup_files, backup_dirs, backup_path)
            self.logger.info(f"Code backup completed at {backup_path}")

        # import pdb

        # pdb.set_trace()
        # Initialize TensorBoard only in main process
        if dist.get_rank() == 0:
            tb_logdir = os.path.join(args.log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tb_logdir)
            self.logger.info(f"TensorBoard logs at {tb_logdir}")
        else:
            self.tb_writer = None

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset

    def backup_code(self, files, dirs, target_dir):
        def ignore_non_py_files(dir, files):
            return [f for f in files if not (f.endswith(".py") or f.endswith(".sh") or os.path.isdir(os.path.join(dir, f)))]

        # Copy single .py files
        for file in files:
            src_path = os.path.abspath(file)
            if os.path.exists(src_path) and src_path.endswith(".py"):
                shutil.copy2(src_path, target_dir)
            else:
                print(f"[Warning] File not found or not a .py file: {src_path}")

        # Copy .py files in directories
        for dir_ in dirs:
            src_path = os.path.abspath(dir_)
            dst_path = os.path.join(target_dir, os.path.basename(dir_))
            if os.path.exists(src_path):
                shutil.copytree(src_path, dst_path, ignore=ignore_non_py_files, dirs_exist_ok=True)
            else:
                print(f"[Warning] Directory not found: {src_path}")

    def get_loaders(self, args):
        """Initialize data loaders."""

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        
        # Only create train_loader if not in eval mode
        if args.eval or train_dataset is None:
            train_loader = None
        else:
            if args.balance_target_counts:
                train_sampler = DistributedTargetCountSampler(
                    train_dataset, args.target_count_samples,
                    seed=args.rng_seed)
            else:
                train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g,
            )
        
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g,
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        losses = ["boxes", "labels"]
        if args.use_contrastive_align:
            losses.append("contrastive_align")
        set_criterion = SetCriterion(matcher=matcher, losses=losses, eos_coef=0.1, temperature=0.07)
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone_net" not in n and "text_encoder" not in n and p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if "backbone_net" in n and p.requires_grad], "lr": args.lr_backbone},
            {"params": [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad], "lr": args.text_encoder_lr},
        ]
        optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        
        # Only check training dataset if not in eval mode
        if not args.eval and train_loader is not None:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
            assert len(train_loader.dataset) > 0, f"training set is empty"
        
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")
        assert len(test_loader.dataset) > 0, f"test set is empty"
        
        # Get model
        # ipdb.set_trace()
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler
        if train_loader is not None:
            scheduler = get_scheduler(optimizer, len(train_loader), args)
        else:
            # In eval mode, create a dummy scheduler
            scheduler = get_scheduler(optimizer, 1, args)

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)  # , find_unused_parameters=True

        # Initialize a new training run from pretrained weights without resuming its epoch.
        if args.init_checkpoint_path:
            checkpoint = torch.load(args.init_checkpoint_path, map_location="cpu")
            if args.allow_partial_init:
                current = model.state_dict()
                source = checkpoint["model"]
                copied, expanded, skipped = [], [], []
                for key, target in current.items():
                    if key not in source:
                        skipped.append(key)
                        continue
                    value = source[key]
                    if value.shape == target.shape:
                        current[key] = value
                        copied.append(key)
                    elif value.ndim == target.ndim:
                        merged = target.clone()
                        slices = tuple(slice(0, min(a, b))
                                       for a, b in zip(value.shape, target.shape))
                        merged[slices] = value[slices]
                        current[key] = merged
                        expanded.append((key, tuple(value.shape), tuple(target.shape)))
                    else:
                        skipped.append(key)
                model.load_state_dict(current, strict=True)
                self.logger.info(
                    "Partially initialized from '%s': %d exact, %d expanded, %d skipped",
                    args.init_checkpoint_path, len(copied), len(expanded), len(skipped))
                for item in expanded:
                    self.logger.info("Expanded checkpoint tensor: %s %s -> %s", *item)
            else:
                model.load_state_dict(checkpoint["model"], strict=True)
                self.logger.info("Initialized model weights from '%s'", args.init_checkpoint_path)
            del checkpoint

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
            self.logger.info("Loaded checkpoint from '{}'".format(args.checkpoint_path))

        # Just eval and end execution
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(args.start_epoch, test_loader, model, criterion, set_criterion, args)
            return

        # Training loop
        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()
            self.train_one_epoch(epoch, train_loader, model, criterion, set_criterion, optimizer, scheduler, args)
            self.logger.info(
                "epoch {}, total time {:.2f}, "
                "lr_base {:.5f}, lr_pointnet {:.5f}".format(epoch, (time.time() - tic), optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
            )
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:  # save model
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Test evaluation.......")
                self.evaluate_one_epoch(epoch, test_loader, model, criterion, set_criterion, args)

        # Training is over, evaluate
        save_checkpoint(args, "last", model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, "ckpt_epoch_last.pth")
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(args.max_epoch, test_loader, model, criterion, set_criterion, args)
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        return {"point_clouds": batch_data["point_clouds"].float(), "text": batch_data["utterances"]}

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(end_points, args.num_decoder_layers, set_criterion, query_points_obj_topk=args.query_points_obj_topk)
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if "loss" in key or "acc" in key or "ratio" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict

    def train_one_epoch(self, epoch, train_loader, model, criterion, set_criterion, optimizer, scheduler, args):
        """
        Run a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode

        # Loop over batches
        for batch_idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train epoch {}".format(epoch)):

            if self.debug and batch_idx > 10:
                self.logger.info("debug mode")
                break

            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            inputs = self._get_inputs(batch_data)

            # Forward pass
            end_points = model(inputs)

            # Compute loss and gradients, update parameters.
            for key in batch_data:
                assert key not in end_points
                end_points[key] = batch_data[key]
            loss, end_points = self._compute_loss(end_points, criterion, set_criterion, args)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                stat_dict["grad_norm"] = grad_total_norm
            optimizer.step()
            scheduler.step()

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            if self.tb_writer:
                global_step = epoch * len(train_loader) + batch_idx
                for key in sorted(stat_dict.keys()):
                    if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key:
                        self.tb_writer.add_scalar(f"Train/{key}", stat_dict[key] / args.print_freq, global_step)

            if (batch_idx + 1) % args.print_freq == 0:

                # Terminal logs
                self.logger.info(f"Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  ")
                self.logger.info(
                    "".join(
                        [
                            f"{key} {stat_dict[key] / args.print_freq:.4f} \t"
                            for key in sorted(stat_dict.keys())
                            if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key
                        ]
                    )
                )

                # Reset statistics
                for key in sorted(stat_dict.keys()):
                    stat_dict[key] = 0

    @torch.no_grad()
    def _main_eval_branch(self, epoch, batch_idx, batch_data, test_loader, model, stat_dict, criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # Forward pass
        if args.eval and args.export_qa_tokens:
            end_points = model(inputs, return_query_features=True)
        else:
            end_points = model(inputs)

        # Compute loss
        for key in batch_data:
            assert key not in end_points
            end_points[key] = batch_data[key]
        _, end_points = self._compute_loss(end_points, criterion, set_criterion, args)
        for key in end_points:
            if "pred_size" in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f"Eval: [{batch_idx + 1}/{len(test_loader)}]  ")
            self.logger.info(
                "".join(
                    [
                        f"{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t"
                        for key in sorted(stat_dict.keys())
                        if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key
                    ]
                )
            )

        if self.tb_writer:
            for key in sorted(stat_dict.keys()):
                if "loss" in key and "proposal_" not in key and "last_" not in key and "head_" not in key:
                    self.tb_writer.add_scalar(f"Eval/{key}", stat_dict[key] / (float(batch_idx + 1)), epoch * len(test_loader) + batch_idx)

        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader, model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
