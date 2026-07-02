#!/usr/bin/env python3
"""Train and evaluate three independent cross-validation folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

import config
from cv_split_utils import assign_pdb_group_folds, assign_sample_folds
from data_loader import PrecomputedDataset, filter_none_collate
from model import ESM_FoldX_DDAffinity, ESM_RAAD_FoldX_DDAffinity
from train_utils import evaluate_model, train_model


logger = logging.getLogger("three_fold_cv")


class DistributedEvalSampler(Sampler[int]):
    """Shard validation indices across ranks without padding or duplication."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train three independent cross-validation models."
    )
    parser.add_argument("--num-folds", type=int, default=config.CV_NUM_FOLDS)
    parser.add_argument("--seed", type=int, default=config.CV_RANDOM_SEED)
    parser.add_argument(
        "--split-mode",
        choices=("sample", "pdb_group"),
        default=getattr(config, "CV_SPLIT_MODE", "sample"),
        help=(
            "sample: random sample-level folds, so the same PDB may appear in "
            "train and validation; pdb_group: keep each pdb_id within one fold."
        ),
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument(
        "--patience", type=int, default=config.EARLY_STOPPING_PATIENCE
    )
    parser.add_argument("--output-dir", type=Path, default=Path(config.CV_OUTPUT_DIR))
    parser.add_argument("--precomputed-dir", type=Path, default=Path(config.PRECOMPUTED_DIR))
    parser.add_argument("--batch-size-per-rank", type=int, default=None)
    parser.add_argument("--num-workers-per-rank", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--save-all-epochs", action="store_true")
    parser.add_argument("--force", action="store_true", help="Retrain completed folds.")
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Validate and write the three-fold manifest without training.",
    )
    return parser.parse_args()


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1, config.DEVICE

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    torch.distributed.init_process_group(backend=backend)
    return True, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed:
        torch.distributed.barrier()


def setup_logging(output_dir: Path, rank: int) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.insert(0, logging.FileHandler(output_dir / "training_3fold.log"))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_runtime_parameters(args, world_size: int) -> Tuple[int, int, int]:
    if args.batch_size_per_rank is not None:
        batch_size = args.batch_size_per_rank
    elif world_size > 1:
        batch_size = getattr(config, "DDP_BATCH_SIZE_PER_RANK", 2)
    else:
        batch_size = config.BATCH_SIZE

    if args.num_workers_per_rank is not None:
        num_workers = args.num_workers_per_rank
    elif world_size > 1:
        num_workers = getattr(
            config,
            "DDP_NUM_WORKERS_PER_RANK",
            max(1, config.NUM_WORKERS // world_size),
        )
    else:
        num_workers = config.NUM_WORKERS

    if args.gradient_accumulation_steps is not None:
        grad_accum = args.gradient_accumulation_steps
    elif world_size > 1 and getattr(
        config, "DDP_GRADIENT_ACCUMULATION_STEPS", None
    ) is not None:
        grad_accum = config.DDP_GRADIENT_ACCUMULATION_STEPS
    else:
        target_batch = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
        global_micro_batch = max(1, batch_size * world_size)
        grad_accum = max(1, round(target_batch / global_micro_batch))
    return int(batch_size), int(num_workers), int(grad_accum)


def build_splits(
    dataset: PrecomputedDataset, num_folds: int, seed: int, split_mode: str
) -> Dict[str, object]:
    sample_records = []
    group_ids = []
    for position in range(len(dataset)):
        sample = dataset[position]
        pdb_id = str(sample.get("pdb_id", "")).strip()
        if not pdb_id:
            raise KeyError(f"Precomputed sample at dataset position {position} lacks pdb_id.")
        sample_index = int(dataset.indices[position])
        group_ids.append(pdb_id)
        sample_records.append(
            {
                "dataset_position": position,
                "sample_index": sample_index,
                "pdb_id": pdb_id,
            }
        )

    if split_mode == "pdb_group":
        folds, fold_groups = assign_pdb_group_folds(group_ids, num_folds, seed)
    elif split_mode == "sample":
        folds = assign_sample_folds(len(dataset), num_folds, seed)
        fold_groups = [
            sorted({group_ids[position] for position in positions})
            for positions in folds
        ]
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    assignment_by_position = {}
    for fold_index, positions in enumerate(folds):
        for position in positions:
            assignment_by_position[position] = fold_index
    for record in sample_records:
        record["fold"] = assignment_by_position[record["dataset_position"]] + 1

    return {
        "num_folds": num_folds,
        "seed": seed,
        "split_mode": split_mode,
        "folds": folds,
        "fold_groups": fold_groups,
        "samples": sample_records,
    }


def distribute_split_payload(
    dataset: PrecomputedDataset,
    num_folds: int,
    seed: int,
    split_mode: str,
    distributed: bool,
    rank: int,
) -> Dict[str, object]:
    payload = build_splits(dataset, num_folds, seed, split_mode) if rank == 0 else None
    if distributed:
        objects = [payload]
        torch.distributed.broadcast_object_list(objects, src=0)
        payload = objects[0]
    assert payload is not None
    return payload


def write_split_manifest(payload: Dict[str, object], output_dir: Path) -> None:
    samples = payload["samples"]
    with (output_dir / "fold_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_position", "sample_index", "pdb_id", "fold"],
        )
        writer.writeheader()
        writer.writerows(samples)

    summary = {
        "num_folds": payload["num_folds"],
        "seed": payload["seed"],
        "split_mode": payload["split_mode"],
        "folds": [
            {
                "fold": index + 1,
                "samples": len(payload["folds"][index]),
                "pdb_groups": len(payload["fold_groups"][index]),
                "unique_pdbs": len(payload["fold_groups"][index]),
            }
            for index in range(int(payload["num_folds"]))
        ],
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    sampler=None,
    shuffle: bool = False,
    device: torch.device | None = None,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "collate_fn": filter_none_collate,
        "num_workers": num_workers,
        "pin_memory": device is not None and device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


def build_model(device: torch.device):
    if config.USE_DYNAMIC_MODELING:
        model = ESM_RAAD_FoldX_DDAffinity(
            esm_model_name=config.ESM_MODEL_NAME,
            hidden_dim=config.HIDDEN_DIM,
            raad_hidden_dim=config.RAAD_HIDDEN_DIM,
            raad_layers=config.RAAD_LAYERS,
            dropout=config.DROPOUT,
            edge_types=config.EDGE_TYPES,
            rball_radius=config.RBALL_RADIUS,
            knn_k=config.KNN_K,
            use_atom_features=config.USE_ATOM_FEATURES,
            use_precomputed_esm=getattr(config, "USE_PRECOMPUTED_ESM", False),
            local_radius=getattr(config, "MUTATION_LOCAL_RADIUS", 10.0),
            esm_mutation_window_radius=getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8),
        )
    else:
        model = ESM_FoldX_DDAffinity(
            esm_model_name=config.ESM_MODEL_NAME,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
            use_precomputed_esm=getattr(config, "USE_PRECOMPUTED_ESM", False),
        )
    return model.to(device)


def get_model_state_dict(model) -> Dict[str, torch.Tensor]:
    return (model.module if hasattr(model, "module") else model).state_dict()


def load_state_dict_flexible(model, state_dict) -> None:
    target = model.module if hasattr(model, "module") else model
    candidates = [
        state_dict,
        OrderedDict(
            (key.replace("module.", "", 1), value)
            for key, value in state_dict.items()
        ),
    ]
    for candidate in candidates:
        try:
            target.load_state_dict(candidate, strict=True)
            return
        except RuntimeError:
            continue
    raise RuntimeError("Failed to strictly load fold checkpoint.")


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_scheduler(optimizer, epochs: int):
    if getattr(config, "LR_SCHEDULE", "plateau") == "cosine":
        warmup_epochs = getattr(config, "WARMUP_EPOCHS", 0)

        def lr_lambda(epoch):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            denominator = max(epochs - warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / denominator
            min_ratio = config.MIN_LR / config.LEARNING_RATE
            return min_ratio + (1.0 - min_ratio) * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR,
    )


def regression_metrics(labels: Sequence[float], predictions: Sequence[float]):
    labels_array = np.asarray(labels, dtype=float)
    predictions_array = np.asarray(predictions, dtype=float)
    residual = predictions_array - labels_array
    if len(labels_array) > 1:
        pearson = float(pearsonr(labels_array, predictions_array)[0])
        spearman = float(spearmanr(labels_array, predictions_array)[0])
    else:
        pearson = 0.0
        spearman = 0.0
    return {
        "samples": int(len(labels_array)),
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(np.mean(np.abs(residual))) if len(residual) else 0.0,
        "rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else 0.0,
    }


def gather_validation_positions(
    local_positions: List[int], distributed: bool
) -> List[int]:
    if not distributed:
        return local_positions
    gathered = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, local_positions)
    return [position for rank_positions in gathered for position in rank_positions]


def write_fold_predictions(
    path: Path,
    dataset: PrecomputedDataset,
    dataset_positions: Sequence[int],
    labels: Sequence[float],
    predictions: Sequence[float],
    fold_number: int,
) -> None:
    if not (len(dataset_positions) == len(labels) == len(predictions)):
        raise RuntimeError(
            "Validation positions, labels, and predictions have inconsistent lengths."
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "dataset_position",
                "sample_index",
                "pdb_id",
                "label_ddg",
                "pred_ddg",
            ],
        )
        writer.writeheader()
        for position, label, prediction in zip(
            dataset_positions, labels, predictions
        ):
            sample = dataset[position]
            writer.writerow(
                {
                    "fold": fold_number,
                    "dataset_position": position,
                    "sample_index": dataset.indices[position],
                    "pdb_id": sample["pdb_id"],
                    "label_ddg": float(label),
                    "pred_ddg": float(prediction),
                }
            )


def validation_split_hash(indices: Sequence[int]) -> str:
    payload = ",".join(str(index) for index in sorted(indices)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def train_fold(
    fold_index: int,
    folds: Sequence[Sequence[int]],
    fold_groups: Sequence[Sequence[str]],
    dataset: PrecomputedDataset,
    args: argparse.Namespace,
    output_dir: Path,
    distributed: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    grad_accum_steps: int,
) -> None:
    fold_number = fold_index + 1
    fold_dir = output_dir / f"fold_{fold_number}"
    best_model_path = fold_dir / "best_model.pth"
    complete_path = fold_dir / "fold_complete.json"
    predictions_path = fold_dir / "validation_predictions.csv"
    val_indices = list(folds[fold_index])
    expected_split_hash = validation_split_hash(val_indices)
    if complete_path.exists() and predictions_path.exists() and not args.force:
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            completed.get("split_seed") != args.seed
            or completed.get("split_mode") != args.split_mode
            or completed.get("validation_split_hash") != expected_split_hash
        ):
            raise RuntimeError(
                f"Fold {fold_number} outputs belong to a different split. "
                "Use a new --output-dir or pass --force."
            )
        if rank == 0:
            logger.info("Fold %d is already complete; skipping.", fold_number)
        barrier(distributed)
        return

    if rank == 0:
        fold_dir.mkdir(parents=True, exist_ok=True)
        if args.save_all_epochs:
            (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    barrier(distributed)

    val_set = set(val_indices)
    train_indices = [position for position in range(len(dataset)) if position not in val_set]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed + fold_index,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedEvalSampler(val_dataset, world_size, rank)
        if distributed
        else None
    )
    train_loader = make_loader(
        train_dataset,
        batch_size,
        num_workers,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size,
        num_workers,
        sampler=val_sampler,
        shuffle=False,
        device=device,
    )

    set_seed(args.seed + fold_index)
    model = build_model(device)
    if distributed:
        model.set_parallel_info(rank, world_size)
        ddp_kwargs = {
            "find_unused_parameters": getattr(
                config, "DDP_FIND_UNUSED_PARAMETERS", True
            ),
            "broadcast_buffers": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=getattr(config, "WEIGHT_DECAY", 0.01),
    )
    scheduler = build_scheduler(optimizer, args.epochs)
    use_bf16 = (
        getattr(config, "USE_BF16", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    scaler = None
    if config.MIXED_PRECISION and device.type == "cuda" and not use_bf16:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()

    train_pdbs = set()
    for index in range(len(folds)):
        if index != fold_index:
            train_pdbs.update(fold_groups[index])
    val_pdbs = set(fold_groups[fold_index])
    shared_pdbs = train_pdbs.intersection(val_pdbs)
    if rank == 0:
        logger.info(
            "Fold %d/%d: split_mode=%s train=%d validation=%d "
            "train_unique_pdbs=%d val_unique_pdbs=%d shared_pdbs=%d",
            fold_number,
            len(folds),
            args.split_mode,
            len(train_indices),
            len(val_indices),
            len(train_pdbs),
            len(val_pdbs),
            len(shared_pdbs),
        )

    best_corr = float("-inf")
    best_epoch = 0
    patience_counter = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if rank == 0:
            logger.info("Fold %d - Epoch %d/%d", fold_number, epoch + 1, args.epochs)
        train_loss = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_bf16=use_bf16,
            gradient_accumulation_steps=grad_accum_steps,
            distributed=distributed,
        )
        val_loss, val_corr, _, _ = evaluate_model(
            model, val_loader, criterion, device, distributed=distributed
        )

        if getattr(config, "LR_SCHEDULE", "plateau") == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_corr)
        current_lr = optimizer.param_groups[0]["lr"]
        if rank == 0:
            logger.info(
                "Fold %d Epoch %d: train_loss=%.4f val_loss=%.4f pearson=%.4f lr=%.2e",
                fold_number,
                epoch + 1,
                train_loss,
                val_loss,
                val_corr,
                current_lr,
            )
            if args.save_all_epochs:
                torch.save(
                    get_model_state_dict(model),
                    fold_dir
                    / "checkpoints"
                    / f"epoch_{epoch + 1:03d}_pearson_{val_corr:.4f}.pth",
                )

        if val_corr > best_corr:
            best_corr = float(val_corr)
            best_epoch = epoch + 1
            patience_counter = 0
            if rank == 0:
                torch.save(get_model_state_dict(model), best_model_path)
        else:
            patience_counter += 1
            if rank == 0:
                logger.info(
                    "Fold %d patience: %d/%d",
                    fold_number,
                    patience_counter,
                    args.patience,
                )
        if patience_counter >= args.patience:
            if rank == 0:
                logger.info("Fold %d early stopping triggered.", fold_number)
            break

    barrier(distributed)
    load_state_dict_flexible(model, load_checkpoint(best_model_path, device))
    final_loss, _, labels, predictions = evaluate_model(
        model, val_loader, criterion, device, distributed=distributed
    )

    if distributed:
        assert isinstance(val_sampler, DistributedEvalSampler)
        local_dataset_positions = [val_indices[index] for index in val_sampler.indices]
    else:
        local_dataset_positions = val_indices
    gathered_positions = gather_validation_positions(
        local_dataset_positions, distributed
    )

    if rank == 0:
        metrics = regression_metrics(labels, predictions)
        metrics.update(
            {
                "fold": fold_number,
                "split_mode": args.split_mode,
                "split_seed": args.seed,
                "validation_split_hash": expected_split_hash,
                "best_epoch": best_epoch,
                "best_validation_pearson": best_corr,
                "final_validation_loss": float(final_loss),
                "train_samples": len(train_indices),
                "validation_samples": len(val_indices),
                "train_pdb_groups": len(train_pdbs),
                "validation_pdb_groups": len(val_pdbs),
                "train_unique_pdbs": len(train_pdbs),
                "validation_unique_pdbs": len(val_pdbs),
                "shared_pdbs": len(shared_pdbs),
                "best_model_path": str(best_model_path),
            }
        )
        write_fold_predictions(
            predictions_path,
            dataset,
            gathered_positions,
            labels,
            predictions,
            fold_number,
        )
        complete_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Fold %d complete: %s", fold_number, metrics)

    barrier(distributed)
    del model, optimizer, scheduler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    barrier(distributed)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_cv_summary(output_dir: Path, num_folds: int, dataset_size: int) -> None:
    fold_metrics = []
    oof_rows = []
    for fold_number in range(1, num_folds + 1):
        fold_dir = output_dir / f"fold_{fold_number}"
        fold_metrics.append(
            json.loads((fold_dir / "fold_complete.json").read_text(encoding="utf-8"))
        )
        oof_rows.extend(read_csv_rows(fold_dir / "validation_predictions.csv"))

    split_modes = {row.get("split_mode", "unknown") for row in fold_metrics}
    if len(split_modes) != 1:
        raise RuntimeError(f"Mixed split modes detected in fold outputs: {split_modes}")
    split_mode = next(iter(split_modes))

    oof_rows.sort(key=lambda row: int(row["dataset_position"]))
    observed_positions = [int(row["dataset_position"]) for row in oof_rows]
    if observed_positions != list(range(dataset_size)):
        raise RuntimeError(
            "OOF predictions do not contain every dataset sample exactly once."
        )

    oof_path = output_dir / "oof_predictions.csv"
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof_rows[0].keys()))
        writer.writeheader()
        writer.writerows(oof_rows)

    oof_metrics = regression_metrics(
        [float(row["label_ddg"]) for row in oof_rows],
        [float(row["pred_ddg"]) for row in oof_rows],
    )
    metric_names = ("pearson", "spearman", "mae", "rmse")
    fold_mean = {
        metric: float(np.mean([float(row[metric]) for row in fold_metrics]))
        for metric in metric_names
    }
    fold_std = {
        metric: float(np.std([float(row[metric]) for row in fold_metrics], ddof=1))
        for metric in metric_names
    }
    summary = {
        "num_folds": num_folds,
        "split_mode": split_mode,
        "dataset_samples": dataset_size,
        "fold_metrics": fold_metrics,
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "oof_metrics": oof_metrics,
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    summary_rows = []
    for metrics in fold_metrics:
        summary_rows.append({"scope": f"fold_{metrics['fold']}", **metrics})
    summary_rows.append(
        {
            "scope": "fold_mean",
            **fold_mean,
            "samples": dataset_size,
        }
    )
    summary_rows.append(
        {
            "scope": "fold_std",
            **fold_std,
            "samples": dataset_size,
        }
    )
    summary_rows.append({"scope": "OOF", **oof_metrics})
    fieldnames = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "cv_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    if args.num_folds != 3:
        raise ValueError("This isolated experiment is designed for exactly three folds.")

    distributed, rank, local_rank, world_size, device = setup_distributed()
    output_dir = args.output_dir.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)
    setup_logging(output_dir, rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    batch_size, num_workers, grad_accum_steps = get_runtime_parameters(
        args, world_size
    )
    if rank == 0:
        logger.info(
            "Runtime: device=%s distributed=%s world_size=%d batch/rank=%d "
            "workers/rank=%d grad_accum=%d",
            device,
            distributed,
            world_size,
            batch_size,
            num_workers,
            grad_accum_steps,
        )
        logger.info("Precomputed dataset: %s", args.precomputed_dir)

    dataset = PrecomputedDataset(str(args.precomputed_dir.expanduser().resolve()))
    split_payload = distribute_split_payload(
        dataset, args.num_folds, args.seed, args.split_mode, distributed, rank
    )
    if rank == 0:
        write_split_manifest(split_payload, output_dir)
        logger.info(
            "Split mode: %s; Fold sizes: %s; unique PDBs/fold: %s",
            split_payload["split_mode"],
            [len(fold) for fold in split_payload["folds"]],
            [len(groups) for groups in split_payload["fold_groups"]],
        )
        if split_payload["split_mode"] == "sample":
            logger.info(
                "Sample-level split is active: the same PDB/protein may appear "
                "in both training and validation."
            )
    barrier(distributed)

    if args.split_only:
        if rank == 0:
            logger.info("Split-only validation complete; no models were trained.")
        if distributed:
            torch.distributed.destroy_process_group()
        return

    for fold_index in range(args.num_folds):
        train_fold(
            fold_index=fold_index,
            folds=split_payload["folds"],
            fold_groups=split_payload["fold_groups"],
            dataset=dataset,
            args=args,
            output_dir=output_dir,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            grad_accum_steps=grad_accum_steps,
        )

    if rank == 0:
        write_cv_summary(output_dir, args.num_folds, len(dataset))
        logger.info("Three-fold CV complete. Summary: %s", output_dir / "cv_summary.csv")
    barrier(distributed)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
