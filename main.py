import os
import json
import random

if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
from collections import OrderedDict, defaultdict
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Subset
try:
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:
    def record(fn):
        return fn

import config
from data_loader import (
    SKEMPI2Dataset,
    PrecomputedDataset,
    build_mutation_site_id,
    filter_none_collate,
)
from foldx_processor import FoldXProcessor
from model import ESM_RAAD_FoldX_DDAffinity, ESM_FoldX_DDAffinity
from train_utils import (
    BeneficialGroupRankLoss,
    audit_training_dataloader,
    evaluate_model,
    train_model,
)
from tqdm import tqdm

# 配置日志 
_is_rank_zero_env = int(os.environ.get("RANK", "0")) == 0
_log_handlers = [logging.StreamHandler(sys.stdout)]
if _is_rank_zero_env:
    os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
    _log_handlers.insert(0, logging.FileHandler(config.LOG_PATH))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_distributed():
    """Initialize torchrun/DDP state when RANK/WORLD_SIZE are present."""
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


def cleanup_distributed(distributed):
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def configure_rank_logging(rank):
    if rank == 0:
        return
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.WARNING)


def _format_metric(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:.4f}" if np.isfinite(value) else "n/a"


def log_validation_diagnostics(metrics):
    if not metrics:
        return
    logger.info(
        "Validation loss components: "
        f"beneficial Huber={_format_metric(metrics.get('beneficial_weighted_huber'))} | "
        f"same-complex rank={_format_metric(metrics.get('same_complex_ranking_loss'))} | "
        f"same-site rank={_format_metric(metrics.get('same_site_ranking_loss'))}"
    )
    logger.info(
        "Validation within-site metrics: "
        f"pair accuracy={_format_metric(metrics.get('same_site_pairwise_accuracy'))} | "
        f"Spearman={_format_metric(metrics.get('within_site_spearman'))} | "
        f"NDCG@5={_format_metric(metrics.get('within_site_ndcg_at_5'))} | "
        f"beneficial Recall@1/3/5="
        f"{_format_metric(metrics.get('within_site_beneficial_recall_at_1'))}/"
        f"{_format_metric(metrics.get('within_site_beneficial_recall_at_3'))}/"
        f"{_format_metric(metrics.get('within_site_beneficial_recall_at_5'))}"
    )


def _read_training_dataframe():
    required_columns = [
        "#Pdb",
        "Mutation(s)_cleaned",
        "Affinity_mut_parsed",
        "Affinity_wt_parsed",
    ]
    required = set(required_columns)
    attempts = [
        {"sep": ";"},
        {"sep": None, "engine": "python"},
    ]
    errors = []
    for kwargs in attempts:
        try:
            df = pd.read_csv(config.CSV_PATH, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
            continue
        if required.issubset(df.columns):
            return df.dropna(subset=required_columns).reset_index(drop=True)
        errors.append(f"{kwargs}: missing columns {sorted(required.difference(df.columns))}")
    raise RuntimeError(f"Could not read SKEMPI dataframe for complex split. Attempts: {' | '.join(errors)}")


def _group_value(row, group_col):
    if group_col == "pdb_id":
        return str(row["#Pdb"]).split("_")[0]
    value = row[group_col] if group_col in row and pd.notna(row[group_col]) else row["#Pdb"]
    value = str(value).strip()
    return value or str(row["#Pdb"])


def _dataset_csv_indices(dataset):
    if isinstance(dataset, PrecomputedDataset):
        return list(dataset.indices)
    if hasattr(dataset, "df"):
        return list(dataset.df.index)
    raise TypeError("Complex split requires PrecomputedDataset or a dataset with a .df attribute.")


def _random_split_sizes(dataset_size, fractions):
    raw_sizes = np.asarray(fractions, dtype=float) * int(dataset_size)
    sizes = np.floor(raw_sizes).astype(int)
    remainder = int(dataset_size) - int(sizes.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw_sizes - sizes), kind="stable")
        for index in fractional_order[:remainder]:
            sizes[int(index)] += 1
    return tuple(int(size) for size in sizes)


def _load_frozen_train_val_test_split(
    dataset,
    split_path,
    fractions,
    seed,
    rank,
):
    with open(split_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload_seed = int(payload.get("seed", seed))
    if payload_seed != int(seed):
        raise ValueError(
            f"Frozen split seed {payload_seed} does not match RANDOM_SEED={seed}. "
            "Set RANDOM_SEED to the split seed or select another SPLIT_JSON."
        )
    payload_fractions = (
        float(payload.get("train_fraction", -1.0)),
        float(payload.get("val_fraction", -1.0)),
        float(payload.get("test_fraction", -1.0)),
    )
    if not np.allclose(payload_fractions, fractions):
        raise ValueError(
            f"Frozen split fractions {payload_fractions} do not match "
            f"configured fractions {fractions}."
        )
    if int(payload.get("samples", -1)) != len(dataset):
        raise ValueError(
            f"Frozen split expects {payload.get('samples')} samples, "
            f"but the loaded dataset contains {len(dataset)}."
        )

    partition_indices = {}
    partition_sets = {}
    for partition in ("train", "val", "test"):
        raw_indices = payload.get(f"{partition}_indices")
        if not isinstance(raw_indices, list):
            raise ValueError(
                f"Frozen split is missing a list-valued "
                f"{partition}_indices field: {split_path}"
            )
        indices = [int(index) for index in raw_indices]
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"Frozen split contains duplicate {partition} indices."
            )
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise IndexError(
                f"Frozen split contains out-of-range {partition} indices "
                f"for a dataset of size {len(dataset)}."
            )
        declared_count = int(payload.get(f"{partition}_samples", len(indices)))
        if declared_count != len(indices):
            raise ValueError(
                f"Frozen split declares {declared_count} {partition} samples "
                f"but contains {len(indices)} indices."
            )
        partition_indices[partition] = indices
        partition_sets[partition] = set(indices)

    if partition_sets["train"] & partition_sets["val"]:
        raise RuntimeError("Frozen split has train/validation index overlap.")
    if partition_sets["train"] & partition_sets["test"]:
        raise RuntimeError("Frozen split has train/test index overlap.")
    if partition_sets["val"] & partition_sets["test"]:
        raise RuntimeError("Frozen split has validation/test index overlap.")
    covered_indices = set().union(*partition_sets.values())
    if covered_indices != set(range(len(dataset))):
        missing = sorted(set(range(len(dataset))) - covered_indices)
        raise RuntimeError(
            "Frozen split does not cover the loaded dataset exactly; "
            f"missing indices include {missing[:10]}."
        )

    df = _read_training_dataframe()
    csv_indices = _dataset_csv_indices(dataset)
    rows = df.iloc[csv_indices].copy().reset_index(drop=True)
    group_col = str(
        payload.get(
            "group_col",
            getattr(config, "SPLIT_GROUP_COL", "#Pdb"),
        )
    )
    groups = rows.apply(
        lambda row: _group_value(row, group_col),
        axis=1,
    ).astype(str).to_numpy()
    partition_groups = {
        partition: {
            groups[index]
            for index in partition_indices[partition]
        }
        for partition in ("train", "val", "test")
    }
    if partition_groups["train"] & partition_groups["val"]:
        raise RuntimeError("Frozen split has train/validation group overlap.")
    if partition_groups["train"] & partition_groups["test"]:
        raise RuntimeError("Frozen split has train/test group overlap.")
    if partition_groups["val"] & partition_groups["test"]:
        raise RuntimeError("Frozen split has validation/test group overlap.")

    for partition in ("train", "val", "test"):
        declared_groups = payload.get(f"{partition}_group_values")
        if declared_groups is not None:
            if set(map(str, declared_groups)) != partition_groups[partition]:
                raise ValueError(
                    f"Frozen split {partition}_group_values do not match "
                    "the groups reconstructed from the current CSV."
                )

    if is_main_process(rank):
        logger.info(
            "Loaded frozen complex-grouped 8:1:1 split: "
            f"{split_path} | "
            f"train={len(partition_indices['train'])} samples/"
            f"{len(partition_groups['train'])} groups, "
            f"val={len(partition_indices['val'])} samples/"
            f"{len(partition_groups['val'])} groups, "
            f"test={len(partition_indices['test'])} samples/"
            f"{len(partition_groups['test'])} groups."
        )

    return (
        Subset(dataset, partition_indices["train"]),
        Subset(dataset, partition_indices["val"]),
        Subset(dataset, partition_indices["test"]),
    )


def make_train_val_test_split(
    dataset,
    train_fraction,
    val_fraction,
    test_fraction,
    seed,
    rank,
):
    split_mode = getattr(config, "SPLIT_MODE", "complex").lower()
    fractions = tuple(
        float(value)
        for value in (train_fraction, val_fraction, test_fraction)
    )
    if any(value <= 0.0 for value in fractions):
        raise ValueError(
            "TRAIN_SPLIT, VAL_SPLIT, and TEST_SPLIT must all be positive."
        )
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(
            "TRAIN_SPLIT, VAL_SPLIT, and TEST_SPLIT must sum to 1.0; "
            f"got {fractions} (sum={sum(fractions):.6f})."
        )
    if len(dataset) < 3:
        raise ValueError(
            f"Train/validation/test split requires at least 3 samples, got {len(dataset)}."
        )

    frozen_split_path = getattr(config, "SPLIT_JSON", "")
    if frozen_split_path:
        if not os.path.isfile(frozen_split_path):
            raise FileNotFoundError(
                f"Configured SPLIT_JSON was not found: {frozen_split_path}"
            )
        return _load_frozen_train_val_test_split(
            dataset=dataset,
            split_path=frozen_split_path,
            fractions=fractions,
            seed=seed,
            rank=rank,
        )

    if split_mode == "random":
        train_size, val_size, test_size = _random_split_sizes(
            len(dataset),
            fractions,
        )
        if min(train_size, val_size, test_size) < 1:
            raise ValueError(
                "The configured 8:1:1 split produced an empty random subset: "
                f"train={train_size}, val={val_size}, test={test_size}."
            )
        split_generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=split_generator,
        )
        if is_main_process(rank):
            logger.warning(
                "Using random 8:1:1 split. This is not complex-level evaluation."
            )
        return train_dataset, val_dataset, test_dataset

    if split_mode not in {"complex", "group"}:
        raise ValueError(f"Unsupported SPLIT_MODE={split_mode!r}; expected complex, group, or random.")

    df = _read_training_dataframe()
    csv_indices = _dataset_csv_indices(dataset)
    if csv_indices and max(csv_indices) >= len(df):
        raise IndexError(
            f"Dataset index {max(csv_indices)} exceeds CSV rows {len(df)}. "
            "Regenerate precomputed samples from the same CSV_PATH."
        )
    rows = df.iloc[csv_indices].copy().reset_index(drop=True)
    group_col = getattr(config, "SPLIT_GROUP_COL", "#Pdb")
    groups = rows.apply(lambda row: _group_value(row, group_col), axis=1).astype(str).to_numpy()
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 3:
        raise ValueError(f"Complex 8:1:1 split needs at least 3 groups, got {len(unique_groups)}.")

    holdout_fraction = val_fraction + test_fraction
    train_holdout_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=float(train_fraction),
        test_size=float(holdout_fraction),
        random_state=seed,
    )
    all_indices = np.arange(len(dataset))
    train_idx, holdout_idx = next(
        train_holdout_splitter.split(all_indices, groups=groups)
    )

    holdout_groups = groups[holdout_idx]
    relative_val_fraction = float(val_fraction / holdout_fraction)
    relative_test_fraction = float(test_fraction / holdout_fraction)
    val_test_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_fraction,
        test_size=relative_test_fraction,
        random_state=seed + 1,
    )
    val_local_idx, test_local_idx = next(
        val_test_splitter.split(holdout_idx, groups=holdout_groups)
    )
    val_idx = holdout_idx[val_local_idx]
    test_idx = holdout_idx[test_local_idx]

    train_groups = {groups[int(i)] for i in train_idx}
    val_groups = {groups[int(i)] for i in val_idx}
    test_groups = {groups[int(i)] for i in test_idx}
    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError(
            "Complex split group leakage detected: "
            f"train/val={sorted(train_val_overlap)[:10]}, "
            f"train/test={sorted(train_test_overlap)[:10]}, "
            f"val/test={sorted(val_test_overlap)[:10]}"
        )

    if is_main_process(rank):
        split_dir = getattr(
            config,
            "SPLIT_DIR",
            os.path.join(config.VARIANT_DIR, "splits"),
        )
        os.makedirs(split_dir, exist_ok=True)
        ratio_tag = getattr(config, "SPLIT_RATIO_TAG", "80_10_10")
        split_path = os.path.join(
            split_dir,
            f"complex_split_{ratio_tag}_seed{seed}.json",
        )
        payload = {
            "split_mode": split_mode,
            "group_col": group_col,
            "seed": seed,
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "test_fraction": float(test_fraction),
            "samples": len(dataset),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
            "train_val_overlap_groups": 0,
            "train_test_overlap_groups": 0,
            "val_test_overlap_groups": 0,
            "train_indices": train_idx.astype(int).tolist(),
            "val_indices": val_idx.astype(int).tolist(),
            "test_indices": test_idx.astype(int).tolist(),
            "train_group_values": sorted(train_groups),
            "val_group_values": sorted(val_groups),
            "test_group_values": sorted(test_groups),
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Using complex 8:1:1 split: "
            f"group_col={group_col}, train={len(train_idx)} samples/{len(train_groups)} groups, "
            f"val={len(val_idx)} samples/{len(val_groups)} groups, "
            f"test={len(test_idx)} samples/{len(test_groups)} groups. Saved {split_path}"
        )

    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


def _dataset_group_metadata(dataset, rank=0):
    """Return complex/site ids and labels aligned to local dataset indices."""
    if isinstance(dataset, Subset):
        parent = dataset.dataset
        parent_csv_indices = _dataset_csv_indices(parent)
        csv_indices = [parent_csv_indices[int(idx)] for idx in dataset.indices]
    else:
        csv_indices = _dataset_csv_indices(dataset)

    df = _read_training_dataframe()
    if csv_indices and max(csv_indices) >= len(df):
        raise IndexError(
            f"Dataset index {max(csv_indices)} exceeds CSV rows {len(df)} while "
            "building group-aware batches."
        )

    rows = df.iloc[csv_indices]
    complex_ids = rows["#Pdb"].astype(str).tolist()
    site_ids = [
        build_mutation_site_id(complex_id, mutation)
        for complex_id, mutation in zip(
            complex_ids,
            rows["Mutation(s)_cleaned"].astype(str).tolist(),
        )
    ]
    wt_affinity = pd.to_numeric(
        rows["Affinity_wt_parsed"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    mutant_affinity = pd.to_numeric(
        rows["Affinity_mut_parsed"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    valid_affinity = (
        np.isfinite(wt_affinity)
        & np.isfinite(mutant_affinity)
        & (wt_affinity > 0)
        & (mutant_affinity > 0)
    )
    labels = np.full(len(rows), np.nan, dtype=np.float64)
    labels[valid_affinity] = (
        (8.314 / 4184)
        * (273.15 + 25.0)
        * (
            np.log(mutant_affinity[valid_affinity])
            - np.log(wt_affinity[valid_affinity])
        )
    )

    fallback_positions = np.flatnonzero(~valid_affinity).tolist()
    for local_position in fallback_positions:
        sample = dataset[int(local_position)]
        cached_label = sample.get("delta_g")
        if isinstance(cached_label, torch.Tensor):
            cached_label = cached_label.detach().cpu().reshape(-1)
            if cached_label.numel() != 1:
                raise ValueError(
                    "Cached delta_g must contain exactly one value for "
                    f"dataset position {local_position}."
                )
            cached_label = cached_label.item()
        labels[local_position] = float(cached_label)

    if not np.isfinite(labels).all():
        bad_positions = np.flatnonzero(~np.isfinite(labels)).tolist()
        raise ValueError(
            "Group-aware batching found non-finite training labels at local "
            f"dataset positions {bad_positions[:20]}."
        )
    if fallback_positions and is_main_process(rank):
        logger.warning(
            "Used cached delta_g for "
            f"{len(fallback_positions)} sample(s) whose raw affinity values "
            "could not be parsed as finite positive numbers."
        )
    return complex_ids, site_ids, labels.tolist()


class DistributedEvalSampler(Sampler):
    """Shard evaluation data without padding or duplicating samples."""

    def __init__(self, dataset, num_replicas, rank):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas.")
        self.dataset_size = len(dataset)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.num_replicas))

    def __len__(self):
        remaining = max(self.dataset_size - self.rank, 0)
        return (remaining + self.num_replicas - 1) // self.num_replicas


class GroupAwareDistributedBatchSampler:
    """Build group-aware batches and guarantee a valid same-site rank pair."""

    def __init__(
        self,
        complex_ids,
        site_ids,
        labels,
        batch_size,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        drop_last=False,
        pairwise_min_label_gap=0.2,
    ):
        if not (len(complex_ids) == len(site_ids) == len(labels)):
            raise ValueError(
                "complex_ids, site_ids, and labels must have the same length."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas.")

        self.site_ids = [str(site_id) for site_id in site_ids]
        self.labels = [float(label) for label in labels]
        if not np.isfinite(np.asarray(self.labels, dtype=np.float64)).all():
            raise ValueError("Group-aware sampler labels must be finite.")
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.pairwise_min_label_gap = float(pairwise_min_label_gap)
        self.valid_pair_gap = self.pairwise_min_label_gap + 1e-6
        self.epoch = 0

        grouped = defaultdict(lambda: defaultdict(list))
        for idx, (complex_id, site_id) in enumerate(zip(complex_ids, site_ids)):
            grouped[str(complex_id)][str(site_id)].append(idx)
        self.grouped_indices = {
            complex_id: dict(site_groups)
            for complex_id, site_groups in grouped.items()
        }
        self.valid_anchor_pairs = self._build_valid_anchor_pairs()

        base_batches = self._build_base_batches(random.Random(self.seed))
        initial_batches, rescue_stats = self._rescue_invalid_batches(
            base_batches,
            random.Random(self.seed),
        )
        if not initial_batches:
            raise ValueError("Group-aware sampler received an empty dataset.")
        self.batch_templates = [list(batch) for batch in initial_batches]
        self.base_batch_count = len(base_batches)
        self.natural_valid_batch_count = rescue_stats["natural_valid_batches"]
        self.rescued_batch_count = rescue_stats["rescued_batches"]
        self.repeated_anchor_sample_count = rescue_stats[
            "repeated_anchor_samples"
        ]
        self.num_batches_per_rank = (
            len(initial_batches) + self.num_replicas - 1
        ) // self.num_replicas

    def _build_valid_anchor_pairs(self):
        anchor_pairs = []
        site_groups = defaultdict(list)
        for index, site_id in enumerate(self.site_ids):
            site_groups[site_id].append(index)
        for indices in site_groups.values():
            if len(indices) < 2:
                continue
            ordered = sorted(indices, key=lambda index: self.labels[index])
            low_index = ordered[0]
            high_index = ordered[-1]
            if (
                abs(self.labels[high_index] - self.labels[low_index])
                > self.valid_pair_gap
            ):
                anchor_pairs.append((low_index, high_index))
        return anchor_pairs

    def _batch_has_valid_same_site_pair(self, batch):
        label_ranges = {}
        for index in batch:
            site_id = self.site_ids[index]
            label = self.labels[index]
            if site_id not in label_ranges:
                label_ranges[site_id] = [label, label]
            else:
                label_ranges[site_id][0] = min(
                    label_ranges[site_id][0],
                    label,
                )
                label_ranges[site_id][1] = max(
                    label_ranges[site_id][1],
                    label,
                )
        return any(
            maximum - minimum > self.valid_pair_gap
            for minimum, maximum in label_ranges.values()
        )

    def _rescue_invalid_batches(self, batches, rng):
        natural_valid_batches = []
        orphan_indices = []
        for batch in batches:
            if self._batch_has_valid_same_site_pair(batch):
                natural_valid_batches.append(list(batch))
            else:
                orphan_indices.extend(batch)

        if not orphan_indices:
            return natural_valid_batches, {
                "natural_valid_batches": len(natural_valid_batches),
                "rescued_batches": 0,
                "repeated_anchor_samples": 0,
            }
        if self.batch_size < 3:
            raise ValueError(
                "At least three samples per batch are required to inject "
                "a valid same-site anchor pair."
            )
        if not self.valid_anchor_pairs:
            raise ValueError(
                "No same-site pair exceeds pairwise_min_label_gap; "
                "same-site ranking cannot be trained on this split."
            )

        anchors = list(self.valid_anchor_pairs)
        if self.shuffle:
            rng.shuffle(anchors)
        payload_size = self.batch_size - 2
        rescued_batches = []
        for start in range(0, len(orphan_indices), payload_size):
            payload = orphan_indices[start : start + payload_size]
            anchor_pair = anchors[
                len(rescued_batches) % len(anchors)
            ]
            rescued = list(anchor_pair) + payload
            if self.drop_last and len(rescued) < self.batch_size:
                fill_source = list(anchor_pair)
                while len(rescued) < self.batch_size:
                    rescued.append(
                        fill_source[
                            (len(rescued) - len(payload)) % len(fill_source)
                        ]
                    )
            rescued_batches.append(rescued)

        combined = natural_valid_batches + rescued_batches
        if self.shuffle:
            rng.shuffle(combined)
        return combined, {
            "natural_valid_batches": len(natural_valid_batches),
            "rescued_batches": len(rescued_batches),
            "repeated_anchor_samples": 2 * len(rescued_batches),
        }

    def _build_base_batches(self, rng):
        complex_keys = list(self.grouped_indices)
        if self.shuffle:
            rng.shuffle(complex_keys)

        full_batches = []
        remainders = []
        for complex_id in complex_keys:
            site_groups = self.grouped_indices[complex_id]
            site_keys = list(site_groups)
            if self.shuffle:
                rng.shuffle(site_keys)

            ordered_indices = []
            for site_id in site_keys:
                site_indices = list(site_groups[site_id])
                if self.shuffle:
                    rng.shuffle(site_indices)
                ordered_indices.extend(site_indices)

            full_count = len(ordered_indices) // self.batch_size
            for batch_idx in range(full_count):
                start = batch_idx * self.batch_size
                full_batches.append(ordered_indices[start : start + self.batch_size])
            remainder = ordered_indices[full_count * self.batch_size :]
            if remainder:
                remainders.append(remainder)

        if self.shuffle:
            rng.shuffle(remainders)

        mixed_batch = []
        for remainder in remainders:
            while remainder:
                room = self.batch_size - len(mixed_batch)
                mixed_batch.extend(remainder[:room])
                remainder = remainder[room:]
                if len(mixed_batch) == self.batch_size:
                    full_batches.append(mixed_batch)
                    mixed_batch = []
        if mixed_batch and not self.drop_last:
            full_batches.append(mixed_batch)

        if self.shuffle:
            rng.shuffle(full_batches)
        return full_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = [list(batch) for batch in self.batch_templates]
        if self.shuffle:
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        total_size = self.num_batches_per_rank * self.num_replicas
        if len(batches) < total_size:
            batches.extend(
                [list(batches[idx % len(batches)]) for idx in range(total_size - len(batches))]
            )
        elif len(batches) > total_size:
            batches = batches[:total_size]
        yield from batches[self.rank:total_size:self.num_replicas]

    def __len__(self):
        return self.num_batches_per_rank

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def get_num_workers(world_size):
    if world_size <= 1:
        return config.NUM_WORKERS
    return getattr(config, "DDP_NUM_WORKERS_PER_RANK", max(1, config.NUM_WORKERS // world_size))


def get_batch_size(world_size):
    if world_size <= 1:
        return config.BATCH_SIZE
    return getattr(config, "DDP_BATCH_SIZE_PER_RANK", max(1, config.BATCH_SIZE // world_size))


def get_grad_accum_steps(world_size, batch_size_per_process):
    if world_size <= 1:
        return config.GRADIENT_ACCUMULATION_STEPS
    configured_steps = getattr(config, "DDP_GRADIENT_ACCUMULATION_STEPS", None)
    if configured_steps is not None:
        return configured_steps
    target_global_batch = config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    global_micro_batch = max(1, batch_size_per_process * world_size)
    return max(1, round(target_global_batch / global_micro_batch))


def load_state_dict_flexible(model, checkpoint):
    """Load plain or DataParallel/DDP-prefixed weights into the current wrapper."""
    target = model.module if hasattr(model, "module") else model
    try:
        target.load_state_dict(checkpoint)
        return
    except RuntimeError:
        pass

    stripped = OrderedDict((k.replace("module.", "", 1), v) for k, v in checkpoint.items())
    try:
        target.load_state_dict(stripped)
        return
    except RuntimeError:
        pass

    prefixed = OrderedDict(
        (k if k.startswith("module.") else f"module.{k}", v) for k, v in checkpoint.items()
    )
    model.load_state_dict(prefixed)


def get_model_state_dict(model):
    target = model.module if hasattr(model, "module") else model
    return target.state_dict()


def collect_unique_pdb_paths(df, pdb_dir):
    # "#Pdb" 形如 "3SE4_A_B"，只取文件名部分 "3SE4"
    pdb_ids = df["#Pdb"].apply(lambda s: s.split("_")[0]).unique().tolist()
    pdb_paths = [os.path.abspath(os.path.join(pdb_dir, f"{pid}.pdb")) for pid in pdb_ids]
    return pdb_ids, pdb_paths

def offline_repair_all_pdbs(dataset, max_items=None):
    """
    在训练开始前调用。
    返回:
      repaired_map: dict[pdb_id] = repaired_pdb_path
      failed_ids:   set[pdb_id]  (修复失败或超时)
    """
    df = dataset.df
    pdb_dir = dataset.pdb_dir
    foldx = dataset.foldx_processor  # 你已有的处理器

    pdb_ids, pdb_paths = collect_unique_pdb_paths(df, pdb_dir)
    if max_items:
        pdb_ids  = pdb_ids[:max_items]
        pdb_paths = pdb_paths[:max_items]

    repaired_map = {}
    failed_ids = set()

    pbar = tqdm(zip(pdb_ids, pdb_paths), total=len(pdb_ids), desc="Pre-repair PDBs", unit="file")
    for pid, path in pbar:
        # 已存在就直接复用（避免多次修）
        repaired_name = f"{pid}_Repair.pdb"
        candidate = os.path.join(foldx.temp_dir, repaired_name)
        if os.path.exists(candidate):
            repaired_map[pid] = candidate
            continue
        # else:
        #     failed_ids.add(pid)

        # 调用 FoldX 修复（你的 preprocess_pdb 里已经有缓存/锁/超时的话更好）
        try:
            repaired_path = foldx.preprocess_pdb(path)
            repaired_map[pid] = repaired_path
        except Exception as e:
            failed_ids.add(pid)
            # 可选：打印一行简洁错误用于排查
            logger.warning(f"Repair failed for {pid}: {repr(e)}")

    return repaired_map, failed_ids

@record
def main():
    distributed, rank, local_rank, world_size, device = setup_distributed()
    configure_rank_logging(rank)
    num_workers = get_num_workers(world_size)
    batch_size = get_batch_size(world_size)
    grad_accum_steps = get_grad_accum_steps(world_size, batch_size)
    if is_main_process(rank):
        logger.info(
            f"Runtime device: {device} | distributed={distributed} | world_size={world_size}"
        )
        logger.info(
            f"Batch size per process: {batch_size} | "
            f"grad accumulation: {grad_accum_steps} | workers per process: {num_workers}"
        )
        if config.NUM_GPUS > 1 and not distributed:
            logger.warning(
                "Multiple GPUs are visible, but DDP is not initialized. "
                f"Launch with: torchrun --standalone --nproc_per_node={config.NUM_GPUS} main.py"
            )

    precomputed_dir = config.PRECOMPUTED_DIR
    precomputed_pt_files = (
        [name for name in os.listdir(precomputed_dir) if name.endswith(".pt")]
        if os.path.isdir(precomputed_dir)
        else []
    )

    if precomputed_pt_files:
        logger.info(f"Using precomputed dataset from {precomputed_dir}")
        validate_precomputed_cache = getattr(
            config,
            "VALIDATE_PRECOMPUTED_CACHE_ON_LOAD",
            False,
        )
        dataset = PrecomputedDataset(
            precomputed_dir,
            validate_current_config=validate_precomputed_cache,
        )
        if is_main_process(rank) and not validate_precomputed_cache:
            logger.info(
                "Skipped per-sample precomputed cache validation; "
                "training audit remains enabled."
            )
        if len(dataset) == 0:
            raise RuntimeError(
                "No valid precomputed samples match the current FoldX/ESM settings. "
                "Run this folder's precompute_samples.py and precompute_esm_embeddings.py."
            )
    else:
        if getattr(config, "USE_PRECOMPUTED_ESM", False):
            raise RuntimeError(
                f"No .pt precomputed samples found in {precomputed_dir}. "
                "This experiment uses USE_PRECOMPUTED_ESM=True, so it cannot fall back to online "
                "SKEMPI2Dataset loading. Run precompute_mutation_foldx.py if mutation FoldX JSON is "
                "missing, then run this folder's precompute_samples.py and precompute_esm_embeddings.py."
            )
        logger.info("Initializing FoldX Processor...")
        foldx_processor = FoldXProcessor(
            foldx_path=config.FOLDX_PATH, temp_dir=config.FOLDX_TEMP_DIR, cache_dir=config.FOLDX_CACHE_DIR
        )
        logger.info("Loading and preprocessing dataset...")
        try:
            dataset = SKEMPI2Dataset(
                csv_path=config.CSV_PATH,
                pdb_dir=config.PDB_DIR,
                foldx_processor=foldx_processor,
                use_structure=config.USE_STRUCTURE_FEATURES,
            )
        except Exception as e:
            raise RuntimeError("Failed to load dataset") from e

    logger.info(f"Initial dataset size: {len(dataset)}")

    train_dataset, val_dataset, test_dataset = make_train_val_test_split(
        dataset,
        train_fraction=config.TRAIN_SPLIT,
        val_fraction=getattr(config, "VAL_SPLIT", 0.1),
        test_fraction=getattr(config, "TEST_SPLIT", 0.1),
        seed=getattr(config, "RANDOM_SEED", 42),
        rank=rank,
    )

    logger.info(
        f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, "
        f"Test size: {len(test_dataset)}"
    )

    use_group_batches = getattr(config, "GROUP_AWARE_BATCHING", True)
    if use_group_batches:
        complex_ids, site_ids, group_labels = _dataset_group_metadata(
            train_dataset,
            rank=rank,
        )
        train_sampler = GroupAwareDistributedBatchSampler(
            complex_ids=complex_ids,
            site_ids=site_ids,
            labels=group_labels,
            batch_size=batch_size,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=True,
            seed=getattr(config, "RANDOM_SEED", 42),
            pairwise_min_label_gap=getattr(
                config,
                "PAIRWISE_MIN_LABEL_GAP",
                0.2,
            ),
        )
        if is_main_process(rank):
            logger.info(
                "Using group-aware batches: "
                f"{len(set(complex_ids))} complexes, {len(set(site_ids))} sites, "
                f"{len(train_sampler)} batches per rank | "
                f"natural valid batches={train_sampler.natural_valid_batch_count}/"
                f"{train_sampler.base_batch_count} | "
                f"rescued batches={train_sampler.rescued_batch_count} | "
                "repeated anchor samples="
                f"{train_sampler.repeated_anchor_sample_count}"
            )
    else:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        ) if distributed else None
    val_sampler = DistributedEvalSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
    ) if distributed else None
    test_sampler = DistributedEvalSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
    ) if distributed else None

    train_loader_kwargs = {
        "collate_fn": filter_none_collate,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 4
    if use_group_batches:
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            **train_loader_kwargs,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **train_loader_kwargs,
        )
    val_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": val_sampler,
        "collate_fn": filter_none_collate,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 4
    val_dataloader = DataLoader(val_dataset, **val_loader_kwargs)
    test_loader_kwargs = {
        **val_loader_kwargs,
        "sampler": test_sampler,
    }
    test_dataloader = DataLoader(test_dataset, **test_loader_kwargs)

    if getattr(config, "RUN_TRAINING_AUDIT", True):
        audit_result = audit_training_dataloader(
            train_dataloader,
            device=device,
            distributed=distributed,
            beneficial_threshold=getattr(config, "BENEFICIAL_THRESHOLD", 0.0),
            pairwise_min_label_gap=getattr(
                config,
                "PAIRWISE_MIN_LABEL_GAP",
                0.2,
            ),
            max_zero_same_site_batch_rate=getattr(
                config,
                "MAX_ZERO_SAME_SITE_BATCH_RATE",
                0.20,
            ),
        )
        if is_main_process(rank):
            with open(config.TRAINING_AUDIT_PATH, "w", encoding="utf-8") as handle:
                json.dump(audit_result, handle, indent=2)
            logger.info(f"Training audit saved to {config.TRAINING_AUDIT_PATH}")
        if distributed:
            torch.distributed.barrier()
        if not audit_result["passed"]:
            raise RuntimeError(
                "Training audit failed; training was not started: "
                + "; ".join(audit_result["failure_reasons"])
            )

    logger.info(f"Using device: {device}")
    
    if config.USE_DYNAMIC_MODELING:
        logger.info("Using ESM-RAAD-FoldX dynamic modeling architecture")
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
            use_precomputed_esm=getattr(config, 'USE_PRECOMPUTED_ESM', False),
            local_radius=getattr(config, 'MUTATION_LOCAL_RADIUS', 10.0),
            esm_mutation_window_radius=getattr(config, 'ESM_MUTATION_WINDOW_RADIUS', 8),
            esm_local_max_tokens=getattr(config, "ESM_LOCAL_MAX_TOKENS", 32),
            struct_local_max_residues=getattr(
                config,
                "STRUCT_LOCAL_MAX_RESIDUES",
                32,
            ),
            coords_agg=getattr(config, "COORDS_AGG", "mean"),
        ).to(device)
    else:
        logger.info("Using original ESM-FoldX architecture")
        model = ESM_FoldX_DDAffinity(
            esm_model_name=config.ESM_MODEL_NAME,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
        ).to(device)

    # 多卡支持
    if distributed:
        logger.info(f"Using {world_size} GPUs with DistributedDataParallel")
        model.set_parallel_info(rank, world_size)
        ddp_kwargs = {
            "find_unused_parameters": getattr(config, "DDP_FIND_UNUSED_PARAMETERS", True),
            "broadcast_buffers": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        model = DDP(model, **ddp_kwargs)
    elif config.NUM_GPUS > 1 and getattr(config, "PARALLEL_BACKEND", "ddp") == "dataparallel":
        logger.warning("Using legacy DataParallel; DDP is faster for this project.")
        model = torch.nn.DataParallel(model)
        model.module.set_parallel_info(0, config.NUM_GPUS)
        # 设置DataParallel的进程信息，让模型知道如何分割数据
        model.module.set_parallel_info(0, config.NUM_GPUS)
    # torch.compile 加速（Blackwell 架构受益显著）
    if getattr(config, 'TORCH_COMPILE', False) and device.type == 'cuda' and not distributed:
        try:
            base_model = model.module if hasattr(model, 'module') else model
            base_model = torch.compile(base_model, mode="reduce-overhead")
            if hasattr(model, 'module'):
                model.module = base_model
            else:
                model = base_model
            logger.info("torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back to eager mode: {e}")

    loss_function = getattr(config, "LOSS_FUNCTION", "mse").lower()
    if loss_function == "beneficial_group_rank":
        criterion = BeneficialGroupRankLoss(
            smooth_l1_beta=getattr(config, "SMOOTH_L1_BETA", 1.0),
            beneficial_threshold=getattr(config, "BENEFICIAL_THRESHOLD", 0.0),
            beneficial_sample_weight=getattr(config, "BENEFICIAL_SAMPLE_WEIGHT", 2.0),
            pairwise_weight=getattr(config, "PAIRWISE_RANK_WEIGHT", 0.25),
            site_pair_weight=getattr(config, "SITE_PAIR_WEIGHT", 1.0),
            complex_pair_weight=getattr(config, "COMPLEX_PAIR_WEIGHT", 0.25),
            pairwise_temperature=getattr(config, "PAIRWISE_TEMPERATURE", 0.5),
            pairwise_min_label_gap=getattr(config, "PAIRWISE_MIN_LABEL_GAP", 0.2),
            pairwise_max_pairs=getattr(config, "PAIRWISE_MAX_PAIRS", 4096),
        )
        logger.info(
            "Using BeneficialGroupRankLoss: "
            "beneficial-weighted SmoothL1 + "
            f"{getattr(config, 'PAIRWISE_RANK_WEIGHT', 0.25)}*same-group rank; "
            f"beta={getattr(config, 'SMOOTH_L1_BETA', 1.0)}, "
            f"beneficial_threshold={getattr(config, 'BENEFICIAL_THRESHOLD', 0.0)}, "
            f"beneficial_weight={getattr(config, 'BENEFICIAL_SAMPLE_WEIGHT', 2.0)}, "
            f"site_pair_weight={getattr(config, 'SITE_PAIR_WEIGHT', 1.0)}, "
            f"complex_pair_weight={getattr(config, 'COMPLEX_PAIR_WEIGHT', 0.25)}, "
            f"temperature={getattr(config, 'PAIRWISE_TEMPERATURE', 0.5)}, "
            f"min_gap={getattr(config, 'PAIRWISE_MIN_LABEL_GAP', 0.2)}, "
            f"max_pairs={getattr(config, 'PAIRWISE_MAX_PAIRS', 4096)}"
        )
    elif loss_function in {"smooth_l1", "smooth_l1_pairwise_rank"}:
        criterion = nn.SmoothL1Loss(beta=getattr(config, "SMOOTH_L1_BETA", 1.0))
        logger.info(f"Using SmoothL1Loss(beta={getattr(config, 'SMOOTH_L1_BETA', 1.0)})")
        if loss_function == "smooth_l1_pairwise_rank":
            logger.info(
                "Using pairwise ranking loss add-on: "
                f"weight={getattr(config, 'RANK_LOSS_WEIGHT', 0.1)}, "
                f"margin={getattr(config, 'RANK_MARGIN', 0.1)}, "
                f"max_pairs={getattr(config, 'RANK_MAX_PAIRS', 512)}, "
                f"min_label_diff={getattr(config, 'RANK_MIN_LABEL_DIFF', 0.05)}"
            )
    else:
        criterion = nn.MSELoss()
        logger.info("Using MSELoss")
    selection_metric = getattr(config, "MODEL_SELECTION_METRIC", "val_loss").lower()
    if selection_metric not in {"val_loss", "pearson"}:
        raise ValueError(
            f"Unsupported MODEL_SELECTION_METRIC={selection_metric!r}; "
            "expected 'val_loss' or 'pearson'."
        )
    weight_decay = getattr(config, 'WEIGHT_DECAY', 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=weight_decay)

    # 学习率调度：warmup + cosine annealing
    if getattr(config, 'LR_SCHEDULE', 'plateau') == 'cosine':
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 10)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性 warmup: 从 1/warmup_epochs 增长到 1.0
                return (epoch + 1) / warmup_epochs
            else:
                # cosine annealing: 从 1.0 衰减到 min_lr/lr
                progress = (epoch - warmup_epochs) / max(config.NUM_EPOCHS - warmup_epochs, 1)
                min_ratio = config.MIN_LR / config.LEARNING_RATE
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info(f"Using cosine annealing with {warmup_epochs}-epoch warmup")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if selection_metric == "val_loss" else "max",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE, min_lr=config.MIN_LR
        )
    
    scaler = None
    use_bf16 = getattr(config, 'USE_BF16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if config.MIXED_PRECISION and device.type == 'cuda':
        if use_bf16:
            # BF16 不需要 GradScaler，直接用 autocast(dtype=bfloat16)
            logger.info("Mixed precision training enabled (BF16 — no GradScaler needed)")
        else:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            logger.info("Mixed precision training enabled (FP16 with GradScaler)")

    logger.info("Starting training...")
    best_selection_value = (
        float("inf") if selection_metric == "val_loss" else float("-inf")
    )
    logger.info(f"Best-model selection metric: {selection_metric}")
    validation_history = []
    patience_counter = 0  # 早停计数器
    
    if is_main_process(rank):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        logger.info(f"--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
        
        train_loss = train_model(
            model, train_dataloader, criterion, optimizer, device, scaler,
            use_bf16=use_bf16,
            gradient_accumulation_steps=grad_accum_steps,
            distributed=distributed,
        )
        
        val_loss, val_corr, _, _, val_metrics = evaluate_model(
            model,
            val_dataloader,
            criterion,
            device,
            distributed=distributed,
            return_diagnostics=True,
        )

        logger.info(
            f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Pearson Corr: {val_corr:.4f}"
        )
        log_validation_diagnostics(val_metrics)
        selection_value = val_loss if selection_metric == "val_loss" else val_corr

        # Step LR scheduler
        if getattr(config, 'LR_SCHEDULE', 'plateau') == 'cosine':
            scheduler.step()
        else:
            scheduler.step(selection_value)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current LR: {current_lr:.2e}")

        if is_main_process(rank):
            validation_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_pearson": val_corr,
                    "learning_rate": current_lr,
                    **val_metrics,
                }
            )
            pd.DataFrame(validation_history).to_csv(
                config.VALIDATION_METRICS_PATH,
                index=False,
            )
            epoch_model_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"epoch_{epoch + 1:03d}_pearson_{val_corr:.4f}.pth",
            )
            torch.save(get_model_state_dict(model), epoch_model_path)
            logger.info(f"Epoch model saved to {epoch_model_path}")

        improved = (
            selection_value < best_selection_value
            if selection_metric == "val_loss"
            else selection_value > best_selection_value
        )
        if improved:
            best_selection_value = selection_value
            # 保存模型时处理DataParallel情况
            if is_main_process(rank):
                torch.save(
                    get_model_state_dict(model),
                    config.TRAINED_MODEL_PATH,
                )
                logger.info(
                    f"New best model saved with {selection_metric}: "
                    f"{best_selection_value:.4f} "
                    f"(Pearson={val_corr:.4f}, Val Loss={val_loss:.4f})"
                )
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            logger.info(f"Validation metric did not improve. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
            
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered. Training stopped.")
                break

    logger.info("--- Final Test Evaluation on Best Model ---")
    # 加载模型时处理DataParallel情况
    if distributed:
        torch.distributed.barrier()
    load_state_dict_flexible(
        model,
        torch.load(config.TRAINED_MODEL_PATH, map_location=device),
    )
    test_loss, test_corr, labels, preds, test_metrics = evaluate_model(
        model,
        test_dataloader,
        criterion,
        device,
        distributed=distributed,
        return_diagnostics=True,
    )
    logger.info(
        f"Final Test Loss: {test_loss:.4f} | Final Test Pearson Corr: {test_corr:.4f}"
    )
    log_validation_diagnostics(test_metrics)

    if is_main_process(rank):
        plt.figure(figsize=(8, 8))
        plt.scatter(labels, preds, alpha=0.5)
        min_val = min(np.min(labels), np.min(preds))
        max_val = max(np.max(labels), np.max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.title(f"Test Predicted vs. True ddG (Pearson: {test_corr:.4f})")
        plt.xlabel("True ddG")
        plt.ylabel("Predicted ddG")
        plt.grid(True)
        plt.savefig(config.PLOT_PATH)
        logger.info(f"Saved prediction scatterplot to {config.PLOT_PATH}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
